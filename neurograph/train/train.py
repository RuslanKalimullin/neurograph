import inspect
import json
import logging
from typing import Type

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn

from neurograph.config import Config, ModelConfig
from neurograph.data.datasets import NeuroDataset
from neurograph.models import GAT
from neurograph.models.available_modules import available_optimizers, available_losses


def train(ds: NeuroDataset, cfg: Config):
    # metrics  # TODO put into dataclass
    accs, aucs, macros, valid_losses = [], [], [], []

    test_loader = ds.get_test_loader(cfg.train.valid_batch_size)

    # run training for each fold
    for fold_i, loaders_dict in enumerate(ds.get_cv_loaders(cfg.train.batch_size, cfg.train.valid_batch_size)):
        logging.info(f'Fold {fold_i}')

        train_loader = loaders_dict['train']
        valid_loader = loaders_dict['valid']

        # create model instance
        model = create_model(ds, cfg.model)

        # set optimizer
        optimizer = available_optimizers[cfg.train.optim](
            model.parameters(),
            **cfg.train.optim_args if cfg.train.optim_args else {}
        )
        # set loss function
        loss_f = available_losses[cfg.train.loss](**cfg.train.loss_args if cfg.train.loss_args else {})

        # train and return valid metrics
        val_acc, val_auc, val_macro, valid_loss = train_one_split(
            model,
            train_loader,
            valid_loader,
            optimizer,
            loss_f=loss_f,
            device=cfg.train.device,
            cfg=cfg,
        )

        # print valid metrics
        logging.info(f'(Last Epoch) | valid_loss={valid_loss},  val_acc={(val_acc * 100):.2f}, '
                     f'val_macro={(val_macro * 100):.2f}, val_auc={(val_auc * 100):.2f}')

        # eval on test  # TODO
        test_acc, test_auc, test_macro, test_loss = evaluate(model, test_loader, loss_f, cfg)

        logging.info(f'(Last Epoch) | test_loss={test_loss},  test_acc={(test_acc * 100):.2f}, '
                         f'test_macro={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f}')

        # store metrics for the current fold
        accs.append(val_acc)
        aucs.append(val_auc)
        macros.append(val_macro)
        valid_losses.append(valid_loss)

    # report valid metrics for all folds
    valid_metrics = {
        'acc': compute_stats(accs),
        'auc': compute_stats(aucs),
        'f1_macro': compute_stats(macros),
        'loss': compute_stats(valid_losses),
    }
    logging.info(f'Valid metrics over folds: {json.dumps(valid_metrics, indent=2)}')


def train_one_split(
    model,
    train_loader,
    valid_loader,
    optimizer,
    loss_f,
    device,
    cfg: Config,
):
    ''' train on a split, return last epoch valid metrics '''

    # set model to train
    model.train()

    test_interval = cfg.log.test_step

    # valid metrics for each epoch
    accs, aucs, macros, losses = [], [], [], []
    epoch_num = cfg.train.epochs

    # run training for a given num of epochs
    for i in range(epoch_num):
        # total loss for epoch (all batches)
        loss_all = 0

        # iterate over batches, run forward-backward-update
        for data in train_loader:
            # move batch to device
            data = data.to(device)
            # zero grad
            optimizer.zero_grad()
            # forward
            out = model(data)
            # compute loss
            loss = loss_f(out, data.y.float().reshape(out.shape))
            # backward and optimizer step
            loss.backward()
            optimizer.step()
            # TODO: add lr_scheduler

            # update total_loss
            loss_all += loss.item()

        # average by total length of the dataset
        epoch_loss = loss_all / len(train_loader.dataset)

        # epoch train metrics (just printed)
        train_acc, train_auc, train_macro, _ = evaluate(model, train_loader, loss_f, cfg)
        logging.info(f'Epoch={i:03d}, loss={epoch_loss:.4f}, '
                     f'train_acc={(train_acc * 100):.2f}, train_macro={(train_macro * 100):.2f}, '
                     f'train_auc={(train_auc * 100):.2f}')

        # epoch valid metrics
        valid_acc, valid_auc, valid_macro, valid_loss = evaluate(model, valid_loader, loss_f, cfg)

        # update epoch metric lists
        accs.append(valid_acc)
        aucs.append(valid_auc)
        macros.append(valid_macro)
        losses.append(valid_loss)

        if (i + 1) % test_interval == 0:
            text = f'Epoch={i:03d}, valid_loss={valid_loss}, valid_acc={(valid_acc * 100):.2f}, ' \
                   f'valid_macro={(valid_macro * 100):.2f}, valid_auc={(valid_auc * 100):.2f}\n'
            logging.info(text)

    #accs, aucs, macros = np.sort(np.array(accs)), np.sort(np.array(aucs)), np.sort(np.array(macros))
    #accs_np, aucs_np, macros_np = np.array(accs), np.array(aucs), np.array(macros)

    # last epoch metrics
    return accs[-1], aucs[-1], macros[-1], losses[-1]


# TODO: add CE and BCE
@torch.no_grad()
def evaluate(model, loader, loss_f, cfg: Config):
    ''' compute metrics on a subset e.g. train, valid or test'''
    model.eval()
    device = cfg.train.device

    thr = cfg.train.prob_thr
    preds, trues, preds_prob = [], [], []

    total_loss = 0.

    # infer, get labels, probs and trues
    for data in loader:
        data = data.to(device)
        c = model(data)
        loss = loss_f(c, data.y.float().reshape(c.shape))
        total_loss += loss.item()

        # append batch preds to a list
        preds += (torch.sigmoid(c) > thr).long().detach().cpu().tolist()
        preds_prob += (torch.sigmoid(c)).detach().cpu().tolist()
        trues += data.y.detach().long().cpu().tolist()

    total_loss = total_loss / len(loader.dataset)

    train_auc = metrics.roc_auc_score(trues, preds_prob)

    if np.isnan(train_auc):
        train_auc = 0.5

    #train_micro = metrics.f1_score(trues, preds, average='micro')
    train_acc = metrics.accuracy_score(trues, preds)
    train_macro = metrics.f1_score(trues, preds, average='macro', labels=[0, 1])

    return train_acc, train_auc, train_macro, total_loss


def create_model(dataset: NeuroDataset, model_cfg: ModelConfig):
    #local_models = {
    #    name: obj for (name, obj) in inspect.getmembers(models)
    #    if inspect.isclass(obj)
    #    if issubclass(obj, nn.Module)
    #}
    if model_cfg.name == 'GAT':
        ModelKlass = GAT
    else:
        raise ValueError('Unknown model')
    return ModelKlass(
        input_dim=dataset.n_features,
        num_nodes=dataset.num_nodes,
        model_cfg=model_cfg,
    )


def build_model_and_loss(cfg: Config):
    pass


def process_bce_preds():
    pass


def process_ce_preds():
    pass


def compute_stats(lst: list[float]):
    arr = np.array(lst)
    return {'mean': arr.mean(), 'std': arr.std(), 'min': arr.min(), 'max': arr.max()}

