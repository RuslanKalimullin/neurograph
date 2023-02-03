import inspect
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


def train_one_split(
    model,
    train_loader,
    valid_loader,
    optimizer,
    loss_f,
    device,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ''' train on a split, return valid metrics '''

    # set model to train
    model.train()

    test_interval = cfg.log.test_step

    # init lists with metrics for the current split
    # for each epochs
    accs, aucs, macros = [], [], []
    epoch_num = cfg.train.epochs

    # run training for a given num of epochs
    for i in range(epoch_num):
        # total loss for epoch (all batches)
        loss_all = 0

        # get batch
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

        # train metrics
        train_micro, train_auc, train_macro = evaluate(model, train_loader, device, cfg)
        logging.info(f'(Train) | Epoch={i:03d}, loss={epoch_loss:.4f}, '
                     f'train_micro={(train_micro * 100):.2f}, train_macro={(train_macro * 100):.2f}, '
                     f'train_auc={(train_auc * 100):.2f}')

        # valid metrics
        if (i + 1) % test_interval == 0:
            test_micro, test_auc, test_macro = evaluate(model, valid_loader, device, cfg)

            accs.append(test_micro)
            aucs.append(test_auc)
            macros.append(test_macro)

            text = f'(Train Epoch {i}), test_micro={(test_micro * 100):.2f}, ' \
                   f'test_macro={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f}\n'
            logging.info(text)

    #accs, aucs, macros = np.sort(np.array(accs)), np.sort(np.array(aucs)), np.sort(np.array(macros))
    accs_np, aucs_np, macros_np = np.array(accs), np.array(aucs), np.array(macros)

    return accs_np.mean(), aucs_np.mean(), macros_np.mean()


def process_bce_preds():
    pass


def process_ce_preds():
    pass


# TODO: add CE and BCE
@torch.no_grad()
def evaluate(model, loader, device, cfg: Config):
    # compute metrics on valid data
    model.eval()

    thr = cfg.train.prob_thr
    preds, trues, preds_prob = [], [], []

    for data in loader:
        data = data.to(device)
        c = model(data)

        # label preds
        preds += (torch.sigmoid(c) > thr).long().detach().cpu().tolist()
        preds_prob += (torch.sigmoid(c)).detach().cpu().tolist()
        trues += data.y.detach().long().cpu().tolist()

    train_auc = metrics.roc_auc_score(trues, preds_prob)

    if np.isnan(train_auc):
        train_auc = 0.5

    train_micro = metrics.f1_score(trues, preds, average='micro')
    train_macro = metrics.f1_score(trues, preds, average='macro', labels=[0, 1])

    return train_micro, train_auc, train_macro


def train(ds: NeuroDataset, cfg: Config):
    # metrics  # TODO put into dataclass
    accs, aucs, macros = [], [], []

    for fold_i, loaders_dict in enumerate(ds.get_cv_loaders()):
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

        # TODO
        # train and eval
        val_micro, val_auc, val_macro = train_one_split(
            model,
            train_loader,
            valid_loader,
            optimizer,
            loss_f=loss_f,
            device=cfg.train.device,
            cfg=cfg,
        )

        # print valid metrics
        logging.info(f'(Initial Performance Last Epoch) | val_micro={(val_micro * 100):.2f}, '
                     f'val_macro={(val_macro * 100):.2f}, val_auc={(val_auc * 100):.2f}')

        # store metrics for the current fold
        accs.append(val_micro)
        aucs.append(val_auc)
        macros.append(val_macro)

        #models.append(model)
        del model


