import inspect
import json
import logging
from collections import defaultdict
from typing import Type

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as pygDataLoader
import wandb

from neurograph.config import Config, ModelConfig
from neurograph.data.datasets import NeuroDataset
import neurograph.models
from neurograph.models.available_modules import available_optimizers, available_losses


def get_log_msg(prefix, fold_i, epoch_i, metrics) -> str:
    return (
        'Fold={fold_i:02d}, '
        'Epoch={epoch_i:03d}, ' if epoch_i else ''
        ' | '.join(f'{prefix}/{name}={val}' for name, val in metrics.items())
    )


def train(ds: NeuroDataset, cfg: Config):
    ''' Run cross-validation, report metrics on valids and test '''

    # metrics  # TODO put into dataclass or dict
    logging.info(f'Model architecture:\n{init_model(ds, cfg.model)})')

    # get test loader beforehand
    test_loader = ds.get_test_loader(cfg.train.valid_batch_size)

    # final metrics per fold
    valid_folds_metrics: list[dict[str, float]] = []
    test_folds_metrics: list[dict[str, float]] = []

    # run training for each fold
    loaders_iter = ds.get_cv_loaders(cfg.train.batch_size, cfg.train.valid_batch_size)
    for fold_i, loaders in enumerate(loaders_iter):
        logging.info(f'Run training on fold: {fold_i}')

        model, optimizer, scheduler, loss_f = init_model_optim_loss(ds, cfg)

        # train and return valid metrics on last epoch
        valid_metrics = train_one_split(
            model,
            loaders,
            optimizer,
            loss_f=loss_f,
            device=cfg.train.device,
            fold_i=fold_i,
            cfg=cfg,
        )

        # eval on test
        test_metrics = evaluate(model, test_loader, loss_f, cfg)
        logging.info(get_log_msg('test', fold_i, None, test_metrics))

        # save valid and test metrics for each fold
        valid_folds_metrics.append(valid_metrics)
        test_folds_metrics.append(test_metrics)

    # report valid metrics for all folds
    final_valid_metrics = agg_fold_metrics(valid_folds_metrics)
    final_test_metrics = agg_fold_metrics(test_folds_metrics)

    logging.info(f'Valid metrics over folds: {json.dumps(final_valid_metrics, indent=2)}')
    logging.info(f'Test metrics over folds: {json.dumps(final_test_metrics, indent=2)}')

    return {'valid': final_valid_metrics, 'test': final_test_metrics}


def train_one_split(
    model: nn.Module,
    loaders: dict[str, DataLoader | pygDataLoader],
    optimizer,
    loss_f,
    device,
    fold_i: int | str,
    cfg: Config,
):
    ''' train on train/val split, return last epoch valid metrics '''

    model.train()
    train_loader = loaders['train']
    valid_loader = loaders['valid']

    for epoch_i in range(cfg.train.epochs):
        total_loss = 0.
        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            out = model(data)
            loss = loss_f(out, data.y.float().reshape(out.shape))
            loss.backward()
            optimizer.step()
            # TODO: add lr_scheduler
            total_loss += loss.item()

        # average by total num of samples
        # https://github.com/pytorch/pytorch/issues/47055
        train_loss = total_loss / len(train_loader.dataset)  # type: ignore

        # rewrite loss
        train_epoch_metrics = evaluate(model, train_loader, loss_f, cfg)
        train_epoch_metrics['loss'] = train_loss
        logging.info(get_log_msg('train', fold_i, epoch_i, train_epoch_metrics))

        # epoch valid metrics
        valid_epoch_metrics = evaluate(model, valid_loader, loss_f, cfg)
        logging.info(get_log_msg('valid', fold_i, epoch_i, valid_epoch_metrics))

        # log to wandb
        wandb.log({
            f'train/fold_{fold_i}': train_epoch_metrics,
            'valid/fold_{fold_i}': valid_epoch_metrics,
        })

    # last epoch valid metrics
    return valid_epoch_metrics


# TODO: add support for both CE and BCE
@torch.inference_mode()
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

    loss = total_loss / len(loader.dataset)
    auc = metrics.roc_auc_score(trues, preds_prob)

    if np.isnan(auc):
        auc = 0.5

    acc = metrics.accuracy_score(trues, preds)
    f1_macro = metrics.f1_score(trues, preds, average='macro', labels=[0, 1])

    return {'acc': acc, 'auc': auc, 'f1_macro': f1_macro, 'loss': loss}


def init_model_optim_loss(ds: NeuroDataset, cfg: Config):
    # create model instance
    model = init_model(ds, cfg.model)
    # set optimizer
    optimizer = available_optimizers[cfg.train.optim](
        model.parameters(),
        **cfg.train.optim_args if cfg.train.optim_args else {}
    )
    # set lr_scheduler
    scheduler = None
    # set loss function
    loss_f = available_losses[cfg.train.loss](
        **cfg.train.loss_args if cfg.train.loss_args else {}
    )
    return model, optimizer, scheduler, loss_f


def init_model(dataset: NeuroDataset, model_cfg: ModelConfig):
    available_models = {name: obj for name, obj in inspect.getmembers(neurograph.models)}
    ModelKlass = available_models[model_cfg.name]
    return ModelKlass(
        input_dim=dataset.n_features,
        num_nodes=dataset.num_nodes,
        model_cfg=model_cfg,
    )


def process_bce_preds():
    pass


def process_ce_preds():
    pass


def agg_fold_metrics(lst: list[dict[str, float]]):
    keys = lst[0].keys()
    res = {}
    for k in keys:
        res[k] = compute_stats([dct[k] for dct in lst])
    return res


def compute_stats(lst: list[float]) -> dict[str, np.ndarray]:
    arr = np.array(lst)
    return {'mean': arr.mean(), 'std': arr.std(), 'min': arr.min(), 'max': arr.max()}
