import inspect
import json
import logging
from collections import defaultdict
from copy import deepcopy
from typing import Type
from operator import gt, lt

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as pygDataLoader
import wandb

from neurograph.config import Config, ModelConfig
from neurograph.data import NeuroDataset, NeuroGraphDataset
import neurograph.models
from neurograph.models.available_modules import available_optimizers, available_losses


def get_log_msg(prefix, fold_i, epoch_i, metrics) -> str:
    return ''.join([
        f'Fold={fold_i:02d}, ',
        f'Epoch={epoch_i:03d}, ' if epoch_i is not None else '',
        ' | '.join(f'{prefix}/{name}={val:.3f}' for name, val in metrics.items()),
    ])


def train(ds: NeuroGraphDataset, cfg: Config):
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
        valid_metrics, best_model = train_one_split(
            model,
            loaders,
            optimizer,
            loss_f=loss_f,
            device=cfg.train.device,
            fold_i=fold_i,
            cfg=cfg,
        )
        # eval on test
        test_metrics = evaluate(best_model, test_loader, loss_f, cfg)
        logging.info(get_log_msg('test', fold_i, None, test_metrics))

        # save valid and test metrics for each fold
        valid_folds_metrics.append(valid_metrics)
        test_folds_metrics.append(test_metrics)

        # just to be sure
        del model, best_model

    # aggregate valid and test metrics for all folds
    final_valid_metrics = agg_fold_metrics(valid_folds_metrics)
    final_test_metrics = agg_fold_metrics(test_folds_metrics)

    wandb.summary['final'] = {'valid': final_valid_metrics, 'test': final_test_metrics}

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
    ''' train on train/val split, return last epoch valid metrics
        Params:
        loaders: dict, must have keys 'train' and 'valid'
    '''

    model.train()
    train_loader = loaders['train']
    valid_loader = loaders['valid']

    best_metric = cfg.train.select_best_metric
    best_result = float('inf') if best_metric == 'loss' else -float('inf')
    comparator = lt if best_metric == 'loss' else gt
    best_nodel: nn.Module
    best_valid_metrics: dict[str, float]

    for epoch_i in range(cfg.train.epochs):
        total_loss = 0.
        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            out = model(data)

            if isinstance(loss_f, BCEWithLogitsLoss):
                loss = loss_f(out, data.y.float().reshape(out.shape))
            elif isinstance(loss_f, CrossEntropyLoss):
                loss = loss_f(out, data.y)
            else:
                ValueError(f'{loss_f} this loss function is not supported')

            loss.backward()
            optimizer.step()
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

        # update best_model
        if comparator(valid_epoch_metrics[best_metric], best_result):
            best_model = deepcopy(model)
            best_result = valid_epoch_metrics[best_metric]
            best_valid_metrics = deepcopy(valid_epoch_metrics)

        # log to wandb, add prefix like 'train/fold_3/'to each metric
        wandb.log({
            **{f'train/fold_{fold_i}/{name}': val for name, val in train_epoch_metrics.items()},
            **{f'valid/fold_{fold_i}/{name}': val for name, val in valid_epoch_metrics.items()},
            'epoch': epoch_i,
        })

    # last epoch valid metrics
    return best_valid_metrics, best_model


# TODO: add support for both CE and BCE
@torch.inference_mode()
def evaluate(model, loader, loss_f, cfg: Config):
    ''' compute metrics on a subset e.g. train, valid or test'''
    model.eval()
    device = cfg.train.device
    thr = cfg.train.prob_thr

    # infer
    y_pred_list, true_list = [], []
    for data in loader:
        data = data.to(device)
        out = model(data)
        y_pred_list.append(out)
        true_list.append(data.y)
    y_pred = torch.cat(y_pred_list, dim=0)
    trues = torch.cat(true_list, dim=0)

    # compute metrics based on loss_f
    metrics = {}
    if isinstance(loss_f, BCEWithLogitsLoss):
        total_loss = loss_f(y_pred, trues.float().reshape(y_pred.shape)).item()
        metrics = process_bce_preds(trues, y_pred, thr)
    elif isinstance(loss_f, CrossEntropyLoss):
        total_loss = loss_f(y_pred, trues).item()
        metrics = process_ce_preds(trues, y_pred)
    else:
        ValueError(f'{loss_f} this loss function is not supported')

    # compute loss per sample, add to final result
    loss = total_loss / len(loader.dataset)
    metrics['loss'] = loss

    return metrics


def init_model_optim_loss(ds: NeuroGraphDataset, cfg: Config):
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


def init_model(dataset: NeuroGraphDataset, model_cfg: ModelConfig):
    available_models = {name: obj for name, obj in inspect.getmembers(neurograph.models)}
    ModelKlass = available_models[model_cfg.name]
    return ModelKlass(
        input_dim=dataset.num_features,
        num_nodes=dataset.num_nodes,
        model_cfg=model_cfg,
    )


def process_bce_preds(trues_pt: torch.Tensor, y_pred: torch.Tensor, thr: float):
    # compute probas, labels (using thr)
    pred_labels = (torch.sigmoid(y_pred) > thr).long().detach().cpu().numpy()
    pred_proba = (torch.sigmoid(y_pred)).detach().cpu().numpy()
    trues = trues_pt.detach().long().cpu().numpy()

    return compute_metrics(pred_labels, pred_proba, trues)


def process_ce_preds(trues_pt: torch.Tensor, y_pred: torch.Tensor):
    pred_proba = (torch.softmax(y_pred, dim=1)).detach().cpu().numpy()
    pred_labels = pred_proba.argmax(axis=1)
    trues = trues_pt.detach().long().cpu().numpy()

    return compute_metrics(pred_labels, pred_proba[:, 1], trues)


def compute_metrics(pred_labels: np.ndarray, pred_proba: np.ndarray, trues: np.ndarray):
    """ Compute metrics for binary classification """
    auc = metrics.roc_auc_score(trues, pred_proba)
    if np.isnan(auc):
        auc = 0.5
    acc = metrics.accuracy_score(trues, pred_labels)
    f1_macro = metrics.f1_score(trues, pred_labels, average='macro', labels=[0, 1])

    return {'acc': acc, 'auc': auc, 'f1_macro': f1_macro}


def agg_fold_metrics(lst: list[dict[str, float]]):
    keys = lst[0].keys()
    res = {}
    for k in keys:
        res[k] = compute_stats([dct[k] for dct in lst])
    return res


def compute_stats(lst: list[float]) -> dict[str, np.ndarray]:
    arr = np.array(lst)
    return {'mean': arr.mean(), 'std': arr.std(), 'min': arr.min(), 'max': arr.max()}
