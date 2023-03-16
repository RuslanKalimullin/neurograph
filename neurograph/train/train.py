""" Module w/ functions for running training """

import inspect
import json
import logging
from copy import deepcopy
from typing import Any
from operator import gt, lt

import numpy as np
from sklearn import metrics
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch_geometric.data import Batch as pygBatch, Data as pygData
from torch_geometric.loader import DataLoader as pygDataLoader
import wandb

from neurograph.config import Config, ModelConfig
from neurograph.data import NeuroDataset
import neurograph.models
from neurograph.models.available_modules import (
    available_optimizers,
    available_losses,
    available_schedulers,
)


def get_log_msg(prefix, fold_i, epoch_i, metrics_dict) -> str:
    """ Form logging message w/ metrics, given epoch and subset (train, valid, test) """
    return ''.join([
        f'Fold={fold_i:02d}, ',
        f'Epoch={epoch_i:03d}, ' if epoch_i is not None else '',
        ' | '.join(f'{prefix}/{name}={val:.3f}' for name, val in metrics_dict.items()),
    ])


def train(dataset: NeuroDataset, cfg: Config):
    ''' Run cross-validation, report metrics on valids and test '''

    logging.info('Model architecture:\n %s', init_model(dataset, cfg))

    # get test loader beforehand
    test_loader = dataset.get_test_loader(cfg.train.valid_batch_size)

    # final metrics per fold
    valid_folds_metrics: list[dict[str, float]] = []
    test_folds_metrics: list[dict[str, float]] = []

    # run training for each fold
    loaders_iter = dataset.get_cv_loaders(cfg.train.batch_size, cfg.train.valid_batch_size)
    for fold_i, loaders in enumerate(loaders_iter):
        logging.info('Run training on fold %s', fold_i)

        # create model, optim, loss etc.
        model, optimizer, scheduler, loss_f = init_model_optim_loss(dataset, cfg)

        # train and return valid metrics on last epoch
        valid_metrics, best_model = train_one_split(
            model,
            loaders,
            optimizer,
            scheduler,
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

    logging.info('Valid metrics over folds: %s', json.dumps(final_valid_metrics, indent=2))
    logging.info('Test metrics over folds: %s', json.dumps(final_test_metrics, indent=2))

    return {'valid': final_valid_metrics, 'test': final_test_metrics}


def handle_batch(
    batch: pygData | pygBatch | list[torch.Tensor] | tuple[torch.Tensor],
    device: str,
):
    """ handles diffrent batch types: DataBatch from pytorch geometric and
        just ordinary torch batch that is tensor.
        Also, it handles unimodal / multimodal cases
    """

    # it's ugly but it works
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            x, y = batch
            x, y = x.to(device), y.to(device)
            batch = (x, y)
        elif len(batch) == 3:
            x_1, x_2, y = batch
            x_1, x_2, y = x_1.to(device), x_2.to(device), y.to(device)
            batch = (x_1, x_2, y)
        else:
            raise ValueError('Unknown batch composition! expect len=2,3')
    else:
        batch = batch.to(device)
        y = batch.y

    return batch, y


def train_one_split(
    model: nn.Module,
    loaders: dict[str, DataLoader | pygDataLoader],
    optimizer,
    scheduler,
    loss_f,
    device,
    fold_i: int | str,
    cfg: Config,
):
    ''' train on train/val split, return last epoch valid metrics
        Params:
        loaders: dict, must have keys 'train' and 'valid'
    '''
    model.to(device)
    model.train()
    train_loader = loaders['train']
    valid_loader = loaders['valid']

    best_metric = cfg.train.select_best_metric
    best_result = float('inf') if best_metric == 'loss' else -float('inf')
    comparator = lt if best_metric == 'loss' else gt
    best_model: nn.Module
    best_valid_metrics: dict[str, float]

    for epoch_i in range(cfg.train.epochs):
        total_loss = 0.
        for data in train_loader:
            optimizer.zero_grad()
            data, y = handle_batch(data, device)
            out = model(data)

            if isinstance(loss_f, BCEWithLogitsLoss):
                loss = loss_f(out, y.float().reshape(out.shape))
            elif isinstance(loss_f, CrossEntropyLoss):
                loss = loss_f(out, y)
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

        # taken from: https://github.com/twitter-research/cwn/blob/main/exp/run_exp.py#L347
        # decay learning rate
        if scheduler is not None:
            if cfg.train.scheduler == 'ReduceLROnPlateau':
                scheduler.step(valid_epoch_metrics[cfg.train.scheduler_metric])
                # We use a strict inequality here like in the benchmarking GNNs paper code
                # https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/main_molecules_graph_regression.py#L217
                #if args.early_stop and optimizer.param_groups[0]['lr'] < args.lr_scheduler_min:
                #    print("\n!! The minimum learning rate has been reached.")
                #    break
            else:
                scheduler.step()

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


@torch.inference_mode()
def evaluate(model, loader, loss_f, cfg: Config):
    ''' compute metrics on a subset e.g. train, valid or test'''
    model.eval()
    device = cfg.train.device
    thr = cfg.train.prob_thr

    # infer
    y_pred_list, true_list = [], []
    for data in loader:
        data, y = handle_batch(data, device)
        model.to(device)
        out = model(data)
        y_pred_list.append(out)
        true_list.append(y)
    y_pred = torch.cat(y_pred_list, dim=0)
    trues = torch.cat(true_list, dim=0)

    # compute metrics based on loss_f
    metrics_dict = {}
    if isinstance(loss_f, BCEWithLogitsLoss):
        total_loss = loss_f(y_pred, trues.float().reshape(y_pred.shape)).item()
        metrics_dict = process_bce_preds(trues, y_pred, thr)
    elif isinstance(loss_f, CrossEntropyLoss):
        total_loss = loss_f(y_pred, trues).item()
        metrics_dict = process_ce_preds(trues, y_pred)
    else:
        ValueError(f'{loss_f} this loss function is not supported')

    # compute loss per sample, add to final result
    loss = total_loss / len(loader.dataset)
    metrics_dict['loss'] = loss

    return metrics_dict


def init_model_optim_loss(dataset: NeuroDataset, cfg: Config):
    """ Init model, optim, scheduler instances given dataset instance and global config """

    # create model instance
    model = init_model(dataset, cfg)
    # set optimizer
    optimizer = available_optimizers[cfg.train.optim](
        model.parameters(),
        **cfg.train.optim_args if cfg.train.optim_args else {}
    )

    # TODO: refactor it somehow or maybe refactor TrainConfig
    # set lr_scheduler
    scheduler = None
    if cfg.train.scheduler is not None:
        scheduler_params: dict[str, Any]
        if cfg.train.scheduler_args is not None:
            scheduler_params = dict(deepcopy(cfg.train.scheduler_args))
        else:
            scheduler_params = {}
        if cfg.train.scheduler == 'ReduceLROnPlateau':
            if cfg.train.scheduler_metric is not None:
                scheduler_params['mode'] = metrics_resistry[cfg.train.scheduler_metric]

        scheduler = available_schedulers[cfg.train.scheduler](
            optimizer,
            **scheduler_params
        )
    # set loss function
    loss_f = available_losses[cfg.train.loss](
        **cfg.train.loss_args if cfg.train.loss_args else {}
    )
    return model, optimizer, scheduler, loss_f


def init_model(dataset: NeuroDataset, cfg: Config):
    """ Init a model instance given dataset instance and global config """

    model_cfg: ModelConfig = cfg.model
    available_models = dict(inspect.getmembers(neurograph.models))

    model_class = available_models[model_cfg.name]

    if cfg.dataset.data_type.startswith('multimodal_'):
        return model_class(
            input_dim_1=dataset.num_fmri_features,
            input_dim_2=dataset.num_dti_features,
            num_nodes_1=dataset.num_fmri_nodes,
            num_nodes_2=dataset.num_dti_nodes,
            model_cfg=model_cfg,
        )

    return model_class(
        input_dim=dataset.num_features,
        num_nodes=dataset.num_nodes,
        model_cfg=model_cfg,
    )


def process_bce_preds(trues_pt: torch.Tensor, y_pred: torch.Tensor, thr: float):
    """ Given 1-class predictions (expected by BCE), compute probas, labels (using thr) """
    pred_labels = (torch.sigmoid(y_pred) > thr).long().detach().cpu().numpy()
    pred_proba = (torch.sigmoid(y_pred)).detach().cpu().numpy()
    trues = trues_pt.detach().long().cpu().numpy()

    return compute_metrics(pred_labels, pred_proba, trues)


def process_ce_preds(trues_pt: torch.Tensor, y_pred: torch.Tensor):
    """ Given k-class predictions (expected by CE), compute probas, labels (by argmax) """
    pred_proba = (torch.softmax(y_pred, dim=1)).detach().cpu().numpy()
    pred_labels = pred_proba.argmax(axis=1)
    trues = trues_pt.detach().long().cpu().numpy()

    return compute_metrics(pred_labels, pred_proba[:, 1], trues)

# used in scheduler ReduceLROnPlateau
metrics_resistry = {'acc': 'max', 'auc': 'max', 'f1_macro': 'max', 'loss': 'min'}


def compute_metrics(pred_labels: np.ndarray, pred_proba: np.ndarray, trues: np.ndarray):
    """ Compute metrics for binary classification """

    auc = metrics.roc_auc_score(trues, pred_proba)
    if np.isnan(auc):
        auc = 0.5
    acc = metrics.accuracy_score(trues, pred_labels)
    f1_macro = metrics.f1_score(trues, pred_labels, average='macro', labels=[0, 1])

    return {'acc': acc, 'auc': auc, 'f1_macro': f1_macro}


def agg_fold_metrics(lst: list[dict[str, float]]):
    """ Compute mean, min, max, std from cross validation metrics """
    keys = lst[0].keys()
    res = {}
    for k in keys:
        res[k] = compute_stats([dct[k] for dct in lst])
    return res


def compute_stats(lst: list[float]) -> dict[str, np.ndarray]:
    """ Compute some stats from a list of floats """
    arr = np.array(lst)
    return {'mean': arr.mean(), 'std': arr.std(), 'min': arr.min(), 'max': arr.max()}
