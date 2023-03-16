""" Entrypoint for running training """

import os
import os.path as osp
import json
import logging

import hydra
import torch
from omegaconf import OmegaConf
from torch_geometric import seed_everything
import wandb

from neurograph.config import DatasetConfig, Config, validate_config
from neurograph.data import (
    dense_datasets,
    graph_datasets,
    multimodal_dense_2,
    NeuroDenseDataset,
    NeuroGraphDataset,
)
from neurograph.train.train import train


def dataset_factory(ds_cfg: DatasetConfig) -> NeuroDenseDataset | NeuroGraphDataset:
    """ Factory func that returns dataset class instance based on dataset config """

    if ds_cfg.data_type == 'graph':
        return graph_datasets[ds_cfg.name](
            root=str(ds_cfg.data_path),
            atlas=ds_cfg.atlas,
            experiment_type=ds_cfg.experiment_type,
            pt_thr=ds_cfg.pt_thr,
            abs_thr=ds_cfg.abs_thr,
            normalize=ds_cfg.normalize,
        )
    if ds_cfg.data_type == 'dense':
        return dense_datasets[ds_cfg.name](
            root=str(ds_cfg.data_path),
            atlas=ds_cfg.atlas,
            experiment_type=ds_cfg.experiment_type,
            feature_type=ds_cfg.feature_type,
            normalize=ds_cfg.normalize,
        )
    if ds_cfg.data_type == 'multimodal_dense_2':
        return multimodal_dense_2[ds_cfg.name](
            root=str(ds_cfg.data_path),
            atlas=ds_cfg.atlas,
            fmri_feature_type=ds_cfg.fmri_feature_type,
            normalize=ds_cfg.normalize,
        )
    raise ValueError('Unknown dataset data_type! Options: dense, graph, multimodel_dense_2')


@hydra.main(version_base=None, config_path='../config', config_name="config")
def main(cfg: Config):
    """ Entrypoint function for running training from CLI """

    seed_everything(cfg.seed)
    if cfg.train.num_threads:
        torch.set_num_threads(cfg.train.num_threads)

    validate_config(cfg)

    logging.info('Config: \n%s', OmegaConf.to_yaml(cfg))

    cfg_dict = OmegaConf.to_container(cfg)
    wandb.init(
        project=cfg.log.wandb_project,
        settings=wandb.Settings(start_method='thread'),
        config=cfg_dict,  # type: ignore
        mode=cfg.log.wandb_mode,
        name=cfg.log.wandb_name,
        entity=cfg.log.wandb_entity,
    )
    wandb.define_metric('train/*', step_metric='epoch')
    wandb.define_metric('valid/*', step_metric='epoch')

    dataset = dataset_factory(cfg.dataset)
    metrics = train(dataset, cfg)
    wandb.finish()

    logging.info('Results saved in: %s', os.getcwd())

    # save metrics and config
    with open(osp.join(os.getcwd(), 'metrics.json'), 'w') as f_metrics:
        json.dump(metrics, f_metrics)

    OmegaConf.save(cfg, osp.join(os.getcwd(), 'config.yaml'))

# pylint: disable=no-value-for-parameter
main()
