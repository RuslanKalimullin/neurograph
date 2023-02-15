import os
import json
import logging
import os.path as osp
from typing import Any, Mapping

import git
import hydra
import wandb
from omegaconf import OmegaConf
from torch_geometric import seed_everything

from neurograph.config import Config
from neurograph.data import available_datasets
from neurograph.data.datasets import NeuroGraphDataset
from neurograph.train.train import train


def load_dataset(cfg: Config) -> NeuroGraphDataset:
    ds_cfg = cfg.dataset
    DsKlass = available_datasets[ds_cfg.name]
    return DsKlass(
        root=ds_cfg.data_path,
        atlas=ds_cfg.atlas,
        experiment_type=ds_cfg.experiment_type,
        pt_thr=ds_cfg.pt_thr,
        abs_thr=ds_cfg.abs_thr,
    )


@hydra.main(version_base=None, config_path='../config', config_name="config")
def main(cfg: Config):
    seed_everything(cfg.seed)

    cfg_dict = OmegaConf.to_container(cfg)
    wandb.init(
        project=cfg.log.wandb_project,
        settings=wandb.Settings(start_method="thread"),
        config=cfg_dict,  # type: ignore
        mode=cfg.log.wandb_mode,
        name=cfg.log.wandb_name,
    )
    wandb.define_metric('train/*', step_metric='epoch')
    wandb.define_metric('valid/*', step_metric='epoch')

    ds = load_dataset(cfg)

    logging.info(f'Config: \n{OmegaConf.to_yaml(cfg)}')
    metrics = train(ds, cfg)
    wandb.finish()

    logging.info(f'Results saved in: {os.getcwd()}')

    # save metrics and config
    with open(osp.join(os.getcwd(), 'metrics.json'), 'w') as f_metrics:
        json.dump(metrics, f_metrics)

    OmegaConf.save(cfg, osp.join(os.getcwd(), 'config.yaml'))

main()
