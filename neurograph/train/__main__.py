import logging
import hydra
from omegaconf import OmegaConf
from neurograph.config import Config
from neurograph.data import available_datasets
from neurograph.data.datasets import NeuroDataset


def load_dataset(cfg: Config) -> NeuroDataset:
    ds_cfg = cfg.dataset
    DsKlass = available_datasets[ds_cfg.name]
    return DsKlass(
        root=ds_cfg.data_path,
        atlas=ds_cfg.atlas,
        experiment_type=ds_cfg.experiment_type,
    )


@hydra.main(version_base=None, config_path='../config', config_name="config")
def train(cfg: Config):
    # load dataset
    ds = load_dataset(cfg)

    # run training
    # TODO

    print(ds)
    print(OmegaConf.to_yaml(cfg))


train()
