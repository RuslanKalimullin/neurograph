""" Config module """

from hydra import compose, initialize
from omegaconf import OmegaConf
from .config import (
    Config,
    MLPConfig,
    MLPlayer,
    DatasetConfig,
    UnimodalDatasetConfig,
    MultimodalDatasetConfig,
    MultiModalTransformerConfig,
    ModelConfig,
    BrainGATConfig,
    BrainGCNConfig,
    StandartGNNConfig,
    TransformerConfig,
    TrainConfig,
    LogConfig,
    validate_config
)


def get_config(name: str = 'config') -> Config:
    """ Get config instance by its name (get default config if not specified """
    with initialize(version_base=None, config_path="."):
        cfg: Config = OmegaConf.structured(compose(name))
    return cfg
