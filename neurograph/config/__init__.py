from hydra import compose, initialize
from omegaconf import OmegaConf
from .config import (
    Config,
    MLPConfig,
    MLPlayer,
    DatasetConfig,
    ModelConfig,
    bgbGATConfig,
    bgbGCNConfig,
    standartGNNConfig,
    TransformerConfig,
    TrainConfig,
    LogConfig,
    validate_config
)


def get_config(name: str = 'config') -> Config:
    with initialize(version_base=None, config_path="."):
        cfg: Config = OmegaConf.structured(compose(name))
    return cfg
