from hydra import compose, initialize
from .config import Config


def get_config(name: str = 'config') -> Config:
    with initialize(version_base=None, config_path="."):
        cfg = compose(name)
    return cfg
