import hydra
from pathlib import Path
from typing import Any, Optional
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field

import neurograph
from neurograph.data import available_datasets


@dataclass
class DatasetConfig:
    name: str = 'cobre'
    experiment_type: str = 'fmri'
    atlas: str = 'aal'
    data_path: Path = Path(neurograph.__file__).resolve().parent.parent / 'datasets'


@dataclass
class ModelConfig:
    name: str = 'GAT'
    n_layer: int = 1
    pool: str = 'concat'
    use_abs_weight: bool = True


@dataclass
class TrainConfig:
    epochs: int = 20
    optim: str = 'Adam'
    lr: float = 1e-3
    weight_decay: Optional[float] = None
    optim_args: Optional[dict[str, Any]] = None


@dataclass
class Config:
    ''' Config schema w/ default values (see dataclasses above) '''
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

# register default config as `base_config`
cs = ConfigStore.instance()
cs.store(name='base_config', node=Config)
