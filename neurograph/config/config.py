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
class MLPlayer:
    out_size: int = 10
    act_func: Optional[str] = 'ReLU'
    act_func_params: Optional[dict] = None
    dropout: Optional[float] = None


@dataclass
class MLPConfig:
    # layers define only hidden dimensions, so
    # if you need one layer NN, leave layers empty

    # in and out sizes are optional and usually depend on upstream model and the task
    in_size: Optional[int] = None
    out_size: Optional[int] = None
    act_func: Optional[str] = None
    act_func_params: Optional[dict] = None
    layers: list[MLPlayer] = field(default_factory=list)


@dataclass
class ModelConfig:
    mlp_config: MLPConfig = field(default_factory=MLPConfig)

    init_node_embed: str = ''  # TODO
    name: str = 'GAT'
    mp_type: str = 'edge_node_concate'
    pooling: str = 'concat'
    num_layers: int = 1
    num_heads: int = 1
    hidden_dim: int = 8  # TODO: support list
    prepool_dim: int = 8  # input dim for prepool layer
    final_node_dim: int = 4  # final node_dim after prepool
    #activation: str = 'ReLU'
    use_abs_weight: bool = True
    dropout: float = 0.1
    use_batchnorm: bool = True
    loss: str = 'BCEWithLogitsLoss'
    loss_args: Optional[dict[str, Any]] = None


@dataclass
class TrainConfig:
    epochs: int = 20
    optim: str = 'Adam'
    lr: float = 1e-3
    weight_decay: Optional[float] = None
    optim_args: Optional[dict[str, Any]] = None


@dataclass
class LogConfig:
    # how often print training metrics
    test_step: int = 5


@dataclass
class Config:
    ''' Config schema w/ default values (see dataclasses above) '''
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    log: LogConfig = field(default_factory=LogConfig)

# register default config as `base_config`
cs = ConfigStore.instance()
cs.store(name='base_config', node=Config)
