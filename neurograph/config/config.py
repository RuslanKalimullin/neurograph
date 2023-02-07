import hydra
from pathlib import Path
from typing import Any, Optional, Sequence
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field

import neurograph
from neurograph.data import available_datasets


@dataclass
class DatasetConfig:
    name: str = 'cobre'
    experiment_type: str = 'fmri' # TODO: support list for multimodal experiments
    atlas: str = 'aal'
    #init_node_features: str = 'conn_profile'  # TODO
    data_path: Path = Path(neurograph.__file__).resolve().parent.parent / 'datasets'


@dataclass
class MLPlayer:
    out_size: int = 10
    act_func: Optional[str] = 'ReLU'
    act_func_params: Optional[dict] = None
    dropout: Optional[float] = None


@dataclass
class MLPConfig:
    # layers define only hidden dimensions, so final MLP will have n+1 layer.
    # So, if you want to create a 1-layer network, just leave layers empty

    # in and out sizes are optional and usually depend on upstream model and the task
    # for now, they are ignored
    in_size: Optional[int] = None
    out_size: Optional[int] = None

    # act func for the last layer. None -> no activation function
    act_func: Optional[str] = None
    act_func_params: Optional[dict] = None
    layers: list[MLPlayer] = field(default_factory=lambda : [
        MLPlayer(out_size=256, dropout=0.5, act_func='LeakyReLU', act_func_params=dict(negative_slope=0.2)),
        MLPlayer(out_size=32, dropout=0.5, act_func='LeakyReLU', act_func_params=dict(negative_slope=0.2)),
    ])


@dataclass
class ModelConfig:
    name: str = 'bgbGCN'
    n_classes: int = 1  # must match with loss
    mp_type: str = 'node_concate'
    pooling: str = 'concat'
    num_layers: int = 1
    hidden_dim: int = 8  # TODO: support list
    prepool_dim: int = 64  # input dim for prepool layer
    final_node_dim: int = 8  # final node_dim after prepool
    use_abs_weight: bool = True
    dropout: float = 0.3
    use_batchnorm: bool = True
    # gat spefic args
    num_heads: int = 1
    # gcn spefic args
    edge_emb_dim: int = 4
    bucket_sz: float = 0.05

    mlp_config: MLPConfig = field(default_factory=MLPConfig)


@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size: int = 8
    valid_batch_size: int = 8
    optim: str = 'Adam'
    optim_args: Optional[dict[str, Any]] = field(
        default_factory=lambda: {
            'lr': 1e-3,
            'weight_decay': 1e-4,
        }
    )
    device: str = 'cpu'

    loss: str = 'BCEWithLogitsLoss'
    loss_args: Optional[dict[str, Any]] = field(
        default_factory=lambda: {'reduction': 'sum'}
    )

    # if BCE is used
    prob_thr: float = 0.5


@dataclass
class LogConfig:
    # how often print training metrics
    test_step: int = 1
    wandb_project: str = 'mri_gnn'
    wandb_name: Optional[str] = None
    wandb_mode: Optional[str] = None  # 'disabled' for testing


@dataclass
class Config:
    ''' Config schema w/ default values (see dataclasses above) '''
    seed: int = 1380
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    log: LogConfig = field(default_factory=LogConfig)

# register default config as `base_config`
cs = ConfigStore.instance()
cs.store(name='base_config', node=Config)
