import hydra
from pathlib import Path
from typing import Any, Optional, Sequence
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
import neurograph

DEFAULT_DATA_PATH = Path(neurograph.__file__).resolve().parent.parent / 'datasets'


@dataclass
class DatasetConfig:
    data_type: str
    name: str = 'cobre'
    atlas: str = 'aal'
    # for DTI
    normalize: Optional[str] = None

    data_path: Path = DEFAULT_DATA_PATH


@dataclass
class UnimodalDatasetConfig(DatasetConfig):
    data_type: str = 'graph'  # or 'dense'
    experiment_type: str = 'fmri'
    # graph specific
    #init_node_features: str = 'conn_profile'  # TODO
    abs_thr: Optional[float] = None
    pt_thr: Optional[float] = None
    # dense specific
    feature_type: str = 'conn_profile'  #'timeseries'


@dataclass
class MultimodalDatasetConfig(DatasetConfig):
    data_type: str = 'multimodal_dense_2'  # only option rn
    fmri_feature_type: str = 'conn_profile'
    normalize: Optional[str] = 'global_max'


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
        MLPlayer(out_size=32, dropout=0.6, act_func='LeakyReLU', act_func_params=dict(negative_slope=0.2)),
        MLPlayer(out_size=32, dropout=0.6, act_func='LeakyReLU', act_func_params=dict(negative_slope=0.2)),
    ])


@dataclass
class ModelConfig:
    name: str  # see neurograph.models/
    n_classes: int  # must match with loss

    # required for correct init of models
    # see `train.train.init_model`

    data_type: str


@dataclass
class DummyMultimodalDense2Config:
    name: str = 'DummyMultimodalDense2Model'
    n_classes: int = 2
    hidden: int = 8
    dropout: float = 0.2
    act_func: Optional[str] = 'ReLU'
    act_func_params: Optional[dict] = None


@dataclass
class standartGNNConfig(ModelConfig):
    name: str = 'baseGNN'  # see neurograph.models/
    n_classes: int = 2  # must match with loss
    num_layers: int = 2
    layer_module: str = 'GCNConv'
    data_type: str = 'graph'
    hidden_dim: int = 32  # TODO: support list
    use_abs_weight: bool = True
    use_weighted_edges: bool =False
    final_node_dim: int =32
    pooling: str ='concat'
    # TODO: use it inside convolutions
    dropout: float = 0.3
    use_batchnorm: bool = True
    # gat spefic args
    num_heads: int = 2
    # TODO: add adding self-loops
    # gcn spefic args

    mlp_config: MLPConfig = field(default_factory=MLPConfig)


@dataclass
class bgbGCNConfig(ModelConfig):
    name: str = 'bgbGCN'  # see neurograph.models
    n_classes: int = 2  # must match with loss
    data_type: str = 'graph'

    mp_type: str = 'node_concate'
    pooling: str = 'concat'
    num_layers: int = 1
    hidden_dim: int = 16  # TODO: support list
    prepool_dim: int = 64  # input dim for prepool layer
    final_node_dim: int = 8  # final node_dim after prepool
    use_abs_weight: bool = True
    # TODO: use it inside convolutions
    dropout: float = 0.3
    use_batchnorm: bool = True

    # gcn spefic args
    edge_emb_dim: int = 4
    bucket_sz: float = 0.05

    mlp_config: MLPConfig = field(default_factory=MLPConfig)


@dataclass
class bgbGATConfig(ModelConfig):
    name: str = 'bgbGAT'  # see neurograph.models
    n_classes: int = 2  # must match with loss
    data_type: str = 'graph'

    mp_type: str = 'node_concate'
    pooling: str = 'concat'
    num_layers: int = 1
    hidden_dim: int = 16  # TODO: support list
    prepool_dim: int = 64  # input dim for prepool layer
    final_node_dim: int = 8  # final node_dim after prepool
    use_abs_weight: bool = True
    # TODO: use it inside convolutions
    dropout: float = 0.0
    use_batchnorm: bool = True
    # gat spefic args
    num_heads: int = 2
    # TODO: add adding self-loops

    mlp_config: MLPConfig = field(default_factory=MLPConfig)


@dataclass
class TransformerConfig(ModelConfig):
    # name is a class name; used for initializing a model
    name: str  = 'Transformer'  # TODO: remove, refactor ModelConfig class?

    n_classes: int = 2
    num_layers: int = 1
    hidden_dim: int = 116
    num_heads: int = 4
    attn_dropout: float = 0.5
    mlp_dropout: float = 0.5
    # hidden layer in transformer block mlp
    mlp_hidden_multiplier: float = 0.2

    data_type: str = 'dense'

    return_attn: bool = False
    # transformer block MLP parameters
    mlp_act_func: Optional[str] = 'GELU'
    mlp_act_func_params: Optional[dict] = None

    pooling: str = 'concat'

    # final MLP layer config
    head_config:  MLPConfig = field(default_factory=lambda: MLPConfig(
        layers = [
            MLPlayer(out_size=4, dropout=0.5, act_func='GELU',),
        ]
    ))


@dataclass
class MultiModalTransformerConfig(ModelConfig):
    # name is a class name; used for initializing a model
    name: str = 'MultiModalTransformer'  # TODO: remove, refactor ModelConfig class?
    attn_type: str = 'concat'
    projection_dim: int = 64
    n_classes: int = 2
    num_layers: int = 1
    hidden_dim: int = 32
    num_heads: int = 2
    make_projection:  bool =False
    attn_dropout: float = 0.5
    mlp_dropout: float = 0.5
    # hidden layer in transformer block mlp
    mlp_hidden_multiplier: float = 0.2

    data_type: str = 'dense'

    return_attn: bool = False
    # transformer block MLP parameters
    mlp_act_func: Optional[str] = 'GELU'
    mlp_act_func_params: Optional[dict] = None

    pooling: str = 'concat'

    # final MLP layer config
    head_config:  MLPConfig = field(default_factory=lambda: MLPConfig(
        layers = [
            MLPlayer(out_size=4, dropout=0.5, act_func='GELU',),
        ]
    ))


@dataclass
class TrainConfig:
    device: str = 'cuda:0'
    num_threads: Optional[int] = None
    epochs: int = 1
    batch_size: int = 8
    valid_batch_size: int = 8
    optim: str = 'Adam'
    optim_args: Optional[dict[str, Any]] = field(
        default_factory=lambda: {
            'lr': 1e-4,
            'weight_decay': 1e-3,
        }
    )
    scheduler: Optional[str] = 'ReduceLROnPlateau'
    # used in ReduceLROnPlateau
    scheduler_metric: Optional[str] = 'loss'
    scheduler_args: Optional[dict[str, Any]] = field(
        default_factory=lambda: {
            'factor': 0.1,
            'patience': 5,
            'verbose': True,
        }
    )
    # select best model on valid based on what metric
    select_best_metric: str = 'f1_macro'
    loss: str = 'CrossEntropyLoss'  #'BCEWithLogitsLoss'
    loss_args: Optional[dict[str, Any]] = field(
        # reduction sum is necessary here
        default_factory=lambda: {'reduction': 'sum'}
    )
    # if BCE is used
    prob_thr: float = 0.5


@dataclass
class LogConfig:
    # how often print training metrics
    test_step: int = 1
    wandb_project: str = 'mri_multimodal'
    wandb_entity: Optional[str] = 'gnn-neuro'
    wandb_name: Optional[str] = None
    wandb_mode: Optional[str] = None  # 'disabled' for testing


@dataclass
class Config:
    ''' Config schema w/ default values (see dataclasses above) '''
    seed: int = 1380
    model: Any = MISSING
#    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataset: Any = MISSING
    train: TrainConfig = field(default_factory=TrainConfig)
    log: LogConfig = field(default_factory=LogConfig)


def validate_config(cfg: Config):
    if cfg.train.loss == 'CrossEntropyLoss' and cfg.model.n_classes < 2:
        raise ValueError(f'For loss = {cfg.train.loss} `Config.model.n_classes` must be > 1')
    if cfg.train.loss == 'BCEWithLogitsLoss' and cfg.model.n_classes != 1:
        raise ValueError(f'For loss = {cfg.train.loss} `Config.model.n_classes` must be = 1')


# register default config as `base_config`
cs = ConfigStore.instance()
cs.store(name='base_config', node=Config)

# base dataset configs
cs.store(group='dataset', name='base_dataset', node=UnimodalDatasetConfig)
cs.store(group='dataset', name='base_multimodal_dataset', node=MultimodalDatasetConfig)

# base model configs
cs.store(group='model', name='bgbGAT', node=bgbGATConfig)
cs.store(group='model', name='bgbGCN', node=bgbGCNConfig)
cs.store(group='model', name='transformer', node=TransformerConfig)
cs.store(group='model', name='baseGNN', node=standartGNNConfig)
cs.store(group='model', name='dummy_mm2', node=DummyMultimodalDense2Config)
cs.store(group='model', name='mm_transformer', node=MultiModalTransformerConfig)
