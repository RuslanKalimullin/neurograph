""" Package w/ all model classes """

from .gat import BrainGAT
from .gcn import BrainGCN
from .mlp import BasicMLP
from .transformers import Transformer
from .multimodal_transformers import MultiModalTransformer
from .gnn_base import StandartGNN
from .dummy import DummyMultimodalDense2Model
#graph_model_classes = {
#    'bgbGAT': bgbGAT,
#    'bgbGCN': bgbGCN,
#}
#dense_model_classes = {
#    'MLP': BasicMLP,
#    'transformer': Transformer,
#}
