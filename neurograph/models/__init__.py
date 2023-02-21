from .gat import bgbGAT
from .gcn import bgbGCN
from .mlp import BasicMLP
from .transformers import Transformer

graph_model_classes = {
    'bgbGAT': bgbGAT,
    'bgbGCN': bgbGCN,
}
dense_model_classes = {
    'MLP': BasicMLP,
    'transformer': Transformer,
}
