from .gat import bgbGAT
from .gcn import bgbGCN
from .mlp import BasicMLP

graph_model_classes = {
    'bgbGAT': bgbGAT,
    'bgbGCN': bgbGCN,
}
dense_model_classes = {
    'MLP': BasicMLP,
}
