import numpy as np
import torch
from torch_geometric.data import Data, Batch

from neurograph.config import get_config, ModelConfig, bgbGATConfig
from neurograph.models.gat import bgbGAT, MPGATConv
from neurograph.data.utils import conn_matrix_to_edges
from .utils import random_batch, random_graph


def create_default_gat(n, f):
    return bgbGAT(input_dim=f, num_nodes=n, model_cfg=bgbGATConfig())


def test_gat_c1(b=3, n=37, f=13):
    batch = random_batch(b=b, n=n, f=f)
    model_cfg = bgbGATConfig()
    m = bgbGAT(input_dim=f, num_nodes=n, model_cfg=model_cfg)
    o = m(batch)
    assert o.shape == (b, model_cfg.n_classes)
    assert o.isnan().sum() == 0

