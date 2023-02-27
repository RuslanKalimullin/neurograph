import numpy as np
import torch
from torch_geometric.data import Data, Batch

from neurograph.config import get_config, ModelConfig, bgbGCNConfig
from neurograph.models.gcn import bgbGCN, MPGCNConv
from neurograph.data.utils import cm_to_edges
from .utils import random_batch, random_graph


def test_gcn_1(b=3, n=37, f=13):
    batch = random_batch(b=b, n=n, f=f)
    model_cfg = bgbGCNConfig()
    m = bgbGCN(input_dim=f, num_nodes=n, model_cfg=model_cfg)
    o = m(batch)
    assert o.shape == (b, model_cfg.n_classes)
    assert o.isnan().sum() == 0
