import numpy as np
import torch
from torch_geometric.data import Data, Batch
from neurograph.config import get_config, ModelConfig
from neurograph.models.gat import bgbGAT, MPGATConv
from neurograph.data.utils import cm_to_edges
from .utils import random_batch, random_graph


def create_default_gat(n, f):
    return bgbGAT(input_dim=f, num_nodes=n, model_cfg=get_config().model)


def test_gat_c1(b=3, n=37, f=13):
    cfg = get_config()
    batch = random_batch(b=b, n=n, f=f)
    m = create_default_gat(n, f)
    o = m(batch)
    assert o.shape == (b, cfg.model.n_classes)
    assert o.isnan().sum() == 0

