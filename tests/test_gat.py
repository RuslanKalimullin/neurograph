import numpy as np
import torch
from torch_geometric.data import Data, Batch
from neurograph.config import get_config, ModelConfig
from neurograph.models.gat import GAT
from neurograph.data.utils import cm_to_edges


def random_graph(n, f=13):
    w = np.random.randn(n, n)
    edge_index, edge_attr = cm_to_edges(w)

    return Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        x=torch.randn(n, f),
        num_nodes=n,
        y=torch.randint(0, 2, (n,)),
        subj_id=[str(i) for i in range(n)],
    )


def random_batch(b=3, n=17, f=13):
    return Batch.from_data_list([random_graph(n, f) for _ in range(b)])


def create_default_gat(n, f):
    return GAT(input_dim=f, num_nodes=n, model_cfg=get_config().model)


def test_gat_c1(b=3, n=37, f=13):
    cfg = get_config()
    batch = random_batch(b=b, n=n, f=f)
    m = create_default_gat(n, f)
    o = m(batch)
    assert o.shape == (b, cfg.model.n_classes)
    assert o.isnan().sum() == 0
