import numpy as np
import torch
from torch_geometric.data import Data, Batch
from neurograph.models.gat import bgbGAT, MPGATConv
from neurograph.data.utils import conn_matrix_to_edges


def random_graph(n, f=13):
    w = np.random.randn(n, n)
    edge_index, edge_attr = conn_matrix_to_edges(w)

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


