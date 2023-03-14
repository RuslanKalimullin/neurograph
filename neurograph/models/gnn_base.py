""" Module offers thin wrappers over standart GCN, GAT from pytorch geometric """

from typing import Any, Optional

import torch
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn import Sequential as pygSequential
from torch import nn

from neurograph.config import standartGNNConfig
from neurograph.models.mlp import BasicMLP
from neurograph.models.utils import concat_pool
from neurograph.models.available_modules import available_pg_modules


def build_gnn_block(
    input_dim: int,
    hidden_dim: int,
    layer_module: str,
    num_heads: Optional[int] = None,  # ignored in GCN
    proj_dim: Optional[int] = None,
    use_batchnorm: bool = True,
    use_weighted_edges: bool = True,
    dropout: float = 0.0,
):
    in_str = 'x, edge_index, edge_attr' if use_weighted_edges else 'x, edge_index'
    out_str = f'{in_str} -> x'

    if layer_module == 'GCNConv':
        conv = available_pg_modules[layer_module](input_dim, hidden_dim)
        return pygSequential(
            in_str,
            [
                (conv, out_str),
                nn.LeakyReLU(negative_slope = 0.2),
                nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity(),
                nn.Dropout(p=dropout)
            ]
        )
    if layer_module == 'GATConv':
        conv = available_pg_modules[layer_module](
            input_dim,
            hidden_dim,
            heads=num_heads,
        )
        return pygSequential(
            in_str,
            [
                (conv, out_str),
                # project concatenated head embeds back into hidden_dim
                nn.Linear(num_heads * hidden_dim, hidden_dim),
                nn.LeakyReLU(negative_slope = 0.2),
                nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity(),
                nn.Dropout(p=dropout)
            ]
        )
    raise ValueError('Unknown `layer_module` name')


class baseGNN(torch.nn.Module):
    def __init__(
        self,
        # determined by dataset
        input_dim: int,
        num_nodes: int,
        model_cfg: standartGNNConfig,  # ModelConfig,
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.num_nodes = num_nodes
        self.pooling = model_cfg.pooling
        self.use_abs_weight = model_cfg.use_abs_weight
        self.use_weighted_edges = model_cfg.use_weighted_edges

        num_classes = model_cfg.n_classes
        hidden_dim = model_cfg.hidden_dim
        num_layers = model_cfg.num_layers
        layer_module = model_cfg.layer_module
        dropout = model_cfg.dropout
        use_batchnorm = model_cfg.use_batchnorm
        num_heads = model_cfg.num_heads

        gcn_input_dim = input_dim
        common_args: dict[str, Any] = dict(
            dropout=dropout,
            num_heads=num_heads,
        )
        for _ in range(num_layers):
            conv = build_gnn_block(
                gcn_input_dim,
                hidden_dim,
                layer_module,
                use_batchnorm=use_batchnorm,
                use_weighted_edges=self.use_weighted_edges,
                **common_args,
            )
            gcn_input_dim = hidden_dim
            self.convs.append(conv)

        fcn_dim = -1

        if self.pooling == 'concat':
            fcn_dim = hidden_dim * num_nodes
        elif self.pooling in ('sum', 'mean'):
            fcn_dim = hidden_dim

        self.fcn = BasicMLP(in_size=fcn_dim, out_size=num_classes, config=model_cfg.mlp_config)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        z = x
        if self.use_abs_weight:
            edge_attr = torch.abs(edge_attr)

        for _, conv in enumerate(self.convs):
            # batch_size * num_nodes, hidden
            if self.use_weighted_edges:
                z = conv(z, edge_index, edge_attr)
            else:
                z = conv(z, edge_index)

        if self.pooling == "concat":
            z = concat_pool(z, self.num_nodes)
        elif self.pooling == 'sum':
            z = global_add_pool(z,  batch)  # [N, F]
        elif self.pooling == 'mean':
            z = global_mean_pool(z, batch)  # [N, F]

        out = self.fcn(z)
        return out
