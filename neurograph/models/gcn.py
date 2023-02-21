import math
from typing import Any, Union, Tuple, Optional

import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, MessagePassing
from torch.nn import Parameter
import numpy as np
from torch.nn import functional as F
from torch_geometric.nn.inits import glorot, zeros
from typing import Tuple
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Sequential as pygSequential
from torch import nn
import torch_geometric

from neurograph.config import Config, ModelConfig, bgbGNNConfig
from neurograph.models.mlp import BasicMLP
from neurograph.models.utils import concat_pool


class MPGCNConv(GCNConv):

    available_mp_types = {
        "weighted_sum",  # aka 'edge_weighted'
        "bin_concate",
        "edge_weight_concate",
        "edge_node_concate",  # aka 'node_edge_concat'
        "node_concate",  # aka 'node_concat'
    }

    def __init__(
        self,
        in_channels,
        out_channels,
        # used only in `bin_concate' mp
        edge_emb_dim: int,
        mp_type: str = 'node_concate',
        # TODO: use dropout
        dropout: float = 0.0,
        bucket_sz: float = 0.05,
        use_abs_weight: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            aggr='add',
            normalize=False,
        )

        self.mp_type = mp_type
        self.use_abs_weight = use_abs_weight
        self.dropout = dropout

        # bin_concate specific params
        self.bucket_sz = bucket_sz
        self.edge_emb_dim = edge_emb_dim
        # weights are in range [-1, 1]
        self.bucket_num = math.ceil(2.0 / self.bucket_sz)
        if mp_type == "bin_concate":
            self.edge2vec = nn.Embedding(self.bucket_num, edge_emb_dim)

        self._cached_edge_index = None
        self._cached_adj_t = None

        input_dim = out_channels
        if mp_type == "bin_concate" or mp_type == "edge_weight_concate":
            input_dim = out_channels + edge_emb_dim
        elif mp_type == "edge_node_concate":
            input_dim = out_channels * 2 + 1
        elif mp_type == "node_concate":
            input_dim = out_channels * 2

        # MLP for computing messages
        self.edge_lin = torch.nn.Linear(input_dim, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def message(self, x_i, x_j, edge_weight):
        assert len(edge_weight.shape) == 1
        if self.use_abs_weight:
            edge_weight = torch.abs(edge_weight)

        # x_j: [E, in_channels]
        if self.mp_type == "weighted_sum":
            # use edge_weight as multiplier
            msg = edge_weight.view(-1, 1) * x_j
        elif self.mp_type == "bin_concate":
            # concat x_j and learned bin embedding
            bucket = torch.div(
                edge_weight + 1,
                self.bucket_sz,
                rounding_mode='trunc',
            ).int()
            msg = torch.cat([x_j, self.edge2vec(bucket)], dim=1)
            msg = self.edge_lin(msg)
        elif self.mp_type == "edge_weight_concate":
            # concat x_j and tiled edge weight
            msg = torch.cat(
                [x_j, edge_weight.view(-1, 1).repeat(1, self.edge_emb_dim)],
                dim=1,
            )
            msg = self.edge_lin(msg)
        elif self.mp_type == "edge_node_concate":
            # concat x_i, x_j and edge_weight
            msg = torch.cat([x_i, x_j, edge_weight.view(-1, 1)], dim=1)
            msg = self.edge_lin(msg)
        elif self.mp_type == "node_concate":
            # concat x_i and x_j
            msg = torch.cat([x_i, x_j], dim=1)
            msg = self.edge_lin(msg)
        else:
            raise ValueError(f'Invalid message passing variant {self.mp_type}')
        return msg


def build_gcn_block(
    input_dim: int,
    hidden_dim: int,

    proj_dim: Optional[int] = None,
    use_batchnorm: bool = True,

    mp_type: str = 'node_concate',
    edge_emb_dim: int = 256,
    bucket_sz: float = 0.05,
    dropout: float = 0.0,
    use_abs_weight: bool = True,
):
    proj_dim = hidden_dim if proj_dim is None else proj_dim
    return pygSequential(
        'x, edge_index, edge_attr',
        [
            (
                MPGCNConv(
                    input_dim,
                    hidden_dim,
                    mp_type=mp_type,
                    edge_emb_dim=edge_emb_dim,
                    bucket_sz=bucket_sz,
                    dropout=dropout,
                    use_abs_weight=use_abs_weight,
                ),
                'x, edge_index, edge_attr -> x'
            ),
            # project concatenated head embeds back into proj_dim (hidden_dim)
            nn.Linear(hidden_dim, proj_dim),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.BatchNorm1d(proj_dim) if use_batchnorm else nn.Identity(),
        ]
    )


class bgbGCN(torch.nn.Module):
    def __init__(
        self,
        # determined by dataset
        input_dim: int,
        num_nodes: int,
        model_cfg: bgbGNNConfig, # ModelConfig,
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        self.convs = torch.nn.ModuleList()
        self.num_nodes = num_nodes
        self.pooling = model_cfg.pooling

        num_classes = model_cfg.n_classes
        hidden_dim = model_cfg.hidden_dim
        num_layers = model_cfg.num_layers
        mp_type = model_cfg.mp_type
        dropout = model_cfg.dropout
        use_batchnorm = model_cfg.use_batchnorm
        use_abs_weight = model_cfg.use_abs_weight

        edge_emb_dim = model_cfg.edge_emb_dim
        bucket_sz = model_cfg.bucket_sz

        gcn_input_dim = input_dim
        common_args: dict[str, Any] = dict(
            mp_type=mp_type,
            edge_emb_dim = edge_emb_dim,
            bucket_sz=bucket_sz,
            dropout=dropout,
            use_abs_weight=use_abs_weight,
        )
        for i in range(num_layers - 1):
            conv = build_gcn_block(
                gcn_input_dim,
                hidden_dim,
                proj_dim=None,
                use_batchnorm=use_batchnorm,
                **common_args,
            )
            gcn_input_dim = hidden_dim
            self.convs.append(conv)

        fcn_dim = -1
        self.prepool: nn.Module = nn.Identity()
        # last conv is different for each type of pooling
        if self.pooling == "concat":
            # gat block return embeddings of`inter_dim` size
            conv = build_gcn_block(
                gcn_input_dim,
                hidden_dim,
                proj_dim=model_cfg.prepool_dim,
                use_batchnorm=False,
                **common_args,
            )
            # add extra projection and batchnorm (prepool)
            self.prepool = nn.Sequential(
                nn.Linear(model_cfg.prepool_dim, model_cfg.final_node_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.BatchNorm1d(model_cfg.final_node_dim) if use_batchnorm else nn.Identity(),
            )

            fcn_dim = model_cfg.final_node_dim * num_nodes
        elif self.pooling == 'sum' or self.pooling == 'mean':
            conv = build_gcn_block(
                gcn_input_dim,
                hidden_dim,

                proj_dim=model_cfg.final_node_dim,
                use_batchnorm=use_batchnorm,
                **common_args,
            )
            fcn_dim = model_cfg.final_node_dim

        # add last conv layer
        self.convs.append(conv)

        self.fcn = BasicMLP(in_size=fcn_dim, out_size=num_classes, config=model_cfg.mlp_config)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        z = x

        for i, conv in enumerate(self.convs):
            # batch_size * num_nodes, hidden
            z = conv(z, edge_index, edge_attr)

        # prepool dim reduction
        z = self.prepool(z)

        if self.pooling == "concat":
            z = concat_pool(z, self.num_nodes)
        elif self.pooling == 'sum':
            z = global_add_pool(z,  batch)  # [N, F]
        elif self.pooling == 'mean':
            z = global_mean_pool(z, batch)  # [N, F]

        out = self.fcn(z)
        return out
