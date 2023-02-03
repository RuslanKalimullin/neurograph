from typing import Union, Tuple, Optional

import torch
from torch import nn
from torch import Tensor
from torch.nn import Parameter, Linear
from torch.nn import functional as F
from torch_geometric.typing import Size, OptTensor
from torch_geometric.nn import global_add_pool, global_mean_pool, MessagePassing, GATConv
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax
from torch_geometric.nn import Sequential as pygSequential
import torch_geometric

from neurograph.config import Config, ModelConfig
from neurograph.models.mlp import BasicMLP


class MPGATConv(GATConv):

    # gat_mp_type choices:
    available_mp_types = {
       'attention_weighted',
       'attention_edge_weighted',
       'sum_attention_edge',
       'edge_node_concate',
       'node_concate',
    }

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        # not used
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
        gat_mp_type: str = 'attention_weighted',
        use_abs_weight=True,
    ):
        ''' custom GATConv layer with custom message passaging procedure '''

        super().__init__(in_channels, out_channels, heads)

        self.gat_mp_type = gat_mp_type
        input_dim = out_channels

        self.abs_weights = use_abs_weight
        self.dropout = dropout

        if gat_mp_type == "edge_node_concate":
            input_dim = out_channels * 2 + 1
        elif gat_mp_type == "node_concate":
            input_dim = out_channels * 2

        # edge_lin is mandatory
        self.edge_lin = torch.nn.Linear(input_dim, out_channels)

    # define our message passage function
    def message(
        self,
        x_i, x_j,  # node embeddgins lifted to edges
        alpha_j, alpha_i,  # attention weights per node lifted to edges
        edge_attr,
        index,
        ptr,
        size_i,
    ):
        # x_j: [num edges, num heads, num channels (out dim)]

        # copied from PyG
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # add extra dim, so we get [m, 1, 1] shape where m = num of edges
        attention_score = alpha.unsqueeze(-1)

        # reshape and apply `abs` to edge_attr (edge weights)
        edge_weights = edge_attr.view(-1, 1).unsqueeze(-1)
        if self.abs_weights:
            edge_weights = torch.abs(edge_weights)

        # compute messages (for each edge)
        if self.gat_mp_type == "attention_weighted":
            # (1) att: s^(l+1) = s^l * alpha
            msg = x_j * attention_score
            return msg
        elif self.gat_mp_type == "attention_edge_weighted":
            # (2) e-att: s^(l+1) = s^l * alpha * e
            msg = x_j * attention_score * edge_weights
            return msg
        elif self.gat_mp_type == "sum_attention_edge":
            # (3) m-att-1: s^(l+1) = s^l * (alpha + e),
            # this one may not make sense cause it doesn't use attention score to control all
            msg = x_j * (attention_score + edge_weights)
            return msg
        elif self.gat_mp_type == "edge_node_concate":
            # (4) m-att-2: s^(l+1) = linear(concat(s^l, e) * alpha)
            msg = torch.cat(
                [
                    x_i,
                    x_j * attention_score,
                    # reshape and expand to the given num of heads
                    # Note that we do not absolute values of weights here!
                    edge_weights.expand(-1, self.heads, -1),
                ],
                dim=-1,
            )
            msg = self.edge_lin(msg)
            return msg
        elif self.gat_mp_type == "node_concate":
            # (4) m-att-2: s^(l+1) = linear(concat(s^l, e) * alpha)
            msg = torch.cat([x_i, x_j * attention_score], dim=-1)
            msg = self.edge_lin(msg)
            return msg
        # elif self.gat_mp_type == "sum_node_edge_weighted":
        #     # (5) m-att-3: s^(l+1) = (s^l + e) * alpha
        #     node_emb_dim = x_j.shape[-1]
        #     extended_edge = torch.cat([edge_weights] * node_emb_dim, dim=-1)
        #     sum_node_edge = x_j + extended_edge
        #     msg = sum_node_edge * attention_score
        #     return msg
        else:
            raise ValueError(f'Invalid message passing variant {self.gat_mp_type}')


def build_gat_block(
    input_dim: int,
    hidden_dim: int,
    proj_dim: Optional[int] = None,
    mp_type: str = 'edge_node_concate',
    num_heads: int = 1,
    dropout: float = 0.0,
    use_batchnorm: bool = True,
    use_abs_weight: bool = True,
):
    proj_dim = hidden_dim if proj_dim is None else proj_dim
    return pygSequential(
        'x, edge_index, edge_attr',
        [
            (
                MPGATConv(
                    input_dim,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    gat_mp_type=mp_type,
                    use_abs_weight=use_abs_weight,
                ),
                'x, edge_index, edge_attr -> x'
            ),
            # project concatenated head embeds back into proj_dim (hidden_dim)
            nn.Linear(hidden_dim * num_heads, proj_dim),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.BatchNorm1d(proj_dim) if use_batchnorm else nn.Identity(),
        ]
    )


def concat_pool(x: torch.Tensor, num_nodes: int) -> torch.Tensor:
    # NB: x must be a batch of xs
    return x.reshape(x.size(0) // num_nodes, -1)


class GAT(nn.Module):
    def __init__(
        self,
        # determined by dataset
        input_dim: int,
        num_nodes: int,
        model_cfg: ModelConfig,
    ):
        """
        Architecture:
            - a list of GATConv blocks (n-1)
            - the last layer GATConv block with diff final embedding size
            - (prepool projection layer)
            - pooling -> graph embeddgin
            - fcn clf
        """
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.num_nodes = num_nodes
        self.pooling = model_cfg.pooling

        num_classes = model_cfg.n_classes
        hidden_dim = model_cfg.hidden_dim
        num_heads = model_cfg.num_heads
        num_layers = model_cfg.num_layers
        use_batchnorm = model_cfg.use_batchnorm
        use_abs_weight = model_cfg.use_abs_weight
        mp_type = model_cfg.mp_type
        dropout = model_cfg.dropout
        # edge_emb_dim = args.edge_emb_dim # not used

        # pack a bunch of convs into a ModuleList
        gat_input_dim = input_dim
        for i in range(num_layers - 1):
            conv = build_gat_block(
                gat_input_dim,
                hidden_dim,
                proj_dim=None,
                mp_type=mp_type,
                num_heads=num_heads,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
                use_abs_weight=use_abs_weight,
            )
            # update current input_dim
            gat_input_dim = hidden_dim
            self.convs.append(conv)

        fcn_dim = -1
        self.prepool: nn.Module = nn.Identity()
        # last conv is different for each type of pooling
        if self.pooling == "concat":
            # gat block return embeddings of`inter_dim` size
            conv = build_gat_block(
                gat_input_dim,
                hidden_dim,
                proj_dim=model_cfg.prepool_dim,
                mp_type=mp_type,
                num_heads=num_heads,
                dropout=dropout,
                use_batchnorm=False,  # batchnorm is applied in prepool layer
                use_abs_weight=use_abs_weight,
            )
            # add extra projection and batchnorm
            self.prepool = nn.Sequential(
                nn.Linear(model_cfg.prepool_dim, model_cfg.final_node_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.BatchNorm1d(model_cfg.final_node_dim) if use_batchnorm else nn.Identity(),
            )

            fcn_dim = model_cfg.final_node_dim * num_nodes
        elif self.pooling == 'sum' or self.pooling == 'mean':
            conv = build_gat_block(
                gat_input_dim,
                hidden_dim,
                proj_dim=model_cfg.final_node_dim,
                mp_type=mp_type,
                num_heads=num_heads,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
                use_abs_weight=use_abs_weight,
            )
            fcn_dim = model_cfg.final_node_dim

        # add last layer and prepool projection
        self.convs.append(conv)

        self.fcn = BasicMLP(in_size=fcn_dim, out_size=num_classes, config=model_cfg.mlp_config)

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        z = x

        # apply conv layers
        for i, conv in enumerate(self.convs):
            # bz * nodes, hidden
            z = conv(z, edge_index, edge_attr)

        # prepool dim reduction
        z = self.prepool(z)

        # pooling
        if self.pooling == "concat":
            z = concat_pool(z, self.num_nodes)
        elif self.pooling == 'sum':
            z = global_add_pool(z,  batch)  # [N, F]
        elif self.pooling == 'mean':
            z = global_mean_pool(z, batch)  # [N, F]
        else:
            ValueError('Unknown concat type')

        # FCN clf one graph embedding
        out = self.fcn(z)
        return out
