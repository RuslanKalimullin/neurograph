""" Module provides implementation of Multimodal version of Vanilla Transformer
    with different cross-modality attention mechanisms
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from neurograph.config import (
    MLPConfig, MLPlayer, MultiModalTransformerConfig, TransformerConfig
)
from neurograph.models.mlp import BasicMLP
from neurograph.models.transformers import (
    MSA, MSAConfig, Transformer, TransformerBlock
)


@dataclass
class MSACrossAttentionOutput:
    """ Dataclass that stores cross-attention
        operation output
    """
    x_1: torch.Tensor
    x_2: torch.Tensor
    attn_1: Optional[torch.Tensor] = None
    attn_2: Optional[torch.Tensor] = None


def compute_raw_attn_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """ Multiply Q and K for MSA """
    return torch.einsum('bihp,bjhp->bhij', q, k)


class MSACrossAttention(MSA):
    """ Multihead cross attention block that accepts two modalities x1 and x2
        and computes attention weights between them
    """
    def __init__(
        self,
        input_dim: int,
        cfg: MSAConfig,
    ):
        super().__init__(input_dim, cfg)

        self.q_lin_head2 = nn.Linear(self.input_dim, self.hidden_dim)
        self.k_lin_head2 = nn.Linear(self.input_dim, self.hidden_dim)
        self.v_lin_head2 = nn.Linear(self.input_dim, self.hidden_dim)
        self.dropout2 = nn.Dropout(self.dropout_rate)

    def project_to_qkv_head1(self, x: torch.Tensor):
        """ project token embeddings to query, keys and values """
        return self.project_to_qkv(x)

    def project_to_qkv_head2(self, x: torch.Tensor):
        """ project token embeddings to query, keys and values
            (weird and ugly hack for 2 modalities)
        """
        # (b, n, d)
        b, n, _ = x.shape
        h = self.num_heads
        p = self.head_dim

        q = self.q_lin_head2(x).reshape(b, n, h, p)
        k = self.k_lin_head2(x).reshape(b, n, h, p)
        v = self.v_lin_head2(x).reshape(b, n, h, p)

        return q, k, v

    # pylint: disable=too-many-locals
    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor) -> MSACrossAttentionOutput:
        # (b, n, d)
        b, n, _ = x_1.shape
        # h is self.num_heads

        # project X1 to Q1, K1, V1 -> (b, n, h, p)
        q_1, k_1, v_1 = self.project_to_qkv_head1(x_1)

        # project X2 to Q2, K2, V2 -> (b, n, h, p)
        q_2, k_2, v_2 = self.project_to_qkv_head2(x_2)

        # compute raw_scores for modality x1
        raw_scores_1 = compute_raw_attn_scores(q_2, k_1)

        # compute raw_scores for modality x2
        raw_scores_2 = compute_raw_attn_scores(q_1, k_2)

        # normalize each head output
        scores_1 = torch.softmax(raw_scores_1 * self.factor, dim=-1)
        scores_2 = torch.softmax(raw_scores_2 * self.factor, dim=-1)

        # save attention matrices for each head
        saved_scores_1, saved_scores_2 = None, None
        if self.return_attn:
            saved_scores_1 = scores_1.clone()
            saved_scores_2 = scores_2.clone()

        # apply dropout to attention matrix
        scores_1 = self.dropout(scores_1)
        scores_2 = self.dropout2(scores_2)

        # compute final embeddings
        out1 = torch.einsum('bhij,bjhp->bihp', scores_1, v_1)
        out2 = torch.einsum('bhij,bjhp->bihp', scores_2, v_2)

        # 'concat' each head output
        return MSACrossAttentionOutput(
            x_1=out1.reshape(b, n, -1),
            x_2=out2.reshape(b, n, -1),
            attn_1=saved_scores_1,
            attn_2=saved_scores_2,
        )


class HierarchicalAttentionBlock(nn.Module):
    """ Hierarchical attention transformer block for 2 modalities.
        Runs 2 transformer blocks in parallel for each X
        and then concats outputs
    """

    def __init__(
        self,
        input_dim_1: int,
        input_dim_2: int,
        msa_cfg: MSAConfig,
        mlp_cfg: MLPConfig,
    ):
        super().__init__()
        self.block_1 = TransformerBlock(input_dim_1, msa_cfg, mlp_cfg)
        self.block_2 = TransformerBlock(input_dim_2, msa_cfg, mlp_cfg)

    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor):
        z_1 = self.block_1(x_1)
        z_2 = self.block_2(x_2)
        return torch.cat((z_1, z_2), dim=2)


class FFN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mlp_cfg: MLPConfig,
    ):
        super().__init__()

        self.mlp = BasicMLP(
            in_size=input_dim,
            out_size=output_dim,
            config=mlp_cfg,
        )
        self.ln_1 = nn.LayerNorm([input_dim])
        self.ln_2 = nn.LayerNorm([input_dim])

    def forward(self, x, z_1):
        # https://arxiv.org/pdf/2002.04745.pdf
        x = self.ln_1(x)
        s_1 = x + z_1  # sum_1

        z_2 = self.ln_2(s_1)
        s_2 = s_1 + self.mlp(z_2)  # sum_2
        return s_2


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        input_dim_1: int,
        input_dim_2: int,
        msa_cfg: MSAConfig,
        mlp_cfg: MLPConfig,
    ):
        super().__init__()

        assert input_dim_1 == input_dim_2
        self.hidden_dim = msa_cfg.hidden_dim

        self.msa = MSACrossAttention(input_dim_1, msa_cfg)
        self.head1 = FFN(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            mlp_cfg=mlp_cfg,
        )
        self.head2 = FFN(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            mlp_cfg=mlp_cfg,
        )

    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor):
        out = self.msa(x_1, x_2)
        z_1 = self.head1(x_1, out.x1)
        z_2 = self.head2(x_2, out.x2)
        return torch.cat((z_1, z_2), dim=2)


class MultiModalTransformer(Transformer):
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        input_dim_1: int,
        input_dim_2: int,
        num_nodes_1: int,  # used for concat pooling
        num_nodes_2: int,
        model_cfg: MultiModalTransformerConfig,
    ):
        assert num_nodes_1 == num_nodes_2

        self.attn_type = model_cfg.attn_type
        self.make_projection = model_cfg.make_projection

        if self.attn_type in ["sum", "multiply"] and not self.make_projection:
            assert input_dim_1 == input_dim_2

        transformer_hidden_dim = model_cfg.hidden_dim

        if self.attn_type == "concat":
            transformer_hidden_dim = (
                model_cfg.hidden_dim if self.make_projection else input_dim_1 + input_dim_2
            )
            super().__init__(transformer_hidden_dim, num_nodes_1, model_cfg)
        elif self.attn_type in ["sum", "multiply"]:
            transformer_hidden_dim = model_cfg.hidden_dim if self.make_projection else input_dim_1
            super().__init__(transformer_hidden_dim, num_nodes_1, model_cfg)
        else:
            super().__init__(transformer_hidden_dim, num_nodes_1, model_cfg)

        if self.make_projection:
            self.lin_proj1 = nn.Linear(input_dim_1, model_cfg.hidden_dim // 2)
            self.lin_proj2 = nn.Linear(input_dim_2, model_cfg.hidden_dim // 2)

            attn_block_params = [
                model_cfg.hidden_dim // 2,
                model_cfg.hidden_dim // 2,
                self.build_msa_attn_cfg(model_cfg),
                self.build_mlp_attn_cfg(model_cfg),
            ]
        else:
            attn_block_params = [
                input_dim_1,
                input_dim_2,
                self.build_msa_attn_cfg(model_cfg),
                self.build_mlp_attn_cfg(model_cfg),
            ]

        if self.attn_type == "hierarchical":
            self.mm_block = HierarchicalAttentionBlock(*attn_block_params)
        elif self.attn_type == "cross-attention":
            self.mm_block = CrossAttentionBlock(*attn_block_params)

    def forward(self, batch):
        x_fmri, x_dti, y = batch
        if self.make_projection:
            x_fmri = self.lin_proj1(x_fmri)
            x_dti = self.lin_proj2(x_dti)

        if self.attn_type == "concat":
            x = torch.cat((x_fmri, x_dti), dim=2)
        elif self.attn_type == "sum":
            x = x_fmri + x_dti
        elif self.attn_type == "multiply":
            x = x_fmri * x_dti
        elif self.attn_type in ["hierarchical", "cross-attention"]:
            x =self.mm_block(x_fmri, x_dti)
        else:
            raise ValueError('Invalid attn_type')
        return super().forward((x,y))

    @staticmethod
    def build_msa_attn_cfg(cfg: TransformerConfig):
        return MSAConfig(
            num_heads=cfg.num_heads,
            hidden_dim=cfg.hidden_dim // 2,
            dropout=cfg.attn_dropout,
            return_attn=cfg.return_attn,
    )

    @staticmethod
    def build_mlp_attn_cfg(cfg: TransformerConfig):
        # 2-layer MLP
        return MLPConfig(
            # no act func on the output of MLP
            layers=[
                MLPlayer(
                    out_size=int((cfg.hidden_dim // 2) * cfg.mlp_hidden_multiplier),
                    dropout=cfg.mlp_dropout,
                    act_func=cfg.mlp_act_func,  # put class name here
                    act_func_params=cfg.mlp_act_func_params,
                ),
            ],
        )
