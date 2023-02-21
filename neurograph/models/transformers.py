import math
from typing import Optional

import torch
import torch.nn as nn

from dataclasses import dataclass, field

from neurograph.models.mlp import BasicMLP
from neurograph.config import MLPConfig, MLPlayer, TransformerConfig


# TODO: move to config maybe?
@dataclass
class MSAConfig:
    num_heads: int
    hidden_dim: int  # token embedding dim after MSA
    return_attn: bool
    dropout: float = 0.


@dataclass
class MSAOutput:
    x: torch.Tensor
    attn: Optional[torch.Tensor] = None


class MSA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cfg: MSAConfig,
    ):
        super().__init__()
        assert cfg.hidden_dim % cfg.num_heads == 0, 'hidden_dim must be multiple of num_heads'

        head_dim = int(cfg.hidden_dim // cfg.num_heads)

        self.input_dim = input_dim
        self.num_heads = cfg.num_heads
        self.hidden_dim = cfg.hidden_dim
        self.dropout_rate = cfg.dropout
        self.return_attn = cfg.return_attn
        self.head_dim = self.hidden_dim // self.num_heads
        self.factor = 1 / math.sqrt(self.head_dim)

        self.q_lin = nn.Linear(self.input_dim, self.hidden_dim)
        self.k_lin = nn.Linear(self.input_dim, self.hidden_dim)
        self.v_lin = nn.Linear(self.input_dim, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def project_to_qkv(self, x: torch.Tensor):
        b, n, d = x.shape
        h = self.num_heads
        p = self.head_dim

        q = self.q_lin(x).reshape(b, n, h, p)
        k = self.k_lin(x).reshape(b, n, h, p)
        v = self.v_lin(x).reshape(b, n, h, p)

        return q, k, v

    def forward(self, x: torch.Tensor) -> MSAOutput:
        b, n, d = x.shape
        h = self.num_heads

        # project X to Q, K, V -> (b, n, h, p)
        q, k, v = self.project_to_qkv(x)

        # compute raw_scores
        raw_scores = torch.einsum('bihp,bjhp->bhij', q, k)

        # normalize each head output
        scores = torch.softmax(raw_scores * self.factor, dim=-1)

        # save attention matrices for each head
        saved_scores = None
        if self.return_attn:
            saved_scores = scores.clone()

        # apply dropout to attention matrix
        scores = self.dropout(scores)

        # compute final embeddings
        out = torch.einsum('bhij,bjhp->bihp', scores, v)

        # 'concat' each head output
        return MSAOutput(x=out.reshape(b, n, -1), attn=saved_scores)


class TransformerBlock(nn.Module):
    """ NB: input_dim must be equal to hidden_dim"""
    def __init__(
        self,
        input_dim: int,
        msa_cfg: MSAConfig,
        mlp_cfg: MLPConfig,
    ):
        super().__init__()
        self.hidden_dim = msa_cfg.hidden_dim
        assert msa_cfg.hidden_dim == input_dim, \
            'First project input to hidden before sending it to TransformerBlock'

        self.msa = MSA(input_dim, msa_cfg)
        self.mlp = BasicMLP(
            in_size=self.hidden_dim,
            out_size=self.hidden_dim,
            config=mlp_cfg,
        )
        self.ln1 = nn.LayerNorm([self.hidden_dim])
        self.ln2 = nn.LayerNorm([self.hidden_dim])

    def forward(self, x):
        # https://arxiv.org/pdf/2002.04745.pdf
        z1 = self.ln1(x)
        s1 = x + self.msa(z1).x  # sum_1


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim: int,  # comes from dataset
        cfg: TransformerConfig,
    ):
        super().__init__()
        #self.cfg = cfg
        self.lin_proj: nn.Module
        if input_dim != cfg.hidden_dim:
            self.lin_proj = nn.Linear(input_dim, cfg.hidden_dim)
        else:
            self.lin_proj = nn.Identity()
        # TODO
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.hidden_dim, self.build_msa_cfg(cfg), self.build_mlp_cfg(cfg))
            for _ in range(cfg.num_layers)
        ])

    def forward(self, x):
        out = self.lin_proj(x)
        for block in self.blocks:
             out = block(out)
        return out

    @staticmethod
    def build_msa_cfg(cfg: TransformerConfig):
        return MSAConfig(
            num_heads=cfg.num_heads,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.attn_dropout,
            return_attn=cfg.return_attn,
    )

    @staticmethod
    def build_mlp_cfg(cfg: TransformerConfig):
        # 2-layer MLP
        return MLPConfig(
            # no act func on the output of MLP
            layers=[
                MLPlayer(
                    out_size=int(cfg.hidden_dim * cfg.mlp_hidden_multiplier),
                    dropout=cfg.mlp_dropout,
                    act_func=cfg.mlp_act_func,  # put class name here
                    act_func_params=cfg.mlp_act_func_params,
                ),
            ],
        )
