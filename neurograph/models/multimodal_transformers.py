import math
from typing import Optional

import torch
import torch.nn as nn
from .transformers import *
from neurograph.config import MultiModalTransformerConfig
from dataclasses import dataclass, field


@dataclass
class MSACrossAttentionOutput:
    x1: torch.Tensor
    x2: torch.Tensor
    attn1: Optional[torch.Tensor] = None
    attn2: Optional[torch.Tensor] = None


class MSACrossAttention(MSA):

    def __init__(
        self,
        input_dim: int,
        cfg: MSAConfig,
    ):
        super().__init__(input_dim,cfg)

        self.q_lin_head2 = nn.Linear(self.input_dim, self.hidden_dim)
        self.k_lin_head2 = nn.Linear(self.input_dim, self.hidden_dim)
        self.v_lin_head2 = nn.Linear(self.input_dim, self.hidden_dim)

        self.dropout2 = nn.Dropout(self.dropout_rate)

    def project_to_qkv_head1(self, x: torch.Tensor):
        return self.project_to_qkv(x)

    def project_to_qkv_head2(self, x: torch.Tensor):
        b, n, d = x.shape
        h = self.num_heads
        p = self.head_dim

        q = self.q_lin_head2(x).reshape(b, n, h, p)
        k = self.k_lin_head2(x).reshape(b, n, h, p)
        v = self.v_lin_head2(x).reshape(b, n, h, p)

        return q, k, v


    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> MSACrossAttentionOutput:
            b, n, d = x1.shape
            h = self.num_heads

            # project X1 to Q1, K1, V1 -> (b, n, h, p)
            q1, k1, v1 = self.project_to_qkv_head1(x1)

            # project X2 to Q2, K2, V2 -> (b, n, h, p)
            q2, k2, v2 = self.project_to_qkv_head2(x2)

            # compute raw_scores for modality x1
            raw_scores1 = torch.einsum('bihp,bjhp->bhij', q2, k1)

            # compute raw_scores for modality x2
            raw_scores2 = torch.einsum('bihp,bjhp->bhij', q1, k2)

            # normalize each head output
            scores1 = torch.softmax(raw_scores1 * self.factor, dim=-1)
            scores2 = torch.softmax(raw_scores2 * self.factor, dim=-1)

            # save attention matrices for each head
            saved_scores1,saved_scores2  = None, None
            if self.return_attn:
                saved_scores1 = scores1.clone()
                saved_scores2 = scores2.clone()

            # apply dropout to attention matrix
            scores1 = self.dropout(scores1)
            scores2 = self.dropout2(scores2)

            # compute final embeddings
            out1 = torch.einsum('bhij,bjhp->bihp', scores1, v1)
            out2 = torch.einsum('bhij,bjhp->bihp', scores2, v2)

            # 'concat' each head output
            return MSACrossAttentionOutput(x1=out1.reshape(b, n, -1),
                             x2=out1.reshape(b, n, -1),
                             attn1=saved_scores1,
                             attn2=saved_scores2)



class HierarchicalAttentionBlock(nn.Module):
    def __init__(
        self,
        input_dim_1: int,
        input_dim_2: int,
        msa_cfg: MSAConfig,
        mlp_cfg: MLPConfig,
    ):
        super().__init__()
        self.head1 =TransformerBlock(input_dim_1,msa_cfg, mlp_cfg)
        self.head2 =TransformerBlock(input_dim_2,msa_cfg, mlp_cfg)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
            z1 =self.head1(x1)
            z2 =self.head2(x2)
            return torch.cat((z1, z2), dim=2)

class FFN(nn.Module):
    def __init__(self,
        input_dim: int,
        output_dim: int,
        mlp_cfg: MLPConfig):

        super().__init__()
       
        self.mlp = BasicMLP(
                in_size=input_dim,
                out_size=output_dim,
                config=mlp_cfg,
            )
        self.ln1 = nn.LayerNorm([input_dim])
        self.ln2 = nn.LayerNorm([input_dim])

    def forward(self, x, z1):
            # https://arxiv.org/pdf/2002.04745.pdf
            x = self.ln1(x)
            s1 = x + z1  # sum_1

            z2 = self.ln2(s1)
            s2 = s1 + self.mlp(z2)  # sum_2
            return s2


class CrossAttentionBlock(nn.Module):
    def __init__(self,
        input_dim_1: int,
        input_dim_2: int,
        msa_cfg: MSAConfig,
        mlp_cfg: MLPConfig):

        super().__init__()

        ##TODO release cross-attention with different dims
        assert input_dim_1==input_dim_2
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
        

    def forward(self, x1: torch.Tensor, x2: torch.Tensor): 
         out =self.msa(x1,x2)
         z1= self.head1(x1,out.x1)
         z2= self.head2(x2,out.x2)
         return torch.cat((z1, z2), dim=2)
      

class MultiModalTransformer(Transformer):

    def __init__(self,
        input_dim_1: int,
        input_dim_2: int,
        num_nodes_1: int,  # used for concat pooling
        num_nodes_2: int,
        model_cfg: MultiModalTransformerConfig):
        
        assert num_nodes_1==num_nodes_2

        self.attn_type=model_cfg.attn_type
        self.make_projection =model_cfg.make_projection

        if self.attn_type in ["sum", "multiply"] and not self.make_projection:
            assert input_dim_1==input_dim_2

        transformer_hidden_dim = model_cfg.hidden_dim

        if self.attn_type=="concat":
            transformer_hidden_dim = model_cfg.hidden_dim if self.make_projection else input_dim_1+input_dim_2
            super().__init__(transformer_hidden_dim, num_nodes_1,model_cfg)
        elif self.attn_type in ["sum", "multiply"]:
            transformer_hidden_dim = model_cfg.hidden_dim if self.make_projection else input_dim_1
            super().__init__(transformer_hidden_dim, num_nodes_1,model_cfg)    
        else:    
            super().__init__(transformer_hidden_dim, num_nodes_1,model_cfg)

        if self.make_projection:
            self.lin_proj1 = nn.Linear(input_dim_1, model_cfg.hidden_dim // 2)
            self.lin_proj2 = nn.Linear(input_dim_2, model_cfg.hidden_dim // 2)

            attn_block_params =[model_cfg.hidden_dim // 2 ,model_cfg.hidden_dim // 2, self.build_msa_attn_cfg(model_cfg), self.build_mlp_attn_cfg(model_cfg)]  
        else:
            attn_block_params =[input_dim_1,input_dim_2, self.build_msa_attn_cfg(model_cfg), self.build_mlp_attn_cfg(model_cfg)]

        if self.attn_type == "hierarchical":
            self.mm_block = HierarchicalAttentionBlock(*attn_block_params)
        elif self.attn_type == "cross-attention":
            self.mm_block = CrossAttentionBlock(*attn_block_params)

    def forward(self, batch):
        x_fmri, x_dti, y = batch
        if self.make_projection:
            x_fmri=self.lin_proj1(x_fmri)
            x_dti =self.lin_proj2(x_dti)

        if self.attn_type=="concat":
            x =torch.cat((x_fmri, x_dti), dim=2)
        elif self.attn_type=="sum":
            x =x_fmri+x_dti
        elif self.attn_type=="multiply":
            x =x_fmri*x_dti    
        elif self.attn_type in ["hierarchical", "cross-attention"]:
            x =self.mm_block(x_fmri, x_dti)
        else:
            raise ValueError(f'Invalid attn_type')    
        return super().forward((x,y))


    @staticmethod
    def build_msa_attn_cfg(cfg: TransformerConfig):
        return MSAConfig(
            num_heads=cfg.num_heads,
            hidden_dim=cfg.hidden_dim //2,
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


