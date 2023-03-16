""" Simple model for testing Multimodal dense dataset """

import torch
from torch import nn

from neurograph.config.config import DummyMultimodalDense2Config
from neurograph.models.available_modules import available_activations


class DummyMultimodalDense2Model(nn.Module):
    """ Simple model for testing Multimodal dense dataset """
    def __init__(
        self,
        # comes from dataset
        input_dim_1: int,
        input_dim_2: int,
        num_nodes_1: int,  # used for concat pooling
        num_nodes_2: int,  # used for concat pooling
        model_cfg: DummyMultimodalDense2Config,
    ):
        super().__init__()
        self.lin_1 = nn.Linear(input_dim_1 * num_nodes_1, model_cfg.hidden)
        self.lin_2 = nn.Linear(input_dim_2 * num_nodes_2, model_cfg.hidden)
        self.lin_3 = nn.Linear(2 * model_cfg.hidden, model_cfg.n_classes)

        act_func: str = model_cfg.act_func if model_cfg.act_func else ''
        act_params = model_cfg.act_func_params if model_cfg.act_func_params else {}
        if act_func:
            self.act = available_activations[act_func](**act_params)
        else:
            self.act = nn.Identity()

        self.dropout = nn.Dropout(model_cfg.dropout)

    # pylint: disable=missing-function-docstring
    def forward(self, batch):
        x_1, x_2, _ = batch

        inp1 = x_1.reshape(x_1.size(0), -1)
        inp2 = x_2.reshape(x_2.size(0), -1)

        y_1 = self.lin_1(inp1)
        y_2 = self.lin_1(inp2)

        out = self.dropout(self.act(torch.cat([y_1, y_2], axis=-1)))

        return self.lin_3(out)
