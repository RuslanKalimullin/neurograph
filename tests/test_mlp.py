from neurograph.config import MLPConfig, MLPlayer
from neurograph.models.mlp import BasicMLP
import torch


def create_mlp(conf):
    return BasicMLP(in_size=100, out_size=10, config=conf)


def run_test(model):
    x = torch.rand(7, 100)
    o = model(x)
    assert o.shape == (7, 10)
    assert torch.isnan(o).sum() == 0


def test_mlp_1l():
    conf = MLPConfig(act_func='LeakyReLU', act_func_params={'negative_slope': 0.2})
    model = create_mlp(conf)
    run_test(model)


def test_mlp_l1_2():
    conf = MLPConfig()
    model = create_mlp(conf)
    run_test(model)


def test_mlp_l2():
    conf = MLPConfig(layers=[
        MLPlayer(out_size=31, act_func='ReLU', dropout=0.2),
        MLPlayer(out_size=17, act_func='ELU', dropout=0.3, act_func_params={'alpha': 0.1})
    ])
    model = create_mlp(conf)
    run_test(model)
