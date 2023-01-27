from functools import wraps
from typing import Union
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data


def load_cm(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[int, str]]:

    """ Load connectivity matrices, fMRI time series
        and mapping node idx -> ROI name
    """

    path = Path(path)

    data = {}
    embed = {}

    # ROI names, extacted from CM
    regions: dict[int, str] = {}

    for p in path.glob('*.csv'):

        name = p.stem.split('_')[0].replace('sub-', '')

        x = pd.read_csv(p).drop('Unnamed: 0', axis=1)

        values = x.values.astype(np.float32)
        if p.stem.endswith('_embed'):
            embed[name] = values
        else:
            data[name] = values
            if not regions:
                regions = {i: c for i, c in enumerate(x.columns)}

    return data, embed, regions


def square_check(f):
    # decotated function must take a np.ndarray as the first argument

    @wraps(f)
    def wrapper(*args, **kwargs):
        m = args[0]
        assert isinstance(m, np.ndarray), 'input matrix must be np.ndarray!'
        assert m.ndim == 2, 'input matrix must be 2d array!'
        assert m.shape[0] == cm.shape[1], 'input matrix must be square!'

        return f(*args, **kwargs)
    return wrapper


@square_check
def prepare_pyg_data(
    cm: np.ndarray,
    subj_id: str,
    targets: pd.DataFrame,
) -> Data:

    # fully connected graph
    n = cm.shape[0]
    edge_index, edge_attr = cm_to_edges(cm)

    # compute initial node embeddings -> just original weights
    x = torch.from_numpy(cm)

    # get labels from DF via subject_id
    y = torch.LongTensor(targets.loc[subj_id].values)

    data = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        x=x,
        num_nodes=n,
        y=y,
        subj_id=subj_id,
    )
    #data.validate()
    return data


@square_check
def cm_to_edges(cm: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert CM to (edge_index, edge_weights) of a fully connected weighted graph
    (including self-loops with zero weights)
    return: (edge_index, edge_weights)
    """
    cm_tensor = torch.FloatTensor(cm)
    index = (torch.isnan(cm_tensor) == 0).nonzero(as_tuple=True)
    edge_attr = cm_tensor[index]

    return torch.stack(index, dim=0), edge_attr


@square_check
def find_thr(
    cm: np.ndarray,
    k: int = 5,
) -> float:
    assert cm.ndim == 2, 'adj matrix must be 2d array!'
    assert cm.shape[0] == cm.shape[1], 'adj matrix must be square!'

    n = cm.shape[0]
    abs_cm = np.abs(cm)

    # find thr to get the desired k
    # = average number of edges for a node
    vals = np.sort(abs_cm.ravel())
    thr_idx = min(max(0, n**2 - 2*k*n - 1), n**2 - 1)
    thr = vals[thr_idx]

    return thr


@square_check
def apply_thr(cm: np.ndarray, thr: float):
    abs_cm = np.abs(cm)
    idx = np.nonzero(abs_cm > thr)
    edge_index = torch.LongTensor(np.stack(idx))
    edge_weights = torch.FloatTensor(cm[idx])

    return edge_index, edge_weights
