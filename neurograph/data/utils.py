from functools import wraps
from typing import Optional
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch_geometric.data import Data


def square_check(f):
    # decotated function must take a np.ndarray as the first argument

    @wraps(f)
    def wrapper(*args, **kwargs):
        m = args[0]
        assert isinstance(m, np.ndarray), 'input matrix must be np.ndarray!'
        assert m.ndim == 2, 'input matrix must be 2d array!'
        assert m.shape[0] == m.shape[1], 'input matrix must be square!'

        return f(*args, **kwargs)
    return wrapper


def load_cms(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[int, str]]:

    """ Load connectivity matrices, fMRI time series
        and mapping node idx -> ROI name.

        Maps sibj_id to CM and ts
    """

    path = Path(path)

    data = {}
    ts = {}
    # ROI names, extacted from CMs
    roi_map: dict[int, str] = {}

    for p in path.glob('*.csv'):
        name = p.stem.split('_')[0].replace('sub-', '')
        x = pd.read_csv(p).drop('Unnamed: 0', axis=1)

        values = x.values.astype(np.float32)
        if p.stem.endswith('_embed'):
            ts[name] = values
        else:
            data[name] = values
            if not roi_map:
                roi_map = {i: c for i, c in enumerate(x.columns)}

    return data, ts, roi_map


@square_check
def prepare_one_graph(
    cm: np.ndarray,
    subj_id: str,
    targets: pd.DataFrame,
    thr: Optional[float] = None,
) -> Data:

    """ Combine CM, subj_id and target to a pyg.Data instance

        targets must be indexed by subj_id
    """

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


def generate_splits(subj_ids: list | np.ndarray, y: np.ndarray, seed: int = 1380):
    # split into train/test
    subj_ids = np.array(subj_ids)
    idx = np.arange(len(subj_ids))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y, shuffle=True, random_state=seed)

    train, y_train = subj_ids[train_idx], y[train_idx]
    test, y_test = subj_ids[test_idx], y[test_idx]

    # split train into cv folds
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds: dict[str, list] = {}
    for i, (train_fold, valid_fold) in enumerate(cv.split(train, y_train)):
        folds['train'].append({
            'train': list(train[train_fold]),
            'valid': list(train[valid_fold]),
        })
    folds['test'] = list(test)

    return folds

