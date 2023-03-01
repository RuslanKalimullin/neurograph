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
def prepare_graph(
    cm: np.ndarray,
    subj_id: str,
    targets: pd.DataFrame,
    abs_thr: Optional[float] = None,
    pt_thr: Optional[float] = None,
    normalize=None,
) -> Data:

    """
    Args:
        cm (np.ndarray): connectivity matrix
        subj_id (str): subject_id
        abs_thr (float, optional): Absolute threshold for sparsification
        pt_thr (float, optional): Proportional threshold for sparsification (pt_thr must be (0, 1)
        Combine CM, subj_id and target to a pyg.Data object
        `targets` must be indexed by subj_id
    """

    cm = cm.astype(np.float32)
    if normalize:
        if normalize == 'global_max':
            cm = cm / cm.max()
        else:
            raise ValueError(f'Unknown `normalize` arg! Given {normalize}')

    # convert CM edge_index, edge_attr (and sparsify if thr are given)
    edge_index, edge_attr = cm_to_edges(cm, abs_thr=abs_thr, pt_thr=pt_thr)

    # compute initial node embeddings -> just original weights
    x = torch.from_numpy(cm).float()

    # get labels from DF via subject_id
    y = torch.LongTensor(targets.loc[subj_id].values)

    data = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        x=x,
        num_nodes=cm.shape[0],
        y=y,
        subj_id=subj_id,
    )
    #data.validate()
    return data


@square_check
def cm_to_edges(
    cm: np.ndarray,
    abs_thr: Optional[float] = None,
    pt_thr: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert CM to (edge_index, edge_weights) of a fully connected weighted graph
    (including self-loops with zero weights)

    return: (edge_index, edge_weights)
    """

    if abs_thr is not None and pt_thr is not None:
        raise ValueError('`abs_thr` and `pt_thr` are both not None! Choose one!')

    if abs_thr:
        abs_cm = np.abs(cm)
        idx = np.nonzero(abs_cm > abs_thr)
    elif pt_thr:
        assert 0 < pt_thr < 1, 'thr must be in range (0, 1)'
        abs_cm = np.abs(cm)
        vals = np.sort(abs_cm.flatten())[::-1]
        top_k = int(pt_thr * cm.shape[0] ** 2)  # pt * num_nodes**2
        idx = np.nonzero(abs_cm > vals[top_k])
    else:
        idx = (np.isnan(cm) == 0).nonzero()

    edge_index = torch.LongTensor(np.stack(idx))
    edge_weights = torch.FloatTensor(cm[idx])

    return edge_index, edge_weights


@square_check
def find_thr(
    cm: np.ndarray,
    k: int = 5,
) -> float:

    ''' For a given CM find a threshold so after sparsification
    the new CM will have `k * num_nodes` edges '''

    n = cm.shape[0]
    abs_cm = np.abs(cm)

    # find thr to get the desired k
    # = average number of edges for a node
    vals = np.sort(abs_cm.ravel())
    thr_idx = min(max(0, n**2 - 2*k*n - 1), n**2 - 1)
    thr = vals[thr_idx]

    return thr


def generate_splits(subj_ids: list | np.ndarray, y: np.ndarray, seed: int = 1380):
    # split into train/test
    subj_ids = np.array(subj_ids)
    idx = np.arange(len(subj_ids))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y, shuffle=True, random_state=seed)

    train, y_train = subj_ids[train_idx], y[train_idx]
    test, y_test = subj_ids[test_idx], y[test_idx]

    # split train into cv folds
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds: dict[str, list] = {'train': []}
    for i, (train_fold, valid_fold) in enumerate(cv.split(train, y_train)):
        folds['train'].append({
            'train': list(train[train_fold]),
            'valid': list(train[valid_fold]),
        })
    folds['test'] = list(test)

    return folds
