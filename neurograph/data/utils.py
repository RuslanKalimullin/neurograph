""" Utils functions for converting CM to graphs; generating dataset splits etc. """

from functools import wraps
from typing import Optional

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch_geometric.data import Data


def square_check(f):
    """ Checks that the first argument is a square 2D numpy array """

    @wraps(f)
    def wrapper(*args, **kwargs):
        m = args[0]
        assert isinstance(m, np.ndarray), 'input matrix must be np.ndarray!'
        assert m.ndim == 2, 'input matrix must be 2d array!'
        assert m.shape[0] == m.shape[1], 'input matrix must be square!'

        return f(*args, **kwargs)
    return wrapper


def normalize_cm(cm: np.ndarray, normalize_type: Optional[str] = None) -> np.ndarray:
    """ Normalize weighted adjacency matrix
        e.g. apply log or divide by global matrix max weight
    """
    cm = cm.astype(np.float32)
    if normalize_type:
        if normalize_type == 'global_max':
            cm = cm / cm.max()
        elif normalize_type == 'log':
            # this line handles zeros in CM (set log(0) to 0)
            cm = np.log(cm, where=0<cm, out=0.*cm)
        else:
            raise ValueError(f'Unknown `normalize` arg! Given {normalize_type}')

    return cm

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
    cm = normalize_cm(cm, normalize)

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

    """ For a given CM find a threshold so after sparsification
        the new CM will have `k * num_nodes` edges
    """

    n = cm.shape[0]
    abs_cm = np.abs(cm)

    # find thr to get the desired k
    # = average number of edges for a node
    vals = np.sort(abs_cm.ravel())
    thr_idx = min(max(0, n**2 - 2*k*n - 1), n**2 - 1)
    thr = vals[thr_idx]

    return thr


def generate_splits(subj_ids: list | np.ndarray, y: np.ndarray, seed: int = 1380):
    """ Generate dict with splits: first split to train/test, then
        split train into 5 folds w/ train and valid
    """

    # split into train/test
    subj_ids = np.array(subj_ids)
    idx = np.arange(len(subj_ids))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, stratify=y, shuffle=True, random_state=seed,
    )

    train, y_train = subj_ids[train_idx], y[train_idx]
    test, _ = subj_ids[test_idx], y[test_idx]

    # split train into cv folds
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds: dict[str, list] = {'train': []}
    for _, (train_fold, valid_fold) in enumerate(cv.split(train, y_train)):
        folds['train'].append({
            'train': list(train[train_fold]),
            'valid': list(train[valid_fold]),
        })
    folds['test'] = list(test)

    return folds


def get_subj_ids_from_folds(id_folds) -> list[str]:
    """ Given a dict with splits (each subset is a list of subject ids),
        collect all subject ids into one list
    """

    subj_ids = []

    if set(id_folds.keys()) == set(['train', 'test']):
        train_folds = id_folds['train']
    else:
        # special case of old splits when we don't have 'train' key,
        # but have integer keys for each train fold
        train_idx = sorted([k for k in id_folds.keys() if isinstance(k, int)])
        train_folds = [id_folds[i] for i in train_idx]

    for fold in train_folds:
        train_ids, valid_ids = fold['train'], fold['valid']
        subj_ids.extend(train_ids)
        subj_ids.extend(valid_ids)

    subj_ids.extend(id_folds['test'])

    return list(set(subj_ids))
