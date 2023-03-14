from functools import reduce

import pytest
from neurograph.config import get_config
from neurograph.data.ppmi import PPMIDenseDataset, PPMIGraphDataset

from .conftest import data_path


@pytest.fixture(scope='session')
def ppmi_ds_no_thr():
    return PPMIGraphDataset(root=data_path, experiment_type='dti',
        normalize='log',
        no_cache=True,
    )


@pytest.fixture(scope='session')
def ppmi_ds_abs_thr():
    return PPMIGraphDataset(root=data_path, experiment_type='dti', abs_thr=0.3,
        normalize='log',
        no_cache=True,
    )


@pytest.fixture(scope='session')
def ppmi_ds_pt_thr():
    return PPMIGraphDataset(
        root=data_path,
        experiment_type='dti',
        pt_thr=0.5,
        normalize='log',
        no_cache=True,
    )


@pytest.fixture(scope='session')
def ppmi_dense_connprofile():
    return PPMIDenseDataset(
        root=data_path,
        experiment_type='dti',
        normalize='global_max',
        feature_type='conn_profile',
    )


def test_ppmi_no_thr(ppmi_ds_no_thr):
    g = ppmi_ds_no_thr
    assert g[0].edge_index.shape[1] == ppmi_ds_no_thr.num_nodes ** 2


def test_ppmi_abs_thr(ppmi_ds_abs_thr):
    g = ppmi_ds_abs_thr[0]
    assert 0 < g.edge_index.shape[1] < ppmi_ds_abs_thr.num_nodes ** 2
    assert g.edge_attr.min().abs() >= 0.5


def test_ppmi_pt_thr(ppmi_ds_pt_thr):
    g = ppmi_ds_pt_thr[0]
    p = ppmi_ds_pt_thr.pt_thr * (ppmi_ds_pt_thr.num_nodes ** 2)
    # Since we have a lot of 0s in DTI matrix,
    # while filtering by some threshold, we remove all zeros

    #assert p // 2 < g.edge_index.shape[1]
    assert g.edge_index.shape[1] <= int(p)


def test_ppmi_target(ppmi_ds_no_thr):
    target, label2idx, idx2label = ppmi_ds_no_thr.load_targets()
    target = target[ppmi_ds_no_thr.target_col]

    assert len(idx2label) == 2
    assert target.nunique() == 2
    assert target.isnull().sum() == 0

    assert target.index.isnull().sum() == 0


@pytest.mark.parametrize('ds', [
    'ppmi_ds_no_thr', 'ppmi_ds_abs_thr', 'ppmi_ds_pt_thr',
])
def test_ppmi_folds(ds, request):
    # workaround for parameterizing tests w/ fixtures
    ds = request.getfixturevalue(ds)
    folds = ds.folds

    all_train = set()  # everything that we run cross-val on
    all_valids = []
    for i, fold in enumerate(folds['train']):
        train, valid = fold['train'], fold['valid']
        tset, vset = set(train), set(valid)

        assert len(tset) == len(train), f'Fold {i}: non unique idx in train'
        assert len(vset) == len(vset), f'Fold {i}: non unique idx in valid'

        assert tset & vset == set(), f'Fold {i}: intersection between train/valid'
        all_valids.append(vset)

        all_train |= tset
        all_train |= vset

    assert reduce(set.intersection, all_valids) == set(), 'Non empty intersection between valids'
    assert set(folds['test']) & all_train == set(), 'Intersection between test and train'


def test_ppmi_loaders(ppmi_ds_no_thr):
    def get_subj_from_loader(loader):
        ids = []
        for x in loader:
            ids.extend(x.subj_id)
        set_ids = set(ids)
        assert len(set_ids) == len(ids)
        return set_ids

    all_valids = []
    for split in ppmi_ds_no_thr.get_cv_loaders():
        # get loaders
        train, valid = split['train'], split['valid']
        t_ids = get_subj_from_loader(train)
        v_ids = get_subj_from_loader(valid)

        assert t_ids & v_ids == set()
        assert train.dataset != valid.dataset

        all_valids.append(v_ids)

    assert reduce(set.intersection, all_valids) == set(), 'Non empty intersection between valids'


def test_ppmi_test_loader(ppmi_ds_no_thr):
    loader = ppmi_ds_no_thr.get_test_loader(8)
    for b in loader:
        assert b


def test_ppmi_dense_connprofile(ppmi_dense_connprofile):
    assert ppmi_dense_connprofile.data.shape[1] == 112
    assert ppmi_dense_connprofile.data.shape[2] == 112

