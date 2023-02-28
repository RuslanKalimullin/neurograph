import pytest
from functools import reduce
from neurograph.config import get_config
from neurograph.data.cobre import CobreDenseDataset, CobreGraphDataset


@pytest.fixture(scope='session')
def cobre_ds_no_thr():
    return CobreGraphDataset(root=get_config().dataset.data_path, no_cache=True)


@pytest.fixture(scope='session')
def cobre_ds_abs_thr():
    return CobreGraphDataset(root=get_config().dataset.data_path, abs_thr=0.3, no_cache=True)


@pytest.fixture(scope='session')
def cobre_ds_pt_thr():
    return CobreGraphDataset(root=get_config().dataset.data_path, pt_thr=0.5, no_cache=True)


@pytest.fixture(scope='session')
def cobre_dense_ts():
    return CobreDenseDataset(
        root=get_config().dataset.data_path,
        atlas='aal',
        experiment_type='fmri',
        feature_type='timeseries',
    )


@pytest.fixture(scope='session')
def cobre_dense_connprofile():
    return CobreDenseDataset(
        root=get_config().dataset.data_path,
        atlas='aal',
        experiment_type='fmri',
        feature_type='conn_profile',
    )


def test_cobre_no_thr(cobre_ds_no_thr):
    g = cobre_ds_no_thr
    assert g[0].edge_index.shape[1] == cobre_ds_no_thr.num_nodes ** 2


def test_cobre_abs_thr(cobre_ds_abs_thr):
    g = cobre_ds_abs_thr[0]
    assert 0 < g.edge_index.shape[1] < cobre_ds_abs_thr.num_nodes ** 2
    assert g.edge_attr.min().abs() >= 0.5


def test_cobre_pt_thr(cobre_ds_pt_thr):
    g = cobre_ds_pt_thr[0]
    p = cobre_ds_pt_thr.pt_thr * (cobre_ds_pt_thr.num_nodes ** 2)
    assert p // 2 < g.edge_index.shape[1]
    assert g.edge_index.shape[1] <= int(p)


def test_cobre_target(cobre_ds_no_thr):
    target, label2idx, idx2label = cobre_ds_no_thr.load_targets()
    target = target[cobre_ds_no_thr.target_col]

    assert len(idx2label) == 2
    assert target.nunique() == 2
    assert target.isnull().sum() == 0

    assert target.index.isnull().sum() == 0

@pytest.mark.parametrize('ds', ['cobre_ds_no_thr', 'cobre_dense_ts'])
def test_cobre_folds(ds, request):
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


def test_cobre_loaders(cobre_ds_no_thr):
    def get_subj_from_loader(loader):
        ids = []
        for x in loader:
            ids.extend(x.subj_id)
        set_ids = set(ids)
        assert len(set_ids) == len(ids)
        return set_ids

    all_valids = []
    for split in cobre_ds_no_thr.get_cv_loaders():
        # get loaders
        train, valid = split['train'], split['valid']
        t_ids = get_subj_from_loader(train)
        v_ids = get_subj_from_loader(valid)

        assert t_ids & v_ids == set()
        assert train.dataset != valid.dataset

        all_valids.append(v_ids)

    assert reduce(set.intersection, all_valids) == set(), 'Non empty intersection between valids'


def test_cobre_test_loader(cobre_ds_no_thr):
    loader = cobre_ds_no_thr.get_test_loader(8)

    for b in loader:
        print(b)


def test_cobre_dense_ts(cobre_dense_ts):
    assert cobre_dense_ts.data.shape[1] == 116
    assert cobre_dense_ts.data.shape[2] == 150


def test_cobre_dense_connprofile(cobre_dense_connprofile):
    assert cobre_dense_connprofile.data.shape[1] == 116
    assert cobre_dense_connprofile.data.shape[2] == 116

