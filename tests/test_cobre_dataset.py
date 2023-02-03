import pytest
from functools import reduce
from neurograph.config import get_config
from neurograph.data.datasets import CobreDataset


@pytest.fixture(scope='session')
def cobre_dataset():
    return CobreDataset(root=get_config().dataset.data_path)


def test_cobre_target(cobre_dataset):
    target, label2idx, idx2label = cobre_dataset.load_targets()
    target = target[cobre_dataset.target_col]

    assert len(idx2label) == 2
    assert target.nunique() == 2
    assert target.isnull().sum() == 0

    assert target.index.isnull().sum() == 0


def test_cobre_folds(cobre_dataset):
    folds = cobre_dataset.folds

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


def test_cobre_loaders(cobre_dataset):
    def get_subj_from_loader(loader):
        ids = []
        for x in loader:
            ids.extend(x.subj_id)
        set_ids = set(ids)
        assert len(set_ids) == len(ids)
        return set_ids

    all_valids = []
    for split in cobre_dataset.get_cv_loaders():
        # get loaders
        train, valid = split['train'], split['valid']
        t_ids = get_subj_from_loader(train)
        v_ids = get_subj_from_loader(valid)

        assert t_ids & v_ids == set()
        assert train.dataset != valid.dataset

        all_valids.append(v_ids)

    assert reduce(set.intersection, all_valids) == set(), 'Non empty intersection between valids'

def test_cobre_test_loader(cobre_dataset):
    loader = cobre_dataset.get_test_loader(8)

    for b in loader:
        print(b)
