import pytest
from neurograph import config
from neurograph.data.datasets import CobreDataset


@pytest.fixture
def cobre_dataset():
    return CobreDataset(root=config.DATA_PATH / 'cobre_fmri')


def test_cobre_target(cobre_dataset):
    target, label2idx, idx2label = cobre_dataset.load_target()

    assert len(idx2label) == 2
    assert target.target.nunique() == 2
    assert target.index.isnull().sum() == 0
    assert target.target.isnull().sum() == 0
    assert target.target.sum() == 66
    assert target.shape == (143, 1)
