import os.path as osp
import json
from typing import Optional

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from neurograph.config import DATA_PATH
from .utils import load_cms, prepare_one_graph



class CobreDataset(InMemoryDataset):

    available_atlases = {'aal', 'msdl'}
    target_file = 'cobre_morphometry_and_target.csv'
    splits_file = 'cobre_splits.json'

    def __init__(
        self,
        root: Optional[str] = str(DATA_PATH / 'cobre_fmri'),
        atlas: str = 'aal',
        thr = None,
        k = None,
    ):
        # TODO: thr, k, add thr to processed file names
        # TODO: throw warning if both thr and k are not None

        self.atlas = atlas
        self.thr = thr
        self.k = k
        self._validate()

        # init fields
        self.splits = None
        self.target = None

        # here `process` is called
        super().__init__(root)

        self.data, self.slices = torch.load(self.processed_paths[0])

        with open(self.processed_paths[1]) as f_ids:
            self.subj_ids = [l.rstrip() for l in f_ids.readlines()]

        # load fixed stratified partition to 5 folds and test
        with open(self.processed_paths[2]) as f_folds:
            self.folds = json.load(f_folds)

    @property
    def processed_file_names(self):
        return [f'{self.atlas}_data.pt', 'subj_ids.txt', 'folds.json']

    @property
    def cm_path(self):
        return osp.join(self.raw_dir, self.atlas)

    def process(self):
        # load data list
        data_list, subj_ids = self.load_datalist()

        # collate DataList and save to disk
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        # save subj_ids as a txt file
        with open(self.processed_paths[1], 'w') as f:
            f.write('\n'.join(subj_ids))

        # load fold splits
        id_folds, num_folds = self.load_folds()
        id2idx = {s: i for i, s in enumerate(subj_ids)}

        # map each subj_id to idx in `data_list`
        folds = {'train': []}
        for i in range(num_folds):
            train_ids, valid_ids = id_folds[i]['train'], id_folds[i]['valid']
            one_fold = {
                'train': [id2idx[subj_id] for subj_id in train_ids],
                'valid': [id2idx[subj_id] for subj_id in valid_ids],
            }
            folds['train'].append(one_fold)
        folds['test'] = [id2idx[subj_id] for subj_id in id_folds['test']]

        with open(self.processed_paths[2], 'w') as f_folds:
            json.dump(folds, f_folds)

    def load_datalist(self) -> tuple[list[Data], list[str]]:
        targets, label2idx, idx2label = self.load_targets()

        # subj_id -> CM, etc.
        cms, ts, roi_map = load_cms(osp.join(self.raw_dir, self.cm_path))

        # prepare data list from cms and targets
        datalist = []
        subj_ids = []
        for subj_id, cm in cms.items():
            try:
                # try to process a graph
                datalist.append(prepare_one_graph(cm, subj_id, targets))
                subj_ids.append(subj_id)
            except KeyError:
                # ignore if subj_id is not in targets
                pass

        return datalist, subj_ids

    def load_targets(self) -> tuple[pd.DataFrame, dict[str, int], dict[int, str]]:
        """ Process csv file with targets """

        target = pd.read_csv(osp.join(self.raw_dir, self.target_file))
        target = target[['ID', 'target']].copy()

        # check that there are no different labels assigned to the same ID
        max_labels_per_id = target.groupby('ID').target.nunique().max()
        assert max_labels_per_id == 1, 'Diffrent targets assigned to the same ID!'

        target.drop_duplicates(inplace=True)
        target.set_index('ID', inplace=True)
        # drop schizoaffective
        target = target[target.target != 'Schizoaffective'].copy()

        # label encoding
        label2idx: dict[str, int] = {x: i for i, x in enumerate(target.target.unique())}
        idx2label: dict[int, str] = {i: x for x, i in label2idx.items()}

        target.target = target.target.map(label2idx)

        return target, label2idx, idx2label

    def load_folds(self):
        with open(osp.join(self.raw_dir, self.splits_file)) as f:
            _folds = json.load(f)

        folds = {}
        num_folds = -1
        for k, v in _folds.items():
            if k.isnumeric():
                new_k = int(k)
                num_folds = max(num_folds, new_k)
                folds[new_k] = v
            else:
                folds[k] = v
        return folds, num_folds + 1

    def get_cv_loaders(self, batch_size=8, valid_batch_size=None):
        valid_batch_size = valid_batch_size if valid_batch_size else batch_size
        for fold in self.folds['train']:
            train_idx, valid_idx = fold['train'], fold['valid']
            yield {
                'train': DataLoader(self[train_idx], batch_size=batch_size, shuffle=True),
                'valid': DataLoader(self[valid_idx], batch_size=valid_batch_size, shuffle=False),
            }

    def _validate(self):
        if self.atlas not in self.available_atlases:
            raise ValueError('Unknown atlas')


class ListDataset(InMemoryDataset):
    """ Basic dataset for ad-hoc experiments """
    def __init__(self, root, data_list: list[Data]):
        # first store `data_list` as attr
        self.data_list = data_list
        super().__init__(root=root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
        data, slices = self.collate(self.data_list)
        torch.save((data, slices), self.processed_paths[0])
