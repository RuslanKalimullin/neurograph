import os.path as osp
import json
from typing import Optional

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from .utils import load_cm


class CobreDataset(InMemoryDataset):

    available_atlases = {'aal', 'msdl'}
    target_file = 'cobre_morphometry_and_target.csv'
    splits_file = 'cobre_splits.json'

    def __init__(self, root: Optional[str], atlas: str = 'aal'):
        self.atlas = atlas
        self._validate()

        # init fields
        self.splits = None
        self.target = None

        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        # load data list
        data_list = []  # TODO

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def load_data(self):
        target = self.load_target

    def load_target(self):
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
        label2id = {i: x for i, x in enumerate(target.target.unique())}
        idx2label = {x: i for i, x in label2id.items()}

        target.target = target.target.map(target_idx2label)

        return target, label2id, idx2label


    def load_cm(self, thr: Optional[float] = None):
        """ Load connectivity matrices """
        pass

    def load_splits(self):
        with open(self.raw_dir / self.splits_file) as f:
            return json.load(f)

    @property
    def processed_file_names(self):
        return [f'{self.atlas}_data.pt']


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
