from abc import ABC, abstractmethod
from shutil import rmtree
import os.path as osp
import json
from typing import Generator, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as thDataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader as pygDataLoader

from .utils import load_cms, prepare_graph


class NeuroDataset(ABC):
    """ Common fields and methods for all our datasets (graph or dense) """
    name: str
    atlas: str
    experiment_type: str
    available_atlases: set[str]
    available_experiments: set[str]

    root: str
    global_dir: str

    # filenames
    splits_file: str
    target_file: str

    data_type: str

    def load_folds(self) -> tuple[dict, int]:
        """ Loads json w/ splits, returns a dict w/ splits and number of folds """
        with open(osp.join(self.global_dir, self.splits_file)) as f:
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

    @abstractmethod
    def get_cv_loaders(
        self,
        batch_size=8,
        valid_batch_size=None,
    ):
    # -> Generator[dict[str, pygDataLoader], None, None]:
        raise NotImplementedError

    @abstractmethod
    def get_test_loader(self, batch_size: int):
        raise NotImplementedError


class NeuroGraphDataset(InMemoryDataset, NeuroDataset):
    """ Base class for every InMemoryDataset used in this project """

    abs_thr: Optional[float]
    pt_thr: Optional[float]
    n_features: int
    num_nodes: int

    init_node_features: str = 'conn_profile'
    data_type: str = 'graph'

    def get_cv_loaders(
        self,
        batch_size=8,
        valid_batch_size=None,
    ) -> Generator[dict[str, pygDataLoader], None, None]:
        raise NotImplementedError

    def get_test_loader(self, batch_size: int) -> pygDataLoader:
        raise NotImplementedError

    @property
    def cm_path(self):
        return osp.join(self.raw_dir, self.atlas)

    def __repr__(self):
        return f'{self.__class__.__name__}: atlas={self.atlas}, experiment_type={self.experiment_type}, pt_thr={self.pt_thr}, abs_thr={self.abs_thr}, size={len(self)}'


class NeuroDenseDataset(thDataset, NeuroDataset):
    data_type: str = 'dense'

    @staticmethod
    def prepare_data(
        matrix_dict: dict[str, np.ndarray],
        targets: pd.DataFrame,
    ) -> tuple[torch.Tensor, list[str], torch.Tensor]:
        # matrix_dict: mapping subj_id -> CM of time series

        datalist = []
        subj_ids = []
        for subj_id, m in matrix_dict.items():
            try:
                datalist.append(torch.tensor(m))
                subj_ids.append(subj_id)
            except KeyError:
                # ignore if subj_id is not in targets
                pass

        # NB: use LongTensor here
        y = torch.LongTensor(targets.loc[subj_ids].copy().values)
        data = torch.cat(datalist, dim=0)
        return data, subj_ids, y


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
