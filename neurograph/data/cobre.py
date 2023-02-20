from shutil import rmtree
import os.path as osp
import json
from typing import Generator, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as thDataloader
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader as pygDataLoader

from .datasets import NeuroDataset, NeuroGraphDataset, NeuroDenseDataset
from .utils import load_cms, prepare_graph


class CobreTrait(NeuroDataset):
    """ Common fields and methods for all Cobre datasets """
    name = 'cobre'
    available_atlases = {'aal', 'msdl'}
    available_experiments = {'fmri', 'dti'}
    splits_file = 'cobre_splits.json'
    target_file = 'meta_data.tsv'
    subj_id_col = 'Subjectid'
    target_col = 'Dx'

    def load_targets(self) -> tuple[pd.DataFrame, dict[str, int], dict[int, str]]:
        """ Load and process *cobre* targets """

        target = pd.read_csv(osp.join(self.global_dir, self.target_file), sep='\t')
        target = target[[self.subj_id_col, self.target_col]]

        # check that there are no different labels assigned to the same ID
        max_labels_per_id = target.groupby(self.subj_id_col)[self.target_col].nunique().max()
        assert max_labels_per_id == 1, 'Diffrent targets assigned to the same id!'

        # remove duplicates by subj_id
        target.drop_duplicates(subset=[self.subj_id_col], inplace=True)
        # set subj_id as index
        target.set_index(self.subj_id_col, inplace=True)

        # leave only Schizo and Control
        target = target[target[self.target_col].isin(('No_Known_Disorder', 'Schizophrenia_Strict'))].copy()

        # label encoding
        label2idx: dict[str, int] = {x: i for i, x in enumerate(target[self.target_col].unique())}
        idx2label: dict[int, str] = {i: x for x, i in label2idx.items()}
        target[self.target_col] = target[self.target_col].map(label2idx)

        return target, label2idx, idx2label


class CobreDataset(NeuroGraphDataset, CobreTrait):
    def __init__(
        self,
        root: str,
        atlas: str = 'aal',
        experiment_type: str = 'fmri',
        init_node_features: str = 'conn_profile',
        abs_thr: Optional[float] = None,
        pt_thr: Optional[float] = None,
        no_cache = False,
    ):
        """
        Args:
            root (str, optional): root dir where dataset should be saved
            atlas (str): atlas name
            thr (float, optional): threshold used for pruning edges #TODO
            k (int, optional): Number of neighbors used to compute a threshold for pruning
                When k is used, thr must be None!
            no_cache (bool): if True, delete processed files and run processing from scratch
        """

        # TODO: thr add thr to processed file names
        self.atlas = atlas
        self.experiment_type = experiment_type
        self.init_node_features = init_node_features
        self.abs_thr = abs_thr
        self.pt_thr = pt_thr

        self._validate()

        # root: experiment specific files (CMs and time series matrices)
        self.root = osp.join(root, self.name, experiment_type)
        # global_dir: dir with meta info and cv_splits
        self.global_dir = osp.join(root, self.name)

        if no_cache:
            rmtree(self.processed_dir, ignore_errors=True)

        # here `self.process` is called
        super().__init__(self.root)

        # load preprocessed graphs
        self.data, self.slices = torch.load(self.processed_paths[0])

        # load dataframes w/ subj_ids and targets
        self.target_df = pd.read_csv(self.processed_paths[3])

        with open(self.processed_paths[1]) as f_ids:
            self.subj_ids = [l.rstrip() for l in f_ids.readlines()]

        # load cv splits
        with open(self.processed_paths[2]) as f_folds:
            self.folds = json.load(f_folds)

        # get some graph attrs (used for initializing models)
        self.n_features = self.data.x.shape[1]
        num_nodes = self.slices['x'].diff().unique()
        assert len(num_nodes) == 1, 'You have different number of nodes in graphs!'
        self.num_nodes = num_nodes.item()

    @property
    def processed_file_names(self):
        thr = ''
        if self.abs_thr:
            thr = f'abs={self.abs_thr}'
        if self.pt_thr:
            thr = f'pt={self.pt_thr}'

        prefix = '_'.join(s for s in [self.atlas, self.experiment_type, thr] if s)
        return [
            f'{prefix}_data.pt',
            f'{prefix}_subj_ids.txt',
            f'{prefix}_folds.json',
            f'{prefix}_targets.csv',
        ]

    @property
    def cm_path(self):
        return osp.join(self.raw_dir, self.atlas)

    def process(self):
        # load data list
        data_list, subj_ids, y = self.load_datalist()

        self.num_nodes = data_list[0].num_nodes

        y.to_csv(self.processed_paths[3])

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

    def load_datalist(self) -> tuple[list[Data], list[str], pd.DataFrame]:
        targets, label2idx, idx2label = self.load_targets()

        # subj_id -> CM, time series matrices, ROI names
        cms, ts, roi_map = load_cms(self.cm_path)

        # prepare data list from cms and targets
        datalist = []
        subj_ids = []
        for subj_id, cm in cms.items():
            try:
                # try to process a graph
                datalist.append(prepare_graph(cm, subj_id, targets, self.abs_thr, self.pt_thr))
                subj_ids.append(subj_id)
            except KeyError:
                # ignore if subj_id is not in targets
                pass
        y = targets.loc[subj_ids].copy()
        return datalist, subj_ids, y

    def load_targets(self) -> tuple[pd.DataFrame, dict[str, int], dict[int, str]]:
        """ Process tsv file with targets """

        target = pd.read_csv(osp.join(self.global_dir, self.target_file), sep='\t')
        target = target[[self.subj_id_col, self.target_col]]

        # check that there are no different labels assigned to the same ID
        max_labels_per_id = target.groupby(self.subj_id_col)[self.target_col].nunique().max()
        assert max_labels_per_id == 1, 'Diffrent targets assigned to the same id!'

        # remove duplicates by subj_id
        target.drop_duplicates(subset=[self.subj_id_col], inplace=True)
        # set subj_id as index
        target.set_index(self.subj_id_col, inplace=True)

        # leave only Schizo and Control
        target = target[target[self.target_col].isin(('No_Known_Disorder', 'Schizophrenia_Strict'))].copy()

        # label encoding
        label2idx: dict[str, int] = {x: i for i, x in enumerate(target[self.target_col].unique())}
        idx2label: dict[int, str] = {i: x for x, i in label2idx.items()}
        target[self.target_col] = target[self.target_col].map(label2idx)

        return target, label2idx, idx2label

    def get_cv_loaders(self, batch_size=8, valid_batch_size=None):
        valid_batch_size = valid_batch_size if valid_batch_size else batch_size
        for fold in self.folds['train']:
            train_idx, valid_idx = fold['train'], fold['valid']
            yield {
                'train': pygDataLoader(self[train_idx], batch_size=batch_size, shuffle=True),
                'valid': pygDataLoader(self[valid_idx], batch_size=valid_batch_size, shuffle=False),
            }

    def get_test_loader(self, batch_size: int) -> pygDataLoader:
        # TODO: test it
        test_idx = self.folds['test']
        return pygDataLoader(self[test_idx], batch_size=batch_size, shuffle=False)

    def _validate(self):
        if self.atlas not in self.available_atlases:
            raise ValueError('Unknown atlas')
        if self.experiment_type not in self.available_experiments:
            raise ValueError(f'Unknown experiment type: {self.experiment_type}')
        if self.pt_thr is not None and self.abs_thr is not None:
            raise ValueError('Both proportional threshold `pt` and absolute threshold `thr` are not None! Choose one!')


class CobreDenseDataset(NeuroDenseDataset, CobreTrait):
    # TODO: create Cobre mixup
    name = 'cobre'
    data_type = 'dense'
    available_atlases = {'aal', 'msdl'}
    available_experiments = {'fmri', 'dti'}
    splits_file = 'cobre_splits.json'
    target_file = 'meta_data.tsv'
    subj_id_col = 'Subjectid'
    target_col = 'Dx'

    def __init__(
        self,
        root: str,
        atlas: str = 'aal',
        experiment_type: str = 'fmri',
        feature_type: str = 'timeseries',  # or 'conn_profile'
    ):
        self.atlas = atlas
        self.experiment_type = experiment_type
        self.feature_type = feature_type

        # root: experiment specific files (CMs and time series matrices)
        self.root = osp.join(root, self.name, experiment_type)
        # global_dir: dir with meta info and cv_splits
        self.global_dir = osp.join(root, self.name)
        # path to CM and time series
        self.cm_path = osp.join(self.root, 'raw', self.atlas)

        self.data: torch.Tensor = torch.tensor([0])
        folds = self.load_folds()

    def load_data(self):
        cms, ts, roi_map = load_cms(self.cm_path)

        # load_targets

        if self.feature_type == 'timeseries':
            return ts
        elif self.feature_type == 'conn_profile':
            return cms
        else:
            raise ValueError(f'Unknown feature_type: {self.feature_type}')

    def __len__(self):
        return len(data)

    def __getitem__(self, idx: int):
        return self.data[idx]


