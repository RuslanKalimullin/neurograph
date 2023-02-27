from abc import ABC
from shutil import rmtree
import os.path as osp
import json
from typing import Generator, Optional

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from .datasets import NeuroDataset, NeuroGraphDataset, NeuroDenseDataset
from .utils import load_cms, prepare_graph


class CobreTrait:
    """ Common fields and methods for all Cobre datasets """
    name = 'cobre'
    available_atlases = {'aal', 'msdl'}
    available_experiments = {'fmri', 'dti'}
    splits_file = 'cobre_splits.json'
    target_file = 'meta_data.tsv'
    subj_id_col = 'Subjectid'
    target_col = 'Dx'

    global_dir: str  # just for type checks

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


# NB: trait must go first
class CobreGraphDataset(CobreTrait, NeuroGraphDataset):
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
        num_nodes = self.slices['x'].diff().unique()
        assert len(num_nodes) == 1, 'You have different number of nodes in graphs!'
        self.num_nodes = num_nodes.item()

    def load_cms(self,path: str | Path,) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[int, str]]:

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


class CobreDenseDataset(CobreTrait, NeuroDenseDataset):
    pass
