import os.path as osp
from typing import Generator, Optional

import numpy as np
import pandas as pd
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

    def load_cms(
        self,
        path: str | Path,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[int, str]]:

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
    pass


class CobreMultimodalGraphDataset(CobreTrait, NeuroGraphDataset):
    pass


class CobreDenseDataset(CobreTrait, NeuroDenseDataset):
    pass
