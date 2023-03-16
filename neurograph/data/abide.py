""" Dataset classes for ABIDE dataset """

import os.path as osp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .datasets import NeuroGraphDataset, NeuroDenseDataset


class ABIDETrait:
    """ Common fields and methods for all ABIDE datasets """
    name = 'abide'
    available_atlases = {'aal', 'msdl'}
    available_experiments = {'fmri', 'dti'}
    splits_file = 'abide_splits.json'
    target_file = 'Phenotypic_V1_0b_preprocessed1.csv'
    subj_id_col = 'SUB_ID'
    target_col = 'DX_GROUP'
    con_matrix_suffix='_cc200_correlation.mat'
    embed_sufix='*.1D'

    global_dir: str  # just for type checks

    def load_targets(self) -> tuple[pd.DataFrame, dict[str, int], dict[int, str]]:
        """ Load and process *cobre* targets """

        target = pd.read_csv(osp.join(self.global_dir, self.target_file))
        target = target[[self.subj_id_col, self.target_col]]
        target[self.subj_id_col]=target[self.subj_id_col].astype(str)
        # check that there are no different labels assigned to the same ID
        max_labels_per_id = target.groupby(self.subj_id_col)[self.target_col].nunique().max()
        assert max_labels_per_id == 1, 'Diffrent targets assigned to the same id!'

        # remove duplicates by subj_id
        target.drop_duplicates(subset=[self.subj_id_col], inplace=True)
        # set subj_id as index
        target.set_index(self.subj_id_col, inplace=True)

        # label encoding
        label2idx: dict[str, int] = {x: i for i, x in enumerate(target[self.target_col].unique())}
        idx2label: dict[int, str] = {i: x for x, i in label2idx.items()}
        target[self.target_col] = target[self.target_col].map(label2idx)

        return target, label2idx, idx2label

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

        for p in path.iterdir():
            if p.is_dir():
                name = p.name

                mat = loadmat(p / f"{name}{self.con_matrix_suffix}")
                values = mat['connectivity'].astype(np.float32)

                embed_name =list(p.glob(self.embed_sufix))[0]
                values_embed = pd.read_csv(embed_name,delimiter="\t").astype(np.float32)
                ts[name] = values_embed
                data[name] = values

        return data, ts, roi_map


# NB: trait must go first
# pylint: disable=too-many-ancestors
class ABIDEGraphDataset(ABIDETrait, NeuroGraphDataset):
    """ Graph dataset for ABIDE dataset """


class ABIDEDenseDataset(ABIDETrait, NeuroDenseDataset):
    """ Dense dataset for ABIDE dataset """
