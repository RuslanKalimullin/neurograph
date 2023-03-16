""" PPMI dataset classes """

import os.path as osp
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import NeuroGraphDataset, NeuroDenseDataset


class PPMITrait:
    """ Common fields and methods for all PPMI datasets """
    name = 'ppmi'
    available_atlases = {'aal', 'desikankilliany'}
    available_experiments = {'dti'}
    # weird hack since we have different split for different atlases
    splits_file_fstr = 'ppmi_splits_{atlas}.json'

    target_file = 'ppmi_baseline_simens_clean.csv'
    subj_id_col = 'Subject'
    image_id_col = 'Image Data ID'
    target_col = 'Group'

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
        time_series: dict[str, np.ndarray] = {}
        # ROI names, extacted from CMs
        roi_map: dict[int, str] = {}

        for p in path.glob('*.csv'):
            name = p.stem
            #subj_id, image_id = name.split('_', 1)
            #subj_id = int(subj_id)
            x = pd.read_csv(p).drop('Unnamed: 0', axis=1)

            data[name] = x.values
            if not roi_map:
                roi_map = {i: c.rstrip() for i, c in enumerate(x.columns)}

        return data, time_series, roi_map

    def load_targets(self) -> tuple[pd.DataFrame, dict[str, int], dict[int, str]]:
        """ Load and process targets """

        # load csv file w/ target, leave only subject ids and target columns
        raw_target = pd.read_csv(osp.join(self.global_dir, self.target_file))

        # since out `subj_id` is really a concat of subject_id and image_id
        # we must concat two columns
        subj_id = raw_target[self.subj_id_col].astype(str) + '_' + raw_target[self.image_id_col]
        # construct target w/ new subj_id
        target: pd.DataFrame = pd.DataFrame(
            {
                self.subj_id_col: subj_id,
                self.target_col: raw_target[self.target_col]
            }
        )

        # check that there are no different labels assigned to the same subject id
        max_labels_per_id = target.groupby(self.subj_id_col)[self.target_col].nunique().max()
        assert max_labels_per_id == 1, 'Diffrent targets assigned to the same subject id!'

        # remove duplicates by subj_id
        target.drop_duplicates(subset=[self.subj_id_col], inplace=True)
        # set subj_id as index
        target.set_index(self.subj_id_col, inplace=True)

        # leave only PD and Control
        labels = ('Control', 'PD')
        target = target[target[self.target_col].isin(labels)].copy()

        # label encoding
        label2idx: dict[str, int] = {x: i for i, x in enumerate(labels)}
        idx2label: dict[int, str] = {i: x for x, i in label2idx.items()}
        target[self.target_col] = target[self.target_col].map(label2idx)

        return target, label2idx, idx2label

    @property
    def splits_file(self):
        """ For PPMI we have (for now) different splits files for different atlases
            because of some pecularities of preprocessing.
            So we added the corresponding property to construct splits file name dynamically
        """
        # pylint: disable=no-member
        # self.atlas is defined in the corresponding class
        return self.splits_file_fstr.format(atlas=self.atlas)


# pylint: disable=too-many-ancestors
class PPMIGraphDataset(PPMITrait, NeuroGraphDataset):
    """ Graph Dataset for PPMI dataset """


# pylint: disable=too-many-ancestors
class PPMIDenseDataset(PPMITrait, NeuroDenseDataset):
    """ Dense Dataset for PPMI dataset """
