import logging
from typing import Any, Mapping
import hydra
import dataclasses
from omegaconf import OmegaConf
from neurograph.config import Config
from neurograph.data import available_datasets
from neurograph.data.datasets import NeuroDataset


def dataclass_from_dict(cls: type, src: Mapping[str, Any]) -> Any:
    # map each field to its type (class)
    field_types_lookup = {
        field.name: field.type
        for field in dataclasses.fields(cls)
    }

    constructor_inputs = {}
    for field_name, value in src.items():
        try:
            constructor_inputs[field_name] = dataclass_from_dict(field_types_lookup[field_name], value)
        except TypeError as e:
            # type error from fields() call in recursive call
            # indicates that field is not a dataclass, this is how we are
            # breaking the recursion. If not a dataclass - no need for loading
            constructor_inputs[field_name] = value
        #except KeyError:
        #    # similar, field not defined on dataclass, pass as plain field value
        #    constructor_inputs[field_name] = value

    return cls(**constructor_inputs)


def load_dataset(cfg: Config) -> NeuroDataset:
    ds_cfg = cfg.dataset
    DsKlass = available_datasets[ds_cfg.name]
    return DsKlass(
        root=ds_cfg.data_path,
        atlas=ds_cfg.atlas,
        experiment_type=ds_cfg.experiment_type,
    )


@hydra.main(version_base=None, config_path='../config', config_name="config")
def train(cfg: Config):
    #cfg = dataclass_from_dict(Config, OmegaConf.to_container(cfg))

    # load dataset
    ds = load_dataset(cfg)
    print(type(cfg), cfg)


train()
