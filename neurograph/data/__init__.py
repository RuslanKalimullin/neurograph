import inspect
from typing import Type

import neurograph.data.cobre as cobre
from neurograph.data.datasets import NeuroDataset, NeuroDenseDataset, NeuroGraphDataset

# later we add other datasets
datasets = [cobre]

available_datasets: dict[str, Type[NeuroDataset]] = {
    '_'.join([obj.name, obj.data_type]): obj
    for modules in datasets
    for (class_name, obj) in inspect.getmembers(modules)
    if inspect.isclass(obj)
    if issubclass(obj, NeuroDataset)
    if hasattr(obj, 'name') and hasattr(obj, 'data_type')
    if obj.name and obj.data_type
    if class_name.endswith('Dataset')
}


def dataset_factory(name: str, data_type: str):
    return available_datasets['_'.join([name, data_type])]
