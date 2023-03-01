import inspect
import logging
from typing import Type

import neurograph.data.abide as abide
import neurograph.data.cobre as cobre
import neurograph.data.ppmi as ppmi
from neurograph.data.datasets import NeuroDataset, NeuroDenseDataset, NeuroGraphDataset

datasets = [cobre, abide, ppmi]

# TODO: refactor this, this is too complicated for such a simple thing
available_datasets: dict[tuple[str, str], Type[NeuroDataset]] = {
    (obj.name, obj.data_type): obj
    for modules in datasets
    for (class_name, obj) in inspect.getmembers(modules)
    if inspect.isclass(obj)
    if issubclass(obj, NeuroDataset)
    if hasattr(obj, 'name') and hasattr(obj, 'data_type')
    if obj.name and obj.data_type
    if class_name.endswith('Dataset')
}
dense_datasets: dict[str, Type[NeuroDenseDataset]] = {
    ds_name: obj for (ds_name, data_type), obj in available_datasets.items()
    if obj.data_type == 'dense'  # TODO: is this check redundant
    if issubclass(obj, NeuroDenseDataset)
}
graph_datasets: dict[str, Type[NeuroGraphDataset]] = {
    ds_name: obj for (ds_name, data_type), obj in available_datasets.items()
    if obj.data_type == 'graph'  # TODO: is this check redundant
    if issubclass(obj, NeuroGraphDataset)
}
