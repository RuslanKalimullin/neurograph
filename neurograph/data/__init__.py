import inspect
import logging
from typing import Type

import neurograph.data.abide as abide
import neurograph.data.cobre as cobre
import neurograph.data.ppmi as ppmi
from neurograph.data.datasets import (
    NeuroDataset,
    NeuroDenseDataset,
    NeuroGraphDataset,
    MutlimodalDense2Dataset,
)

datasets = [cobre, abide, ppmi]

available_datasets: dict[tuple[str, str], Type[NeuroDataset]] = {
    (obj.name, obj.data_type): obj
    for modules in datasets
    for (class_name, obj) in inspect.getmembers(modules) if inspect.isclass(obj)
    if issubclass(obj, NeuroDataset)
    if hasattr(obj, 'name') and hasattr(obj, 'data_type')
    if obj.name and obj.data_type
    if class_name.endswith('Dataset')
}
dense_datasets: dict[str, Type[NeuroDenseDataset]] = {
    ds_name: obj for (ds_name, data_type), obj in available_datasets.items()
    if obj.data_type == 'dense'
    if issubclass(obj, NeuroDenseDataset)
}
graph_datasets: dict[str, Type[NeuroGraphDataset]] = {
    ds_name: obj for (ds_name, data_type), obj in available_datasets.items()
    if obj.data_type == 'graph'
    if issubclass(obj, NeuroGraphDataset)
}
multimodal_dense_2: dict[str, Type[MutlimodalDense2Dataset]] = {
    ds_name: obj for (ds_name, data_type), obj in available_datasets.items()
    if issubclass(obj, MutlimodalDense2Dataset)
}
