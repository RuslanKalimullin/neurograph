import inspect
from typing import Type

import neurograph.data.datasets as datasets

available_datasets: dict[str, Type[datasets.NeuroGraphDataset]] = {
    obj.name: obj for (name, obj) in inspect.getmembers(datasets)
    if inspect.isclass(obj)
    if issubclass(obj, datasets.NeuroGraphDataset)
    if hasattr(obj, 'name')
    if obj.name
}
