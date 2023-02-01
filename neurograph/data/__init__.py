import inspect
from torch_geometric.data import InMemoryDataset
import neurograph.data.datasets as datasets

available_datasets = {
    obj.name for (name, obj) in inspect.getmembers(datasets)
    if inspect.isclass(obj)
    if issubclass(obj, InMemoryDataset)
    if hasattr(obj, 'name')
}
