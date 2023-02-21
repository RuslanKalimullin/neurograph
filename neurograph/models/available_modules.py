import inspect
from typing import Type
import torch.nn as nn
import torch.optim as optim

# put different pytorch models into dicts for easier instantiation
available_modules: dict[str, Type[nn.Module]] = {
    name: obj for (name, obj) in inspect.getmembers(nn)
    if inspect.isclass(obj)
    if issubclass(obj, nn.Module)
}

available_activations: dict[str, Type[nn.Module]] = {
    name: obj for (name, obj) in available_modules.items()
    if obj.__module__.endswith('activation')
}

available_losses: dict[str, Type[nn.Module]] = {
    name: obj for name, obj in available_modules.items()
    if name.endswith('Loss')
}

available_optimizers: dict[str, Type[optim.Optimizer]] = {
    name: obj for (name, obj) in inspect.getmembers(optim)
    if inspect.isclass(obj)
    if issubclass(obj, optim.Optimizer)
}
