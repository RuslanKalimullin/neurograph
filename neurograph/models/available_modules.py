import inspect
from typing import Type
import torch.nn as nn

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

