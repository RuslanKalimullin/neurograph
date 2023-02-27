import inspect
from typing import Any, Type
import torch.nn as nn
from torch_geometric import nn as tg_nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# put different pytorch models into dicts for easier instantiation
available_pg_modules: dict[str, Type[nn.Module]] = {
    name: obj for (name, obj) in inspect.getmembers(tg_nn)
    if inspect.isclass(obj)
    if issubclass(obj, nn.Module)
}

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

# there's no base class for all schedulers
available_schedulers: dict[str, Any] = {
    name: obj for (name, obj) in inspect.getmembers(lr_scheduler)
    if inspect.isclass(obj)
}
