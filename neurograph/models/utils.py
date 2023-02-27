import torch

def concat_pool(x: torch.Tensor, num_nodes: int) -> torch.Tensor:
    # NB: x must be a batch of xs
    return x.reshape(x.size(0) // num_nodes, -1)
