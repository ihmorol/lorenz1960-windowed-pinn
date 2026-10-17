import torch
from torch import Tensor


def time_derivative(u: Tensor, t: Tensor) -> Tensor:
    cols = [torch.autograd.grad(u[:, j].sum(), t, create_graph=True)[0][:, 0] for j in range(u.shape[1])]
    return torch.stack(cols, dim=1)
