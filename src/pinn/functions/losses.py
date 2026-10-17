import torch
from torch import Tensor


def mean_squared_residual(r: Tensor) -> Tensor:
    return r.pow(2).mean()


def causal_weights(L_sorted: Tensor, eps: float) -> Tensor:
    """w_i = exp(-eps * sum_{k<i} L_k); Wang, Sankaran & Perdikaris 2024, eq. 3.5."""
    earlier = torch.cumsum(L_sorted, 0) - L_sorted
    return torch.exp(-eps * earlier).detach()


def causal_loss(r: Tensor, t: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    order = torch.argsort(t.reshape(-1))
    L = r[order].pow(2).mean(dim=1)
    w = causal_weights(L, eps)
    return (w * L).mean(), w
