from typing import Callable

import torch
from torch import Tensor


def lorenz1960_rhs(u: Tensor, coeffs) -> Tensor:
    c = torch.as_tensor(coeffs, dtype=u.dtype, device=u.device).reshape(3)
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    return torch.stack([c[0] * y * z, c[1] * x * z, c[2] * x * y], dim=1)


def residual(u: Tensor, dudt: Tensor, rhs: Callable[[Tensor], Tensor]) -> Tensor:
    return dudt - rhs(u)
