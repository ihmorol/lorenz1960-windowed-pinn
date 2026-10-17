"""The equation being solved, as data: right-hand side, start state, window, reference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from torch import Tensor

from .functions.physics import lorenz1960_rhs


@dataclass(frozen=True)
class Problem:
    name: str
    dim: int
    rhs: Callable[[Tensor], Tensor]
    initial_state: tuple[float, ...]
    t_span: tuple[float, float]
    coefficients: np.ndarray | None = None       # used by the invariant figures
    end_state: tuple[float, ...] | None = None   # future two-point BVP
    reference: Callable[[np.ndarray], np.ndarray] | None = None


def lorenz1960(coefficients: np.ndarray, initial_state, t_span, end_state=None,
               reference=None) -> Problem:
    return Problem("lorenz1960", 3, lambda u: lorenz1960_rhs(u, coefficients),
                   tuple(initial_state), tuple(t_span), coefficients, end_state, reference)


PROBLEMS: dict[str, Callable[..., Problem]] = {"lorenz1960": lorenz1960}
