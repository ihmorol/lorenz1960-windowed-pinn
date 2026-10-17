from torch import Tensor


def hard_initial_condition(t: Tensor, n: Tensor, u0: Tensor, t0: float, tf: float, form: str) -> Tensor:
    """u = u0 + g(t) N(t). 'span': g = (t - t0)/(tf - t0). 'unit': g = t - t0 (Lagaris 1998)."""
    g = (t - t0) / (tf - t0) if form == "span" else (t - t0)
    return u0 + g * n
