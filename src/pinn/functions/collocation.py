import numpy as np
from scipy.stats import qmc

from .windows import split_windows


def latin_hypercube_points(t_span: tuple[float, float], n: int, seed: int) -> np.ndarray:
    t0, tf = t_span
    sample = qmc.LatinHypercube(d=1, seed=seed).random(n)
    return (t0 + (tf - t0) * sample).reshape(-1, 1)


def uniform_points(t_span: tuple[float, float], n: int, n_windows: int = 1) -> np.ndarray:
    if n_windows == 1:
        return np.linspace(t_span[0], t_span[1], n).reshape(-1, 1)
    cells, extra = divmod(n - 1, n_windows)
    points = [np.linspace(a, b, cells + (k < extra) + 1)[:-1]
              for k, (a, b) in enumerate(split_windows(t_span, n_windows))]
    return np.concatenate([*points, np.array([t_span[1]])]).reshape(-1, 1)
