import numpy as np
import torch
from scipy.integrate import solve_ivp


def solve_reference(problem, t: np.ndarray) -> np.ndarray:
    """States at the requested (possibly unsorted) times; DOP853 at 1e-10 / 1e-12."""
    t = np.asarray(t, dtype=float).reshape(-1)
    if problem.reference is not None:
        return problem.reference(t)
    if not len(t):
        return np.empty((0, 3))
    unique, inverse = np.unique(t, return_inverse=True)
    if unique[0] < problem.t_span[0] or unique[-1] > problem.t_span[1]:
        raise ValueError("reference times must be inside the problem span")
    if unique[-1] == problem.t_span[0]:
        return np.tile(np.asarray(problem.initial_state, float), (len(t), 1))

    def f(_, u):
        return problem.rhs(torch.as_tensor(u, dtype=torch.float64).reshape(1, -1))[0].numpy()

    sol = solve_ivp(f, (problem.t_span[0], float(unique[-1])), np.asarray(problem.initial_state, float),
                    method="DOP853", rtol=1e-10, atol=1e-12, t_eval=unique)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T[inverse]
