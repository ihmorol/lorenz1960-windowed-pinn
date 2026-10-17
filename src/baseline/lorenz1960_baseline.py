from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Lorenz1960Config:
    k: float = 2.0
    l: float = 1.0
    initial_state: tuple[float, float, float] = (0.5, 0.75, 1.0)
    t_span: tuple[float, float] = (0.0, 1.0)
    rk4_step: float = 1e-3
    n_eval: int = 1001
    scipy_method: str = "DOP853"
    scipy_rtol: float = 1e-10
    scipy_atol: float = 1e-12


DEFAULT_CONFIG = Lorenz1960Config()
STATE_LABELS = ("x", "y", "z")


def lorenz1960_coefficients(k: float = DEFAULT_CONFIG.k, l: float = DEFAULT_CONFIG.l) -> np.ndarray:
    coeff_x = k * l * (1.0 / (k**2 + l**2) - 1.0 / (k**2))
    coeff_y = k * l * (1.0 / (l**2) - 1.0 / (k**2 + l**2))
    coeff_z = 0.5 * k * l * (1.0 / (k**2) - 1.0 / (l**2))
    return np.array([coeff_x, coeff_y, coeff_z], dtype=float)


def lorenz1960_rhs(
    t: float,
    state: np.ndarray,
    k: float = DEFAULT_CONFIG.k,
    l: float = DEFAULT_CONFIG.l,
) -> np.ndarray:
    del t
    x, y, z = np.asarray(state, dtype=float)
    coeff_x, coeff_y, coeff_z = lorenz1960_coefficients(k=k, l=l)
    dx = coeff_x * y * z
    dy = coeff_y * x * z
    dz = coeff_z * x * y
    return np.array([dx, dy, dz], dtype=float)


def make_uniform_grid(
    t_span: tuple[float, float] = DEFAULT_CONFIG.t_span,
    *,
    step: float | None = None,
    n_eval: int | None = None,
) -> np.ndarray:
    if step is not None and n_eval is not None:
        raise ValueError("Provide either step or n_eval, not both.")
    if step is None and n_eval is None:
        n_eval = DEFAULT_CONFIG.n_eval

    t0, t1 = t_span
    if step is not None:
        n_steps_float = (t1 - t0) / step
        n_steps = int(round(n_steps_float))
        if not np.isclose(n_steps_float, n_steps, atol=1e-12):
            raise ValueError("Step size does not divide the interval exactly.")
        return np.linspace(t0, t1, n_steps + 1)

    return np.linspace(t0, t1, int(n_eval))


def rk4_step(f, t: float, y: np.ndarray, h: float) -> np.ndarray:
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_solve(
    f,
    t_span: tuple[float, float],
    y0: np.ndarray | tuple[float, ...],
    h: float,
) -> tuple[np.ndarray, np.ndarray]:
    ts = make_uniform_grid(t_span, step=h)
    ys = np.zeros((ts.size, len(y0)), dtype=float)
    ys[0] = np.asarray(y0, dtype=float)
    y = ys[0].copy()

    for idx in range(ts.size - 1):
        y = rk4_step(f, ts[idx], y, h)
        ys[idx + 1] = y

    return ts, ys


def solve_lorenz1960_scipy(
    *,
    config: Lorenz1960Config = DEFAULT_CONFIG,
    t_eval: np.ndarray | None = None,
    method: str | None = None,
    rtol: float | None = None,
    atol: float | None = None,
) -> tuple[np.ndarray, np.ndarray, object]:
    if t_eval is None:
        t_eval = make_uniform_grid(config.t_span, n_eval=config.n_eval)

    solution = solve_ivp(
        lambda t, y: lorenz1960_rhs(t, y, k=config.k, l=config.l),
        config.t_span,
        np.asarray(config.initial_state, dtype=float),
        method=method or config.scipy_method,
        t_eval=t_eval,
        rtol=rtol if rtol is not None else config.scipy_rtol,
        atol=atol if atol is not None else config.scipy_atol,
    )

    if not solution.success:
        raise RuntimeError(f"SciPy solver failed: {solution.message}")

    return solution.t, solution.y.T, solution


def compute_error_metrics(reference: np.ndarray, candidate: np.ndarray) -> pd.DataFrame:
    if reference.shape != candidate.shape:
        raise ValueError("Reference and candidate arrays must have the same shape.")

    error = candidate - reference
    abs_error = np.abs(error)

    rows: list[dict[str, float | str]] = []
    for idx, label in enumerate(STATE_LABELS):
        rows.append(
            {
                "state": label,
                "mae": float(abs_error[:, idx].mean()),
                "rmse": float(np.sqrt(np.mean(error[:, idx] ** 2))),
                "max_abs_error": float(abs_error[:, idx].max()),
            }
        )

    euclidean_error = np.linalg.norm(error, axis=1)
    rows.append(
        {
            "state": "combined_l2",
            "mae": float(euclidean_error.mean()),
            "rmse": float(np.sqrt(np.mean(euclidean_error**2))),
            "max_abs_error": float(euclidean_error.max()),
        }
    )
    return pd.DataFrame(rows)


def subsample_uniform_solution(ts: np.ndarray, ys: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    sampled = np.zeros((target_ts.size, ys.shape[1]), dtype=float)
    for idx in range(ys.shape[1]):
        sampled[:, idx] = np.interp(target_ts, ts, ys[:, idx])
    return sampled


def final_state_table(*, ts: np.ndarray, ys: np.ndarray, solver_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "solver": [solver_name] * len(STATE_LABELS),
            "state": list(STATE_LABELS),
            "final_value": ys[-1],
            "time": [ts[-1]] * len(STATE_LABELS),
        }
    )


def plot_state_time_series(
    ts: np.ndarray,
    ys: np.ndarray,
    *,
    title: str,
    labels: tuple[str, ...] = STATE_LABELS,
    overlay: np.ndarray | None = None,
    overlay_name: str = "reference",
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    colors = ("tab:blue", "tab:orange", "tab:green")

    for idx, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(ts, ys[:, idx], color=color, linewidth=2, label=label)
        if overlay is not None:
            ax.plot(ts, overlay[:, idx], color="black", linestyle="--", linewidth=1.3, label=overlay_name)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("t")
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_3d_trajectory(
    ys: np.ndarray,
    *,
    title: str,
    overlay: np.ndarray | None = None,
    labels: tuple[str, ...] = STATE_LABELS,
) -> tuple[plt.Figure, object]:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ys[:, 0], ys[:, 1], ys[:, 2], linewidth=2, label="primary")
    if overlay is not None:
        ax.plot(overlay[:, 0], overlay[:, 1], overlay[:, 2], linestyle="--", linewidth=1.3, label="overlay")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(title)
    ax.legend(loc="best")
    return fig, ax


def plot_error_curves(ts: np.ndarray, reference: np.ndarray, candidate: np.ndarray, *, title: str) -> tuple[plt.Figure, np.ndarray]:
    error = np.abs(candidate - reference)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for idx, ax in enumerate(axes):
        ax.plot(ts, error[:, idx], linewidth=2)
        ax.set_ylabel(f"|e_{STATE_LABELS[idx]}|")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("t")
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes
