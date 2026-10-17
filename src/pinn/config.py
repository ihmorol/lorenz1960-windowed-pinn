"""Central configuration and locked-baseline access for the FYDP-2 PINN."""
from __future__ import annotations

import sys
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

_SRC_ROOT = Path(__file__).resolve().parents[1]   # .../src
_REPO_ROOT = _SRC_ROOT.parent                     # repository root (anchors results/ and data/)
sys.path.insert(0, str(_SRC_ROOT / "baseline"))

from lorenz1960_baseline import (  # noqa: E402
    Lorenz1960Config,
    compute_error_metrics,
    lorenz1960_coefficients,
    solve_lorenz1960_scipy,
)

ACTIVATIONS = ("tanh", "relu", "sigmoid", "gelu", "swish")
IC_MODES = ("hard", "soft")


@dataclass(frozen=True)
class Config:
    k: float = 2.0
    l: float = 1.0
    initial_state: tuple[float, float, float] = (0.5, 0.75, 1.0)
    t_span: tuple[float, float] = (0.0, 1.0)
    problem: str = "lorenz1960"
    end_state: tuple[float, float, float] | None = None   # set for a two-point BVP
    n_windows: int = 1        # >1: one network per time window, chained by end state
    # Causal training (Wang, Sankaran & Perdikaris 2024, Algorithm 1): per window, run
    # Adam under each eps in turn, advancing when min_i w_i > causal_delta or after
    # causal_max_iters. Empty schedule = plain mean-squared residual.
    causal_eps_schedule: tuple[float, ...] = ()
    causal_delta: float = 0.99
    causal_max_iters: int = 0
    warm_start: bool = False
    shared_network: bool = False   # windows share one network: sequential single-network training

    depth: int = 4
    width: int = 60
    activation: str = "tanh"

    ic: str = "hard"          # "hard" trial solution, or "soft" IC penalty (Raissi-style)
    gamma: float = 1.0        # soft-IC penalty weight (unused when ic="hard")
    ic_scale: str = "span"    # "span": g = (t - t0)/(tf - t0); "unit": g = t - t0

    epochs: int = 20000       # Adam epochs
    lbfgs_iters: int = 0   # L-BFGS fine-tuning iterations (0 disables)
    lr_start: float = 1e-3
    lr_end: float = 1e-4
    seed: int = 0

    n_collocation: int = 3000
    n_eval: int = 1001
    points_per_unit: float | None = None    # overrides n_collocation as density x window length
    eval_per_unit: float | None = None      # overrides n_eval the same way
    collocation: str = "lhs"                # "lhs" | "uniform"
    dtype: str = "float32"                  # "float32" | "float64"
    lr_decay: float | None = None           # None: linear lr_start -> lr_end; else StepLR gamma
    lr_decay_every: int = 5000

    log_every: int = 10       # epochs between gradient/loss-component diagnostics
    eval_every: int = 100     # epochs between reference-solution error evaluations
    print_every: int = 250    # epochs between console progress lines (0 = silent)

    # Epochs between full per-collocation-point snapshots. 0 disables snapshotting
    # entirely, which is the default so the run-of-record pipeline is unchanged;
    # the architecture sweep sets it to 50.
    snapshot_every: int = 0
    checkpoint_every: int = 0   # Adam epochs between resumable checkpoints (0 = off)

    results_dir: str = "src/pinn/results"
    ckpt_dir: str = "src/pinn/history"
    runs_dir: str = "runs"

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        if self.ic not in IC_MODES:
            raise ValueError(f"ic must be one of {IC_MODES}")
        if self.ic_scale not in ("span", "unit"):
            raise ValueError("ic_scale must be 'span' or 'unit'")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")
        if self.collocation not in ("lhs", "uniform"):
            raise ValueError("collocation must be 'lhs' or 'uniform'")
        if self.problem != "lorenz1960":
            raise ValueError("only the lorenz1960 problem is implemented")
        span = self.t_span[1] - self.t_span[0]
        if not np.isfinite(span) or span <= 0 or self.n_windows < 1 or self.depth < 1 or self.width < 1:
            raise ValueError("time span, window count, depth, and width must be positive")
        if len(self.initial_state) != 3 or not np.isfinite((self.k, self.l, *self.initial_state)).all():
            raise ValueError("Lorenz coefficients and three initial-state values must be finite")
        if self.end_state is not None and (len(self.end_state) != 3 or
                                           not np.isfinite(self.end_state).all()):
            raise ValueError("end_state must have three finite values")
        if not np.isfinite(self.gamma) or (self.ic == "soft" and self.gamma <= 0):
            raise ValueError("soft initial conditions require a positive finite gamma")
        if not np.isfinite((self.lr_start, self.lr_end)).all() or self.lr_start <= 0 or self.lr_end <= 0 \
                or self.lr_decay_every < 1:
            raise ValueError("learning rates and decay interval must be positive")
        if self.lr_decay is not None and (not np.isfinite(self.lr_decay) or not 0 < self.lr_decay <= 1):
            raise ValueError("lr_decay must be in (0, 1]")
        if not np.isfinite(self.causal_delta) or not 0 < self.causal_delta < 1 or \
                any(not np.isfinite(e) or e <= 0 for e in self.causal_eps_schedule):
            raise ValueError("causal_delta must be in (0, 1) and eps values must be positive")
        if self.epochs < 1 or min(self.lbfgs_iters, self.causal_max_iters, self.checkpoint_every,
               self.snapshot_every, self.print_every) < 0 or min(self.log_every, self.eval_every) < 1:
            raise ValueError("epochs and log intervals must be positive; other counts nonnegative")
        if self.shared_network and self.n_windows < 2:
            raise ValueError("shared_network needs n_windows > 1")
        if self.n_windows > 1 and self.end_state is not None:
            raise ValueError("global end_state is not implemented for windowed training")
        if self.points_per_unit is not None:
            if not np.isfinite(self.points_per_unit) or self.points_per_unit <= 0:
                raise ValueError("points_per_unit must be finite and positive")
            object.__setattr__(self, "n_collocation", int(round(self.points_per_unit * span)))
        if self.eval_per_unit is not None:
            if not np.isfinite(self.eval_per_unit) or self.eval_per_unit <= 0:
                raise ValueError("eval_per_unit must be finite and positive")
            object.__setattr__(self, "n_eval", int(round(self.eval_per_unit * span)) + 1)
        if self.n_collocation < self.n_windows or self.n_eval < 2:
            raise ValueError("need at least one collocation point per window and two evaluation points")

    def record(self) -> dict:
        """Exact run settings; the JSON file is the authority for reuse and reload."""
        record = asdict(self)
        record.update(results_dir=str(self.results_path), ckpt_dir=str(self.ckpt_path),
                      runs_dir=str(self.runs_path))
        return record

    @classmethod
    def from_record(cls, record: dict) -> "Config":
        for name in ("initial_state", "t_span", "causal_eps_schedule"):
            record[name] = tuple(record[name])
        if record["end_state"] is not None:
            record["end_state"] = tuple(record["end_state"])
        return cls(**record)

    def matches(self, record: dict) -> bool:
        """Same training settings as a saved record, wherever either was written."""
        return settings(record) == settings(self.record())

    def ensure_record(self) -> None:
        path = self.ckpt_path / "config.json"
        if path.exists():
            if not self.matches(json.loads(path.read_text())):
                raise ValueError(f"run configuration differs from {path}; choose a new run directory")
            return
        if (self.ckpt_path / "pinn.pt").exists() or (self.ckpt_path / "progress.pt").exists():
            raise ValueError(f"existing run lacks {path}; choose a new run directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(".tmp")
        pending.write_text(json.dumps(self.record(), indent=2) + "\n")
        pending.replace(path)

    @property
    def coefficients(self) -> np.ndarray:
        return lorenz1960_coefficients(self.k, self.l)

    @property
    def results_path(self) -> Path:
        return _resolve(self.results_dir)

    @property
    def ckpt_path(self) -> Path:
        return _resolve(self.ckpt_dir)

    @property
    def runs_path(self) -> Path:
        return _resolve(self.runs_dir)

    @property
    def spec(self):
        from .problems import PROBLEMS
        return PROBLEMS[self.problem](self.coefficients, self.initial_state, self.t_span,
                                      self.end_state, lambda t: reference_at(self, t))

    @property
    def torch_dtype(self):
        import torch
        return torch.float64 if self.dtype == "float64" else torch.float32

    @property
    def arch(self) -> str:
        """Directory-safe run tag: ``4x60``, plus a suffix per non-default knob."""
        tag = f"{self.depth}x{self.width}"
        if self.dtype == "float64":
            tag += "_f64"
        if self.ic_scale == "unit":
            tag += "_unit"
        if self.n_windows > 1:
            tag += f"_win{self.n_windows}"
        if self.causal_eps_schedule:
            tag += "_causal"
        if self.warm_start:
            tag += "_warm"
        if self.shared_network:
            tag += "_shared"
        return tag

    @property
    def label(self) -> str:
        """Short run descriptor used in figure titles."""
        return (
            f"{self.depth}x{self.width} {self.activation}, {self.ic} IC, "
            f"{self.n_collocation} {self.collocation.upper() if self.collocation == 'lhs' else self.collocation} points"
            + (f", {self.n_windows} windows" if self.n_windows > 1 else "")
            + (", causal" if self.causal_eps_schedule else "")
        )


_PATHS = ("results_dir", "ckpt_dir", "runs_dir")


def settings(record: dict) -> dict:
    """A record without machine-specific output paths; fields added after it was
    written take their defaults, which is how the run was actually trained."""
    full = {f.name: f.default for f in fields(Config)} | record
    return {k: v for k, v in json.loads(json.dumps(full)).items() if k not in _PATHS}


def _resolve(path: str) -> Path:
    """Anchor relative output paths to the repo root, not the caller's cwd."""
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


def _baseline(cfg: Config, n: int = 1001) -> Lorenz1960Config:
    return Lorenz1960Config(
        k=cfg.k, l=cfg.l, initial_state=cfg.initial_state, t_span=cfg.t_span, n_eval=n
    )


def reference_trajectory(cfg: Config, n: int = 1001) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth (t, [x,y,z]) from the locked SciPy baseline solver."""
    t, ys, _ = solve_lorenz1960_scipy(config=_baseline(cfg, n))
    return t, ys


def reference_at(cfg: Config, t: np.ndarray) -> np.ndarray:
    """Ground-truth states at arbitrary, possibly unsorted times.

    Integrated directly at the requested times rather than interpolated from
    the uniform evaluation grid.
    """
    t = np.clip(np.asarray(t, dtype=float).reshape(-1), *cfg.t_span)   # float32 grids overshoot tf
    if not len(t):
        return np.empty((0, 3))
    unique, inverse = np.unique(t, return_inverse=True)
    if len(unique) == 1 and unique[0] == cfg.t_span[0]:
        return np.tile(np.asarray(cfg.initial_state, float), (len(t), 1))
    _, ys, _ = solve_lorenz1960_scipy(config=_baseline(cfg), t_eval=unique)
    out = ys[inverse]
    return out


__all__ = ["Config", "reference_trajectory", "reference_at", "compute_error_metrics",
           "ACTIVATIONS", "IC_MODES"]
