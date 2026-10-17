"""Optimisation telemetry recorded during training, consumed by :mod:`pinn.figures`."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .pinn import PINN, ResidualParts

# Round-trippable for float32 (the dtype the network trains in) while costing
# ~40% fewer bytes than pandas' 17-significant-digit default.
FLOAT_FORMAT = "%.9g"

SNAPSHOT_COLUMNS = (
    "epoch", "i", "t", "trained_until",         # index and trained prefix
    "n_x", "n_y", "n_z",                        # raw network output N(t)
    "x", "y", "z",                              # trial solution u_T = u0 + g(t) N(t)
    "dx_dt", "dy_dt", "dz_dt",                  # autograd time derivative
    "f_x", "f_y", "f_z",                        # physics RHS f(u_T)
    "r_x", "r_y", "r_z",                        # residual r = du_T/dt - f(u_T)
    "r_sq",                                     # r_x^2 + r_y^2 + r_z^2
    "loss_contribution",                        # r_sq / (3 trained points); raw prefix residual
    "ref_x", "ref_y", "ref_z",                  # EVALUATION ONLY - never enters the loss
    "err_x", "err_y", "err_z", "err_norm",      # EVALUATION ONLY
)

CONVERGED_RESIDUAL = 1e-4  # |r| threshold for the per-point convergence-epoch column


def flat_params(model: PINN) -> Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def flat_grads(model: PINN) -> Tensor:
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in model.parameters()]).detach()


def _grad_norms(model: PINN) -> tuple[float, dict[str, float]]:
    """Global L2 gradient norm plus one norm per weight matrix."""
    total = 0.0
    per_layer: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        sq = float(p.grad.pow(2).sum())
        total += sq
        if name.endswith("weight"):
            per_layer[f"layer {len(per_layer) + 1}"] = sq**0.5
    return total**0.5, per_layer


@dataclass
class TrainHistory:
    """Per-iteration optimisation record.

    ``loss`` holds every iteration (Adam, then L-BFGS); the remaining series are
    sampled every ``cfg.log_every`` / ``cfg.eval_every`` epochs so the
    instrumentation stays cheap.
    """

    loss: list[float] = field(default_factory=list)
    loss_phase: list[str] = field(default_factory=list)
    loss_window: list[int] = field(default_factory=list)
    adam_iters: int = 0
    resumed_from: int = 0
    wall_clock_s: float = 0.0
    saved_n_snapshots: int = 0

    # Live snapshot writer for the run, when ``cfg.snapshot_every`` is enabled.
    # Not part of the CSV round trip: the breakdown files are the record.
    snapshots: "SnapshotWriter | None" = field(default=None, repr=False, compare=False)
    param_epochs: list[int] = field(default_factory=list, repr=False, compare=False)
    param_trail: list[np.ndarray] = field(default_factory=list, repr=False, compare=False)
    grad_trail: list[np.ndarray] = field(default_factory=list, repr=False, compare=False)
    eps_marks: list[tuple[int, float]] = field(default_factory=list)
    stage_status: list[dict] = field(default_factory=list)
    min_w: list[float] = field(default_factory=list)
    window_marks: list[int] = field(default_factory=list)
    weight_profiles: list[np.ndarray] = field(default_factory=list, repr=False, compare=False)

    log_epoch: list[int] = field(default_factory=list)
    logged_loss: list[float] = field(default_factory=list)
    residual_loss: list[float] = field(default_factory=list)
    ic_loss: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    update_norm: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    layer_grad_norms: dict[str, list[float]] = field(default_factory=dict)

    ref_epoch: list[int] = field(default_factory=list)
    ref_mse: list[float] = field(default_factory=list)
    ref_until: list[float] = field(default_factory=list)

    def record_step(
        self,
        epoch: int,
        *,
        model: PINN,
        loss: Tensor,
        residual: Tensor,
        ic: Tensor,
        lr: float,
        params_before: Tensor,
    ) -> None:
        """Snapshot gradients and the step just taken. Call after ``optimizer.step()``."""
        grad_norm, per_layer = _grad_norms(model)
        self.log_epoch.append(epoch)
        self.logged_loss.append(float(loss.detach()))
        self.residual_loss.append(float(residual.detach()))
        self.ic_loss.append(float(ic.detach()))
        self.grad_norm.append(grad_norm)
        self.lr.append(lr)
        self.update_norm.append(float((flat_params(model) - params_before).norm()))
        for name, value in per_layer.items():
            self.layer_grad_norms.setdefault(name, []).append(value)

    def record_reference(self, epoch: int, mse: float, trained_until: float | None = None) -> None:
        self.ref_epoch.append(epoch)
        self.ref_mse.append(mse)
        self.ref_until.append(float(trained_until) if trained_until is not None else float("nan"))

    def diagnostics_frame(self) -> pd.DataFrame:
        """Sampled optimiser diagnostics, one row per logged epoch."""
        return pd.DataFrame({
            "epoch": self.log_epoch,
            "loss": self.logged_loss,
            "residual_loss": self.residual_loss,
            "ic_loss": self.ic_loss,
            "grad_norm": self.grad_norm,
            "update_norm": self.update_norm,
            "lr": self.lr,
            **{name.replace(" ", "_") + "_grad_norm": v
               for name, v in self.layer_grad_norms.items()},
        })

    def reference_frame(self) -> pd.DataFrame:
        """Error against the trusted solution, sampled on its own cadence."""
        return pd.DataFrame({"epoch": self.ref_epoch, "ref_mse": self.ref_mse,
                             "trained_until": self.ref_until or [float("nan")] * len(self.ref_epoch)})

    @classmethod
    def from_saved(cls, data_dir: Path | str, results_dir: Path | str | None = None) -> "TrainHistory":
        """Reload a finished run's telemetry, so figures can be redrawn without retraining."""
        out = Path(data_dir)
        results = Path(results_dir) if results_dir is not None else out.parent
        diag = pd.read_csv(out / "training_diagnostics.csv")
        ref = pd.read_csv(out / "reference_error.csv")
        loss_frame = pd.read_csv(out / "loss_history.csv")
        loss = loss_frame["loss"].tolist()
        layer_cols = [c for c in diag.columns if c.startswith("layer_") and c.endswith("_grad_norm")]
        causal = out / "causal.csv"
        marks = out / "marks.csv"
        stages = out / "stages.csv"
        extra = {}
        if causal.exists():
            extra["min_w"] = pd.read_csv(causal)["min_w"].tolist()
        if marks.exists():
            m = pd.read_csv(marks)
            extra["eps_marks"] = [(int(i), float(e)) for i, e in zip(m["iteration"], m["eps"]) if e == e]
            extra["window_marks"] = [int(i) for i, k in zip(m["iteration"], m["kind"]) if k == "window"]
        if stages.exists():
            extra["stage_status"] = pd.read_csv(stages).to_dict("records")
        summary_path = results / "run_summary.csv"
        summary = pd.read_csv(summary_path).iloc[0] if summary_path.exists() else None
        breakdown_count = len(list((results / "breakdown").glob("epoch_*.csv")))
        return cls(
            **extra,
            loss=loss,
            loss_phase=loss_frame["phase"].tolist() if "phase" in loss_frame else [],
            loss_window=loss_frame["window"].astype(int).tolist() if "window" in loss_frame else [],
            adam_iters=(int(summary["adam_steps"]) if summary is not None and "adam_steps" in summary
                        else (sum(loss_frame["phase"] == "adam") if "phase" in loss_frame
                              else (len(extra["min_w"]) if "min_w" in extra
                                    else int(diag["epoch"].max()) + 1))),
            wall_clock_s=float(summary["wall_clock_s"]) if summary is not None else 0.0,
            saved_n_snapshots=(breakdown_count or
                               (int(summary["n_snapshots"]) if summary is not None else 0)),
            log_epoch=diag["epoch"].tolist(),
            logged_loss=diag["loss"].tolist(),
            residual_loss=diag["residual_loss"].tolist(),
            ic_loss=diag["ic_loss"].tolist(),
            grad_norm=diag["grad_norm"].tolist(),
            update_norm=diag["update_norm"].tolist(),
            lr=diag["lr"].tolist(),
            layer_grad_norms={c.replace("_grad_norm", "").replace("_", " "): diag[c].tolist()
                              for c in layer_cols},
            ref_epoch=ref["epoch"].tolist(),
            ref_mse=ref["ref_mse"].tolist(),
            ref_until=ref["trained_until"].tolist() if "trained_until" in ref else [],
        )


class SnapshotWriter:
    """Streams the full per-collocation-point state to ``breakdown/epoch_<n>.csv``.

    One file per snapshot epoch, ``Nc`` rows wide (see :data:`SNAPSHOT_COLUMNS`).
    Single-domain Adam snapshots use the training step's residual tensors.
    Windowed snapshots evaluate the full model separately and mask windows
    that have not yet been trained.

    Running per-point statistics are accumulated as the snapshots stream past and
    written once at the end as ``point_summary.csv``, which answers "which
    collocation points were hard, and when" without re-reading the breakdown.
    """

    def __init__(self, outdir: Path | str, t: np.ndarray, reference: np.ndarray) -> None:
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t = np.asarray(t, dtype=float).reshape(-1)
        self.reference = np.asarray(reference, dtype=float)
        self.n_points = self.t.size
        self.epochs: list[int] = []

        zeros = np.zeros(self.n_points)
        self._sum, self._sumsq = zeros.copy(), zeros.copy()
        self._count = np.zeros(self.n_points, dtype=int)
        self._max, self._argmax = zeros.copy(), np.zeros(self.n_points, dtype=int)
        self._first = np.full(self.n_points, np.nan)
        self._last, self._last_contrib, self._last_err = zeros.copy(), zeros.copy(), zeros.copy()
        self._converged_at = np.full(self.n_points, np.nan)

    def write(self, epoch: int, parts: ResidualParts, trained_until: float | None = None) -> Path:
        """Record a snapshot; ``trained_until`` masks the untrained future."""
        cols = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in parts._asdict().items()}
        r = cols["r"]
        r_sq = (r ** 2).sum(axis=1)
        r_norm = np.sqrt(r_sq)
        err = cols["u"] - self.reference
        err_norm = np.linalg.norm(err, axis=1)
        valid = self.t <= trained_until if trained_until is not None else np.ones(self.n_points, dtype=bool)
        if not valid.any():
            raise ValueError("snapshot has no trained collocation points")
        contrib = r_sq / (3.0 * valid.sum())

        frame = pd.DataFrame({
            "epoch": epoch, "i": np.arange(self.n_points), "t": self.t,
            "trained_until": trained_until if trained_until is not None else float(self.t.max()),
            **{"n_" + a: cols["n"][:, j] for j, a in enumerate("xyz")},
            **{a: cols["u"][:, j] for j, a in enumerate("xyz")},
            **{"d" + a + "_dt": cols["dudt"][:, j] for j, a in enumerate("xyz")},
            **{"f_" + a: cols["f"][:, j] for j, a in enumerate("xyz")},
            **{"r_" + a: r[:, j] for j, a in enumerate("xyz")},
            "r_sq": r_sq, "loss_contribution": contrib,
            **{"ref_" + a: self.reference[:, j] for j, a in enumerate("xyz")},
            **{"err_" + a: err[:, j] for j, a in enumerate("xyz")},
            "err_norm": err_norm,
        })[list(SNAPSHOT_COLUMNS)]
        measured = [c for c in frame if c not in ("epoch", "i", "t", "trained_until")
                    and not c.startswith("ref_")]
        frame.loc[~valid, measured] = np.nan

        path = self.dir / ("epoch_%06d.csv" % epoch)
        frame.to_csv(path, index=False, float_format=FLOAT_FORMAT)

        self._accumulate(epoch, np.where(valid, r_norm, np.nan),
                         np.where(valid, contrib, np.nan), np.where(valid, err_norm, np.nan))
        return path

    def _accumulate(self, epoch: int, r_norm, contrib, err_norm) -> None:
        self.epochs.append(epoch)
        valid = np.isfinite(r_norm)
        self._count += valid
        self._sum += np.nan_to_num(r_norm)
        self._sumsq += np.nan_to_num(r_norm) ** 2
        beat = valid & (r_norm > self._max)
        self._max[beat], self._argmax[beat] = r_norm[beat], epoch
        first = valid & np.isnan(self._first)
        self._first[first] = r_norm[first]
        self._last[valid], self._last_contrib[valid], self._last_err[valid] = (
            r_norm[valid], contrib[valid], err_norm[valid])
        hit = valid & np.isnan(self._converged_at) & (r_norm < CONVERGED_RESIDUAL)
        self._converged_at[hit] = epoch

    @classmethod
    def replay(cls, breakdown_dir: Path | str) -> "SnapshotWriter":
        """Rebuild the running statistics from every ``epoch_*.csv`` already on disk."""
        files = sorted(Path(breakdown_dir).glob("epoch_*.csv"))
        first = pd.read_csv(files[0])
        writer = cls(breakdown_dir, first.t.to_numpy(), first[["ref_x", "ref_y", "ref_z"]].to_numpy())
        for f in files:
            frame = pd.read_csv(f)
            writer._accumulate(int(frame.epoch.iloc[0]), np.sqrt(frame.r_sq.to_numpy()),
                               frame.loss_contribution.to_numpy(), frame.err_norm.to_numpy())
        return writer

    def summary_frame(self) -> pd.DataFrame:
        """One row per collocation point, aggregated over every snapshot taken."""
        n = len(self.epochs)
        if n == 0:
            return pd.DataFrame(columns=["i", "t"])
        count = np.maximum(self._count, 1)
        mean = self._sum / count
        var = np.maximum(self._sumsq / count - mean ** 2, 0.0)
        frame = pd.DataFrame({
            "i": np.arange(self.n_points),
            "t": self.t,
            "r_init": self._first,
            "r_final": self._last,
            "r_max": self._max,
            "epoch_of_max": self._argmax,
            "r_mean": mean,
            "r_std": np.sqrt(var),
            "loss_contribution_final": self._last_contrib,
            "err_norm_final": self._last_err,
            "epoch_below_1e-4": self._converged_at,
            "n_snapshots": self._count,
        })
        frame["rank_by_final_residual"] = frame["r_final"].rank(ascending=False).astype(int)
        return frame

    def finalize(self, outdir: Path | str) -> Path | None:
        """Write ``point_summary.csv`` beside the run's other tracked tables."""
        if not self.epochs:
            return None
        path = Path(outdir) / "point_summary.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_frame().to_csv(path, index=False, float_format=FLOAT_FORMAT)
        return path


def load_snapshots(breakdown_dir: Path | str) -> pd.DataFrame:
    """Concatenate every ``epoch_*.csv`` in a breakdown directory into one frame."""
    files = sorted(Path(breakdown_dir).glob("epoch_*.csv"))
    if not files:
        raise FileNotFoundError("no epoch_*.csv under %s" % breakdown_dir)
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def point_history(breakdown_dir: Path | str, i: int) -> pd.DataFrame:
    """The training history of one collocation point, one row per snapshot.

    Slices the breakdown on demand rather than storing a second, transposed copy
    of the same numbers.
    """
    files = sorted(Path(breakdown_dir).glob("epoch_*.csv"))
    if not files:
        raise FileNotFoundError("no epoch_*.csv under %s" % breakdown_dir)
    rows = []
    for f in files:
        frame = pd.read_csv(f)
        match = frame[frame["i"] == i]
        if match.empty:
            raise IndexError("collocation point %d not present in %s" % (i, f.name))
        rows.append(match)
    return pd.concat(rows, ignore_index=True).sort_values("epoch").reset_index(drop=True)


def residual_grid(
    breakdown_dir: Path | str, n_bins: int = 240, column: str = "r_sq", sqrt: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce a breakdown directory to an ``(epoch x t)`` field for plotting.

    Collocation points are a scattered Latin-hypercube sample, so the raw cube is
    not a grid. Binning ``t`` into ``n_bins`` equal intervals and taking the median
    within each bin gives a regular surface without pretending to a resolution the
    sample does not have. Reads one snapshot at a time, so memory stays flat.

    Returns ``(epochs, t_centres, values)`` with ``values`` shaped
    ``(n_epochs, n_bins)`` holding the binned median of ``sqrt(column)``.
    """
    files = sorted(Path(breakdown_dir).glob("epoch_*.csv"))
    if not files:
        raise FileNotFoundError("no epoch_*.csv under %s" % breakdown_dir)

    first = pd.read_csv(files[0], usecols=["t"])
    edges = np.linspace(float(first["t"].min()), float(first["t"].max()), n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    epochs, rows = [], []
    for f in files:
        frame = pd.read_csv(f, usecols=["epoch", "t", column])
        which = np.clip(np.digitize(frame["t"].to_numpy(), edges) - 1, 0, n_bins - 1)
        magnitude = frame[column].to_numpy()
        if sqrt:
            magnitude = np.sqrt(magnitude)
        binned = pd.Series(magnitude).groupby(which).median()
        row = np.full(n_bins, np.nan)
        row[binned.index.to_numpy()] = binned.to_numpy()
        epochs.append(int(frame["epoch"].iloc[0]))
        rows.append(row)

    return np.asarray(epochs), centres, np.vstack(rows)
