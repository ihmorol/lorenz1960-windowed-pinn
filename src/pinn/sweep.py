"""Architecture sweep: depth x width, one run each, at a single fixed seed.

Nine configurations -- depth in {3, 4, 5} crossed with width in {50, 60, 70} --
trained under otherwise identical settings and written to ``runs/<depth>x<width>/``.

Because :attr:`Config.seed` drives both the weight initialisation and the Latin
hypercube draw, every architecture sees the *same* 3000 collocation points, so
differences between cells are attributable to the network alone. With one seed
per cell there is no estimate of run-to-run variance, so the resulting table
reports what each architecture achieved on this seed; it does not establish that
one architecture is better than another. Pass ``seeds=(0, 1, 2)`` to
:func:`sweep` if that stronger claim is ever needed.
"""
from __future__ import annotations

import itertools
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from . import viz as figures
from .config import Config
from .history import FLOAT_FORMAT
from .train import main as train_and_save

DEPTHS = (3, 4, 5)
WIDTHS = (50, 60, 70)
SNAPSHOT_EVERY = 50
COMPARISON_METRIC = "rmse_combined_l2"


def sweep_config(base: Config, depth: int, width: int, seed: int | None = None) -> Config:
    """Derive one run directory; causal settings receive a collision-resistant tag."""
    cfg = replace(base, depth=depth, width=width, seed=base.seed if seed is None else seed,
                  snapshot_every=base.snapshot_every or SNAPSHOT_EVERY)
    tag = cfg.arch if seed is None else f"{cfg.arch}_seed{seed}"
    if cfg.causal_eps_schedule:
        settings = {k: v for k, v in cfg.record().items()
                    if k not in ("results_dir", "ckpt_dir", "runs_dir")}
        tag += "_" + hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:8]
    root = f"{base.runs_dir}/{tag}"
    return replace(
        cfg,
        results_dir=root,
        ckpt_dir=f"{root}/history",
    )


def run_one(cfg: Config, resume: bool = False) -> pd.DataFrame:
    """Train one architecture and return the summary row it just wrote.

    With ``resume``, a run whose ``run_summary.csv`` already exists is reused
    rather than retrained -- the runs are deterministic, so retraining one would
    only reproduce it. A run interrupted mid-training has no summary and is
    therefore redone.
    """
    summary = cfg.results_path / "run_summary.csv"
    if resume and summary.exists():
        cfg.ensure_record()
        print(f"[skip]  {cfg.arch} already complete -> {summary}", flush=True)
        return pd.read_csv(summary)
    train_and_save(cfg)
    return pd.read_csv(summary)


def sweep(
    base: Config | None = None,
    depths: tuple[int, ...] = DEPTHS,
    widths: tuple[int, ...] = WIDTHS,
    seeds: tuple[int, ...] = (),
    metric: str = COMPARISON_METRIC,
    resume: bool = False,
) -> pd.DataFrame:
    """Run the full grid, write ``runs/comparison.csv`` and the sweep figures.

    ``runs/<arch>/`` is overwritten on rerun: the runs are deterministic at a fixed
    seed, so a rerun reproduces the same bytes. Nothing here touches the
    run-of-record under ``src/pinn/``.
    """
    base = base or Config()
    combos = list(itertools.product(depths, widths, seeds or (base.seed,)))
    out = base.runs_path
    out.mkdir(parents=True, exist_ok=True)

    rows: list[pd.DataFrame] = []
    started = time.perf_counter()
    for k, (depth, width, seed) in enumerate(combos, start=1):
        cfg = sweep_config(base, depth, width, seed if seeds else None)
        elapsed = time.perf_counter() - started
        print(f"\n=== [{k}/{len(combos)}] {cfg.arch} seed {cfg.seed} "
              f"({elapsed / 60:.1f} min elapsed) -> {cfg.results_path} ===", flush=True)
        rows.append(run_one(cfg, resume=resume))

    comparison = pd.concat(rows, ignore_index=True).sort_values(
        ["depth", "width", "seed"]).reset_index(drop=True)
    comparison.to_csv(out / "comparison.csv", index=False, float_format=FLOAT_FORMAT)

    figures.generate_sweep(comparison, out / "figures", metric)
    print(f"\n[sweep] {len(combos)} runs in {(time.perf_counter() - started) / 60:.1f} min")
    print(f"[sweep] comparison table -> {out / 'comparison.csv'}")
    print(comparison[["arch", "n_params", metric, "final_loss",
                      "residual_generalisation_gap", "wall_clock_s"]].to_string(index=False))
    return comparison


def load_comparison(runs_dir: str | Path = "runs") -> pd.DataFrame:
    return pd.read_csv(Path(runs_dir) / "comparison.csv")


def config_for(run_dir: str | Path) -> Config:
    """Rebuild the Config a finished run was trained with, from its own summary.

    Needed to reload a checkpoint: the weights only fit a network of the same
    shape. Falls back to the defaults for runs predating ``run_summary.csv``,
    which is correct for the 4x60 run of record.
    """
    run_dir = Path(run_dir)
    default = Config()
    ckpt = default.ckpt_path if run_dir.resolve() == default.results_path.resolve() else run_dir / "history"
    paths = {"results_dir": str(run_dir), "ckpt_dir": str(ckpt)}
    manifest = ckpt / "config.json"
    if manifest.exists():
        return Config.from_record({**json.loads(manifest.read_text()), **paths})
    summary = run_dir / "run_summary.csv"
    if not summary.exists():
        return replace(Config(), **paths) if run_dir != Config().results_path else Config()

    row = pd.read_csv(summary).iloc[0]
    return replace(
        Config(), **paths,
        depth=int(row["depth"]), width=int(row["width"]), activation=str(row["activation"]),
        ic=str(row["ic"]), seed=int(row["seed"]), epochs=int(row["epochs"]),
        n_collocation=int(row["n_collocation"]),
        t_span=(float(row["t_start"]), float(row["t_end"])),
        lbfgs_iters=int(row["lbfgs_iters"]), gamma=float(row["gamma"]),
        lr_start=float(row["lr_start"]), lr_end=float(row["lr_end"]),
        dtype=str(row.get("dtype", "float32")), ic_scale=str(row.get("ic_scale", "span")),
        collocation=str(row.get("collocation", "lhs")), n_eval=int(row.get("n_eval", 1001)),
        lr_decay=None if pd.isna(row.get("lr_decay", float("nan"))) else float(row["lr_decay"]),
        lr_decay_every=int(row.get("lr_decay_every", 5000)),
        problem=str(row.get("problem", "lorenz1960")), n_windows=int(row.get("n_windows", 1)),
        causal_eps_schedule=tuple(float(e) for e in str(row.get("causal_eps_schedule", "")).split()
                                  if e not in ("", "nan")),
        causal_delta=float(row.get("causal_delta", 0.99)), causal_max_iters=int(row.get("causal_max_iters", 0)),
        warm_start=bool(row.get("warm_start", False)),
        k=float(row.get("k", 2.0)), l=float(row.get("l", 1.0)),
        initial_state=tuple(json.loads(row["initial_state"])) if "initial_state" in row else (0.5, 0.75, 1.0),
        end_state=tuple(json.loads(row["end_state"])) if "end_state" in row and pd.notna(row["end_state"])
                  and row["end_state"] != "null" else None,
    )


if __name__ == "__main__":
    sweep()
