from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn.config import Config, reference_trajectory
from pinn.pinn import build_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_directories() -> list[Path]:
    runs = sorted((ROOT / "runs/paper_ablations").glob("r*_seed*"))
    runs += [
        ROOT / "runs/4x60_f64_unit",
        ROOT / "runs/causal-window/candidate-1536pt/4x60_f64_unit_win27_causal_warm_f1cbeebd",
    ]
    if len(runs) != 16:
        raise AssertionError(f"expected 16 archived runs, found {len(runs)}")
    return runs


def verify_checkpoint_hashes(runs: list[Path]) -> None:
    entries = {}
    for line in (ROOT / "artifacts/checkpoints.sha256").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        entries[relative.lstrip("* ")] = expected
    expected_paths = {str((run / "history/pinn.pt").relative_to(ROOT)).replace("\\", "/") for run in runs}
    if set(entries) != expected_paths:
        raise AssertionError("checkpoint manifest paths do not match the 16 archived runs")
    for relative, expected in entries.items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise AssertionError(f"checkpoint hash mismatch: {relative}")


def verify_run_files(runs: list[Path]) -> None:
    required = (
        "history/config.json",
        "history/loss_history.csv",
        "history/pinn.pt",
        "history/reference_error.csv",
        "history/training_diagnostics.csv",
        "metrics.csv",
        "run_summary.csv",
    )
    for run in runs:
        missing = [name for name in required if not (run / name).is_file()]
        if missing:
            raise AssertionError(f"{run.relative_to(ROOT)} is missing {missing}")
        loss = pd.read_csv(run / "history/loss_history.csv")
        summary = pd.read_csv(run / "run_summary.csv").iloc[0]
        windows = int(summary.get("n_windows", 1))
        observed = loss["window"].nunique() if "window" in loss else 1
        lbfgs = loss.loc[loss["phase"] == "lbfgs", "window"].nunique() if "phase" in loss else 1
        if observed != windows or lbfgs != windows:
            raise AssertionError(f"incomplete optimizer history: {run.relative_to(ROOT)}")
        if loss["loss"].isna().any() or not (np.diff(loss["iteration"]) == 1).all():
            raise AssertionError(f"invalid loss history: {run.relative_to(ROOT)}")


def verify_aggregate_table(runs: list[Path]) -> None:
    ablation_runs = [run for run in runs if run.parent == ROOT / "runs/paper_ablations"]
    aggregate = pd.read_csv(ROOT / "runs/paper_ablations/ablations.csv").set_index("arm")
    names = {run.name for run in ablation_runs}
    if set(aggregate.index) != names:
        raise AssertionError("ablations.csv rows do not match the 14 paper ablation runs")
    for run in ablation_runs:
        summary = pd.read_csv(run / "run_summary.csv").iloc[0]
        archived = aggregate.loc[run.name]
        for column in summary.index.intersection(archived.index):
            left, right = summary[column], archived[column]
            if pd.isna(left) and pd.isna(right):
                continue
            if isinstance(left, (int, float, np.number)) and isinstance(right, (int, float, np.number)):
                if not np.isclose(float(left), float(right), rtol=1e-9, atol=1e-12):
                    raise AssertionError(f"aggregate mismatch: {run.name}.{column}")
            elif str(left) != str(right):
                raise AssertionError(f"aggregate mismatch: {run.name}.{column}")


def load_config(run: Path) -> Config:
    record = json.loads((run / "history/config.json").read_text())
    return Config.from_record(record)


def evaluate_checkpoints(runs: list[Path]) -> pd.DataFrame:
    rows = []
    reference_cache: dict[tuple, np.ndarray] = {}
    for index, run in enumerate(runs, start=1):
        cfg = load_config(run)
        model = build_model(cfg).to(dtype=torch.float64 if cfg.dtype == "float64" else torch.float32)
        state = torch.load(run / "history/pinn.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        t = np.linspace(*cfg.t_span, 13265)
        tensor = torch.tensor(t.reshape(-1, 1), dtype=next(model.parameters()).dtype)
        predictions = []
        with torch.no_grad():
            for chunk in tensor.split(2048):
                predictions.append(model(chunk).cpu())
        predicted = torch.cat(predictions).numpy()
        key = (cfg.k, cfg.l, cfg.initial_state, cfg.t_span)
        if key not in reference_cache:
            reference_cache[key] = reference_trajectory(cfg, n=13265)[1]
        error = predicted - reference_cache[key]
        actual = float(np.sqrt(np.mean(np.linalg.norm(error, axis=1) ** 2)))
        archived = float(pd.read_csv(run / "run_summary.csv").iloc[0]["rmse_combined_l2"])
        if not np.isclose(actual, archived, rtol=1e-6, atol=1e-10):
            raise AssertionError(
                f"RMSE mismatch for {run.relative_to(ROOT)}: evaluated={actual:.12g}, archived={archived:.12g}"
            )
        rows.append(
            {
                "run": run.relative_to(ROOT).as_posix(),
                "eval_points": 13265,
                "reference_solver": "DOP853",
                "archived_rmse_combined_l2": archived,
                "evaluated_rmse_combined_l2": actual,
                "relative_difference": abs(actual - archived) / max(abs(archived), np.finfo(float).tiny),
                "checkpoint_sha256": sha256(run / "history/pinn.pt"),
            }
        )
        print(f"[{index:02d}/16] {run.name}: RMSE {actual:.8g}")
    return pd.DataFrame(rows)


def main() -> None:
    started = time.perf_counter()
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    runs = run_directories()
    verify_checkpoint_hashes(runs)
    for line in (ROOT / "artifacts/archive.sha256").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        if sha256(ROOT / relative) != expected:
            raise AssertionError(f"archive hash mismatch: {relative}")
    verify_run_files(runs)
    verify_aggregate_table(runs)
    results = evaluate_checkpoints(runs)
    results.to_csv(ROOT / "artifacts/checkpoint-evaluation.csv", index=False)
    report = {
        "status": "pass",
        "runs": len(runs),
        "evaluation_points_per_run": 13265,
        "reference_solver": "DOP853",
        "rmse_tolerance": {"rtol": 1e-6, "atol": 1e-10},
        "torch_threads": torch.get_num_threads(),
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }
    (ROOT / "artifacts/verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
