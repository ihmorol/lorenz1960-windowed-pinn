"""Reference-free physics checks on the candidate run's prediction (R7).

Loads the candidate-1536pt windowed checkpoint, predicts the 13,265-point
evaluation grid, and reports two dynamical-systems metrics that need no
reference fit: drift of the conserved quadratic E = 16x^2 + y^2 (k=2, l=1)
and the dominant FFT period, each against the same metric on the DOP853
reference.

Usage:
    python scripts/paper_physics_checks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from pinn.config import reference_trajectory  # noqa: E402
from pinn.pinn import build_model  # noqa: E402
from pinn.sweep import config_for  # noqa: E402

CANDIDATE = _REPO / "runs" / "causal-window" / "candidate-1536pt" / "4x60_f64_unit_win27_causal_warm_f1cbeebd"
OUT = _REPO / "runs" / "paper_ablations" / "physics_checks.json"


def dominant_period(t: np.ndarray, signal: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
    freqs = np.fft.rfftfreq(t.size, t[1] - t[0])
    return float(1.0 / freqs[1:][np.argmax(spectrum[1:])])


def invariant_drift(ys: np.ndarray) -> float:
    e = 16.0 * ys[:, 0] ** 2 + ys[:, 1] ** 2
    return float(np.abs(e - e[0]).max() / abs(e[0]))


def main() -> None:
    cfg = config_for(CANDIDATE)
    model = build_model(cfg).to(dtype=torch.float64)
    model.load_state_dict(torch.load(CANDIDATE / "history" / "pinn.pt", map_location="cpu"))
    model.eval()

    t = np.linspace(*cfg.t_span, 13265)
    with torch.no_grad():
        pred = model(torch.tensor(t.reshape(-1, 1), dtype=torch.float64)).numpy()
    _, ref = reference_trajectory(cfg, n=13265)

    report = {
        "invariant_drift_prediction": invariant_drift(pred),
        "invariant_drift_reference": invariant_drift(ref),
        "period_prediction": dominant_period(t, pred[:, 1]),
        "period_reference": dominant_period(t, ref[:, 1]),
        "eval_points": len(t),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
