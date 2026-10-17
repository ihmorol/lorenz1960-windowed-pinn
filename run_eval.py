"""Entry point: evaluate a trained model against the reference. No retraining.

    python run_eval.py                # the run of record in src/pinn/
    python run_eval.py runs/5x70      # any sweep run

Reloads the checkpoint, re-derives the network shape from the run's own summary,
and reports the error table plus the full run summary. The reference
solution is used here and only here -- never during training.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402

from pinn.config import Config  # noqa: E402
from pinn.sweep import config_for  # noqa: E402
from pinn.train import evaluate, load_run, residual_at, run_summary  # noqa: E402

if __name__ == "__main__":
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Config().results_path
    model, history, cfg = load_run(config_for(run_dir))

    print(f"\n{run_dir}  ({cfg.label}, {sum(p.numel() for p in model.parameters())} params)\n")
    print(evaluate(model, cfg).to_string(index=False))

    # Residual on a dense grid the network never saw: does the ODE still hold?
    dense = np.linspace(*cfg.t_span, 2001)
    print(f"\nmean squared residual off the collocation set: "
          f"{np.mean(residual_at(model, dense) ** 2):.6e}\n")
    print(run_summary(model, history, cfg).T.to_string(header=False))
