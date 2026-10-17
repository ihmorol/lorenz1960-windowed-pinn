"""Paper ablation and seed runs: one artifact per claim in writings/conference-paper.

Fixed problem throughout: k=2, l=1, IC (0.5, 0.75, 1.0), t in [0, 13.26446].
Arms, all derived from the candidate-1536pt run of record via its config.json:
  r1_no_causal_seedS   windowed-27, causal off, Adam capped at 1,300/window
                       (matches the candidate's recorded 34,765 updates)
  r2_no_warm_seedS     windowed-27, causal on, warm start off
  r4_candidate_seedS   the candidate config itself, seeds 1-4 (seed 0 exists)
  r3_causal_single     single network + the causal schedule (isolates windowing)
Optimizer budgets are kept exactly as the candidate's; nothing is increased.
Resume-safe: finished arms are skipped, interrupted ones redone.

Usage:
    python scripts/paper_ablations.py --arms r1,r4 --seeds 1,2,3
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from pinn.sweep import config_for, run_one  # noqa: E402

CANDIDATE = _REPO / "runs" / "causal-window" / "candidate-1536pt" / "4x60_f64_unit_win27_causal_warm_f1cbeebd"
OUT = "runs/paper_ablations"


def arms(base, seeds):
    for s in seeds:
        yield f"r1_no_causal_seed{s}", replace(base, seed=s, causal_eps_schedule=(),
                                               causal_max_iters=0, epochs=1300)
        yield f"r2_no_warm_seed{s}", replace(base, seed=s, warm_start=False)
        yield f"r4_candidate_seed{s}", replace(base, seed=s)
        yield f"r3_causal_single_seed{s}", replace(
            base, seed=s, n_windows=1, n_collocation=39793, collocation="lhs",
            epochs=40000, lbfgs_iters=5000,
            # periodic resume checkpoints only; training math unchanged
            checkpoint_every=2000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="r1,r2,r4,r3",
                    help="comma list of arm prefixes to run (default: all)")
    ap.add_argument("--seeds", default="1,2,3", help="seeds for the windowed arms")
    args = ap.parse_args()
    wanted = tuple(args.arms.split(","))
    seeds = tuple(int(s) for s in args.seeds.split(","))

    base = config_for(CANDIDATE)
    out = _REPO / OUT
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "ablations.csv"
    # a previous invocation (possibly on another machine) may have already
    # recorded arms; keep those rows and replace by arm name as we go
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    started = time.perf_counter()
    for name, cfg in arms(base, seeds):
        if not name.startswith(wanted):
            continue
        cfg = replace(cfg, results_dir=f"{OUT}/{name}", ckpt_dir=f"{OUT}/{name}/history")
        print(f"\n=== {name} ({(time.perf_counter() - started) / 60:.1f} min elapsed) ===",
              flush=True)
        row = run_one(cfg, resume=True).iloc[0]
        entry = {"arm": name, "seed": cfg.seed, "arch": cfg.arch,
                 "final_loss": row["final_loss"],
                 "rmse_combined_l2": row["rmse_combined_l2"],
                 "max_abs_error": row["max_abs_error_combined_l2"],
                 "wall_clock_s": row["wall_clock_s"]}
        rows = [r for r in rows if r.get("arm") != name] + [entry]
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
