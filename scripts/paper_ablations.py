"""Paper ablation and seed runs, one run directory per (arm, seed).

Fixed problem throughout: k=2, l=1, IC (0.5, 0.75, 1.0), t in [0, 13.26446].
Windowed arms derive from the candidate-1536pt run (seed 0, RMSE 6.97e-5):
  r4_candidate   the candidate config, other seeds
  r1_no_causal   causal off, 1,300 Adam/window = 35,100 (candidate seed 0 used 34,765;
                 causal arms stop stages adaptively, so per-seed counts differ)
  r2_no_warm     warm start off; every window starts from its own random init
  r6_sequential  r1 with one network shared by all windows (handoff forgetting)
The single-network arm derives from F1 (runs/4x60_f64_unit: 40k Adam + 5k L-BFGS,
39,793 LHS points, seed 0, RMSE 2.24):
  r3_causal_single  F1 + the candidate's causal schedule, 4 stages x 10,000 Adam cap
Uncontrolled differences are listed in README.md; every row in ablations.csv
carries the actual Adam steps, L-BFGS evaluations, parameter count and device.

Resume-safe: finished runs are reused, interrupted ones continue from progress.pt.

Usage:
    python scripts/paper_ablations.py --arms r1,r4 --seeds 1,2,3 [--out runs/paper_ablations]
    python scripts/paper_ablations.py --collect      # rebuild ablations.csv only
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from pinn.sweep import config_for, run_one  # noqa: E402

CANDIDATE = _REPO / "runs" / "causal-window" / "candidate-1536pt" / "4x60_f64_unit_win27_causal_warm_f1cbeebd"
OUT = "runs/paper_ablations"
ARMS = ("r1", "r2", "r3", "r4", "r6")


def arm_config(arm: str, seed: int):
    base = replace(config_for(CANDIDATE), seed=seed)
    no_causal = replace(base, causal_eps_schedule=(), causal_max_iters=0, epochs=1300)
    return {
        "r1": no_causal,
        "r2": replace(base, warm_start=False),
        "r4": base,
        "r6": replace(no_causal, shared_network=True),
        "r3": replace(base, n_windows=1, n_collocation=39793, collocation="lhs", warm_start=False,
                      causal_max_iters=10000, epochs=40000, lbfgs_iters=5000, lr_decay_every=5000,
                      checkpoint_every=2000),
    }[arm]


NAMES = {"r1": "r1_no_causal", "r2": "r2_no_warm", "r3": "r3_causal_single",
         "r4": "r4_candidate", "r6": "r6_sequential"}


def run_arm(arm: str, seed: int, out: str = OUT):
    name = f"{NAMES[arm]}_seed{seed}"
    cfg = replace(arm_config(arm, seed), results_dir=f"{out}/{name}", ckpt_dir=f"{out}/{name}/history",
                  runs_dir=out)
    print(f"\n=== {name} -> {cfg.results_path} ===", flush=True)
    return run_one(cfg, resume=True)


def collect(out: str = OUT) -> pd.DataFrame:
    """ablations.csv is rebuilt from every finished run_summary.csv, never appended to."""
    root = _REPO / out
    rows = []
    for summary in sorted(root.glob("*_seed*/run_summary.csv")):
        row = pd.read_csv(summary).iloc[0]
        name = summary.parent.name
        rows.append({"arm": name, "group": name.rsplit("_seed", 1)[0], **row.to_dict()})
    table = pd.DataFrame(rows)
    if rows:
        pending = root / f"ablations.csv.{os.getpid()}.tmp"
        table.to_csv(pending, index=False)
        pending.replace(root / "ablations.csv")
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="r1,r2,r4,r6", help=f"comma list from {ARMS}")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", default=OUT, help="output root (repo-relative or absolute)")
    ap.add_argument("--collect", action="store_true", help="only rebuild ablations.csv")
    args = ap.parse_args()
    if not args.collect:
        arms = args.arms.split(",")
        if bad := set(arms) - set(ARMS):
            sys.exit(f"unknown arms {sorted(bad)}; choose from {ARMS}")
        for arm in arms:
            for seed in (int(s) for s in args.seeds.split(",")):
                run_arm(arm, seed, args.out)
    table = collect(args.out)
    if len(table):
        print(table[["arm", "device", "adam_steps", "lbfgs_evals", "rmse_combined_l2",
                     "wall_clock_s"]].to_string(index=False))


if __name__ == "__main__":
    main()
