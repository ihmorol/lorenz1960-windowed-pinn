"""Audit and summarise runs/paper_ablations (source of docs/ABLATION_RESULTS.md).

    python scripts/ablation_report.py

Statistics pool the 14 Colab-suite runs in runs/paper_ablations plus r4 seed 0,
the paper's run of record (identical config, trained earlier). F1 is printed
alongside as reference and never pooled.
A run counts as complete when every window has Adam and L-BFGS entries in
loss_history.csv, the iteration index has no gaps or NaN, and run_summary.csv
exists.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

RUNS = Path(__file__).resolve().parents[1] / "runs"
ABL = RUNS / "paper_ablations"
RECORDS = {"r4_candidate_seed0": RUNS / "causal-window/candidate-1536pt/4x60_f64_unit_win27_causal_warm_f1cbeebd",
           "F1_seed0": RUNS / "4x60_f64_unit"}
METRICS = ["rmse_combined_l2", "max_abs_error_combined_l2", "residual", "invariant_drift_prediction"]


def complete(run: Path) -> bool:
    lh, summary = run / "history/loss_history.csv", run / "run_summary.csv"
    if not (lh.exists() and summary.exists()):
        return False
    L, s = pd.read_csv(lh), pd.read_csv(summary)
    n_windows = int(s["n_windows"].fillna(1).iloc[0]) if "n_windows" in s else 1
    windows = L["window"].nunique() if "window" in L else 1
    with_lbfgs = L.loc[L.phase == "lbfgs", "window"].nunique() if "phase" in L else 1
    return (windows == with_lbfgs == n_windows and not L.loss.isna().any()
            and (np.diff(L.iteration) == 1).all())


def load() -> pd.DataFrame:
    runs = {d.name: d for d in sorted(ABL.glob("r*_seed*"))} | RECORDS
    rows = []
    for name, d in runs.items():
        ok = complete(d)
        print(f"{'ok        ' if ok else 'INCOMPLETE'} {name}")
        if ok:
            s = pd.read_csv(d / "run_summary.csv").iloc[0].to_dict()
            rows.append(s | {"arm": name, "group": name.rsplit("_seed", 1)[0],
                             "seed": int(name.rsplit("seed", 1)[1])})
    df = pd.DataFrame(rows)
    pc = pd.read_csv(ABL / "physics_checks.csv").rename(columns={"run": "arm"})
    pc["arm"] = pc["arm"].replace({RECORDS["r4_candidate_seed0"].name: "r4_candidate_seed0"})
    df = df.merge(pc[["arm", "invariant_drift_prediction", "closure_prediction"]], on="arm", how="left")
    df["residual"] = df["final_residual_mse_full"].fillna(df["final_loss"])
    return df


def main() -> None:
    df = load()
    records, df = df[df.arm == "F1_seed0"], df[df.arm != "F1_seed0"]
    ref = df[df.group == "r4_candidate"]
    print()
    for group, d in df.groupby("group", sort=False):
        r = d.rmse_combined_l2
        line = (f"{group:18s} n={len(d)} adam={d.adam_steps.mean():9,.0f} "
                f"rmse mean={r.mean():.3g} std={r.std():.2g} median={r.median():.3g} "
                f"range={r.min():.3g}..{r.max():.3g} below1e-4={(r < 1e-4).sum()}/{len(r)}")
        if group != "r4_candidate" and len(d) > 1:
            p = {m[:8]: mannwhitneyu(d[m], ref[m], alternative="greater", method="exact").pvalue for m in METRICS}
            line += " | MWU p vs r4: " + ", ".join(f"{k}={v:.3f}" for k, v in p.items())
        print(line)
    print("\nreference only, not pooled:")
    for _, r in records.iterrows():
        print(f"  {r.arm:18s} rmse={r.rmse_combined_l2:.3g} max={r.max_abs_error_combined_l2:.3g}")


if __name__ == "__main__":
    main()
