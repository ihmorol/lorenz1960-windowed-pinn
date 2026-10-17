"""Paper result tables from the run records: markdown (<out>/PAPER_TABLES.md) or --latex.

Sources, all repo-relative: <out>/ablations.csv (rebuilt from every run_summary.csv),
<out>/physics_checks.csv, and the two runs of record (F1 full-interval failure,
candidate seed 0). Paired differences use only seeds present in both arms.

Usage:
    python scripts/paper_tables.py [--out runs/paper_ablations] [--latex]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
F1 = REPO / "runs" / "4x60_f64_unit"
CANDIDATE = REPO / "runs" / "causal-window" / "candidate-1536pt" / "4x60_f64_unit_win27_causal_warm_f1cbeebd"

ARMS = {  # group -> (what changes vs its comparison run, comparison group)
    "r4_candidate": ("candidate config (27 windows, causal, warm start), other seeds", None),
    "r1_no_causal": ("causal weighting off; 1,300 Adam/window (35,100 total)", "r4_candidate"),
    "r2_no_warm": ("warm start off", "r4_candidate"),
    "r6_sequential": ("r1 with one network shared by all windows", "r1_no_causal"),
    "r3_causal_single": ("F1 + causal schedule, 4 x 10,000 Adam cap", "F1"),
}


def fmt(x, spec=".3g") -> str:
    return "--" if x is None or pd.isna(x) else format(x, spec)


def runs_of_record() -> pd.DataFrame:
    rows = []
    for group, path in (("F1", F1), ("r4_candidate", CANDIDATE)):
        if (path / "run_summary.csv").exists():
            row = pd.read_csv(path / "run_summary.csv").iloc[0].to_dict()
            if pd.isna(row.get("adam_steps")):   # F1 predates the column; plain Adam ran exactly `epochs`
                row["adam_steps"] = row["epochs"]
            rows.append({"arm": f"{group}_seed0 (run of record)", "group": group, **row})
    return pd.DataFrame(rows)


def tables(out: Path) -> dict[str, tuple[list[str], list[list[str]]]]:
    abl = pd.read_csv(out / "ablations.csv") if (out / "ablations.csv").exists() else pd.DataFrame()
    allruns = pd.concat([runs_of_record(), abl], ignore_index=True)
    by = {g: d for g, d in allruns.groupby("group")} if len(allruns) else {}

    per_run = []
    for _, r in allruns.sort_values(["group", "seed"]).iterrows():
        per_run.append([r["arm"], fmt(r["seed"], ".0f"), r["device"], fmt(r["n_params"], ",.0f"),
                        f"{fmt(r['n_collocation'], ',.0f')} {r['collocation']}",
                        fmt(r["adam_steps"], ",.0f"), fmt(r["lbfgs_evals"], ",.0f"),
                        fmt(r.get("capped_stages")), fmt(r["rmse_combined_l2"]),
                        fmt(r["max_abs_error_combined_l2"]), fmt(r["wall_clock_s"], ".0f")])

    summary, paired = [], []
    for group, (change, partner) in ARMS.items():
        d = by.get(group)
        if d is None:
            summary.append([group, change, "0 (PENDING)", "--", "--", "--"])
            continue
        rmse = d["rmse_combined_l2"]
        summary.append([group, change, str(len(d)),
                        f"{fmt(rmse.mean())} +/- {fmt(rmse.std(ddof=1))}" if len(d) > 1 else f"{fmt(rmse.iloc[0])} (n=1)",
                        f"{fmt(rmse.min())} .. {fmt(rmse.max())}", fmt(d["adam_steps"].mean(), ",.0f")])
        if partner and partner in by:
            mine = d.set_index("seed")["rmse_combined_l2"]
            theirs = by[partner].set_index("seed")["rmse_combined_l2"]
            seeds = sorted(set(mine.index) & set(theirs.index))
            ratios = [mine[s] / theirs[s] for s in seeds]
            paired.append([f"{group} / {partner}", ", ".join(f"{s:.0f}" for s in seeds) or "none",
                           ", ".join(fmt(x) for x in ratios) or "--",
                           f"{sum(x > 1 for x in ratios)}/{len(ratios)}" if ratios else "--"])

    physics = []
    if (out / "physics_checks.csv").exists():
        for _, r in pd.read_csv(out / "physics_checks.csv").iterrows():
            physics.append([r["run"], fmt(r["invariant_drift_prediction"]), fmt(r["invariant_drift_reference"]),
                            fmt(r["closure_prediction"]), fmt(r["closure_reference"])])

    return {
        "T1. Per-run results (actual optimizer work; RMSE and max error are combined L2)": (
            ["run", "seed", "device", "params", "points", "Adam steps", "L-BFGS evals", "capped stages",
             "RMSE", "max error", "wall s"], per_run),
        "T2. Arm summary (seed 0 runs of record included where they belong to the arm)": (
            ["arm", "what changes", "n", "RMSE mean +/- std", "RMSE range", "mean Adam steps"], summary),
        "T3. Paired seeds: RMSE ratio arm / comparison (>1 means the arm is worse)": (
            ["pair", "common seeds", "ratios", "arm worse in"], paired),
        "T4. R7 reference-free physics (E = k^4 x^2 + l^4 y^2 drift; closure ||u(T) - u(0)||)": (
            ["run", "E drift pred", "E drift ref", "closure pred", "closure ref"], physics),
    }


def markdown(out: Path, ts: dict) -> str:
    env = out / "environment.jsonl"
    lines = ["# Paper tables", "",
             f"Generated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC} from `{out.relative_to(REPO)}`.",
             "Wall clock is per run; runs on different devices or under parallel jobs are not",
             "timing-comparable. Seed 0 of r4_candidate is the candidate run of record.", ""]
    if env.exists():
        for line in env.read_text().splitlines():
            e = json.loads(line)
            lines.append(f"- suite run {e['started_utc']}: commit {e['git_commit'][:10]}"
                         f"{' (dirty)' if e['git_dirty'] else ''}, gpu {e['gpu']}, jobs {e['jobs']}, "
                         f"torch {e['packages'].get('torch')}")
        lines.append("")
    for title, (header, rows) in ts.items():
        lines += [f"## {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        lines += ["| " + " | ".join(r) + " |" for r in rows] + [""]
    return "\n".join(lines)


def latex(ts: dict) -> str:
    def esc(x: str) -> str:
        return x.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%") \
                .replace("_", r"\_").replace("#", r"\#").replace("^", r"\^{}")
    out = []
    for title, (header, rows) in ts.items():
        out += [f"% {title}", r"\begin{table}[htbp]", r"\centering",
                r"\begin{tabular}{" + "l" * len(header) + "}", r"\toprule",
                " & ".join(map(esc, header)) + r" \\", r"\midrule",
                *(" & ".join(map(esc, r)) + r" \\" for r in rows),
                r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/paper_ablations")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()
    out = Path(args.out) if Path(args.out).is_absolute() else REPO / args.out
    ts = tables(out)
    if args.latex:
        print(latex(ts))
        return
    target = out / "PAPER_TABLES.md"
    target.write_text(markdown(out, ts), encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
