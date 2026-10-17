"""One HTML page linking every figure, page, film and table of the given runs."""
from pathlib import Path

PHASES = {
    "Comparison": ["compare_*.png", "figures/precision_floor.png"],
    "Film": ["film/final.mp4"],
    "Trajectory in 3-D": ["figures/trajectory.html", "figures/phase_portraits.png",
                          "figures/solution_vs_reference.png"],
    "Training dynamics": ["figures/loss_phases.png", "figures/training_dynamics.png",
                          "figures/window_grid.png", "figures/min_w.png", "figures/causal_weights.png",
                          "figures/point_convergence.png", "figures/residual_evolution.png",
                          "figures/residual_profiles.png", "figures/residual_surface.html"],
    "Gradient descent": ["figures/loss_landscape.html", "figures/loss_landscape.png",
                         "figures/weight_path_pca3.html", "figures/gradient_stability.png",
                         "figures/gradient_histograms.png", "figures/layer_grad_norms.png",
                         "figures/gradient_diagnostics.png", "figures/ntk_spectrum.png"],
    "Evaluation": ["figures/error_vs_t.png", "figures/error_growth.png", "figures/error_surface.html",
                   "figures/error_analysis.png", "figures/physics_residual.png",
                   "figures/joint_continuity.png", "figures/invariant_drift.png",
                   "figures/metrics_summary.png", "figures/collocation_points.png", "results.png"],
    "Tables": ["run_summary.csv", "metrics.csv", "point_summary.csv", "history/loss_history.csv",
               "history/marks.csv", "history/causal.csv", "history/stages.csv", "history/config.json"],
}


def _cell(root: Path, base: Path, pattern: str) -> str:
    hits = sorted(base.glob(pattern))
    if not hits:
        return "<td class=none>-</td>"
    links = []
    for h in hits:
        rel = h.relative_to(root).as_posix()
        thumb = f'<br><a href="{rel}"><img src="{rel}"></a>' if h.suffix == ".png" else ""
        links.append(f'<a href="{rel}">{h.name}</a>{thumb}')
    return "<td>" + "<br>".join(links) + "</td>"


def write_index(runs_dir: Path | str, run_names: list[str]) -> Path:
    root = Path(runs_dir)
    cols = ["(shared)"] + run_names
    rows = []
    for phase, patterns in PHASES.items():
        rows.append(f'<tr class=phase><th colspan={len(cols) + 1}>{phase}</th></tr>')
        for pat in patterns:
            shared = pat.startswith("compare_") or pat.startswith("figures/precision")
            cells = [_cell(root, root, pat) if shared else "<td class=none></td>"]
            cells += ["<td class=none></td>" if shared else _cell(root, root / r, pat) for r in run_names]
            rows.append(f"<tr><td class=name>{pat}</td>{''.join(cells)}</tr>")
    head = "".join(f"<th>{c}</th>" for c in cols)
    html = f"""<!doctype html><meta charset=utf-8><title>runs index</title>
<style>body{{font:14px sans-serif;margin:20px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;
padding:4px 8px;vertical-align:top}}tr.phase th{{background:#eee;text-align:left;font-size:16px}}
td.name{{white-space:nowrap;color:#555}}td.none{{color:#bbb;text-align:center}}img{{max-width:260px}}</style>
<h1>runs</h1><table><tr><th></th>{head}</tr>{''.join(rows)}</table>"""
    path = root / "index.html"
    path.write_text(html, encoding="utf-8")
    return path
