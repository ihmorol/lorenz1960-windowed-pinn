"""Rotatable 3-D views from a run's per-point snapshots (breakdown/epoch_*.csv)."""
from pathlib import Path

import numpy as np
import pandas as pd

from ..history import residual_grid
from .figures import write_residual_surface_html

N_FRAMES = 60
N_POINTS = 300


def load_frames(run: Path) -> list[pd.DataFrame]:
    files = sorted((run / "breakdown").glob("epoch_*.csv"))
    pick = np.unique(np.linspace(0, len(files) - 1, N_FRAMES).round().astype(int))
    out = []
    for i in pick:
        f = pd.read_csv(files[i]).sort_values("t").dropna(subset=["x", "y", "z"])
        out.append(f.iloc[:: max(1, len(f) // N_POINTS)])
    return out


def trajectory_html(frames: list[pd.DataFrame], path: Path, name: str,
                    ref: pd.DataFrame | None = None) -> Path:
    import plotly.graph_objects as go

    ref = frames[0] if ref is None else ref
    epochs = [int(f.epoch.iloc[0]) for f in frames]

    def traces(k: int) -> list:
        f = frames[k]
        err = np.log10(np.maximum(f.err_norm, 1e-12))
        return [
            go.Scatter3d(x=ref.ref_x, y=ref.ref_y, z=ref.ref_z, mode="lines",
                         line={"color": "black", "width": 6}, name="reference"),
            go.Scatter3d(x=f.x, y=f.y, z=f.z, mode="lines+markers", name=f"{name} u(t)",
                         marker={"size": 3, "color": err, "colorscale": "Inferno_r", "cmin": -8,
                                 "cmax": 0, "colorbar": {"title": "log10 |err|"}},
                         line={"color": "#1f77b4", "width": 3},
                         text=[f"t={t:.3f}<br>|err|={e:.2e}" for t, e in zip(f.t, f.err_norm)],
                         hoverinfo="text+name"),
            go.Scatter3d(x=f.n_x, y=f.n_y, z=f.n_z, mode="lines", name="raw N(t)",
                         line={"color": "#1f77b4", "width": 2, "dash": "dash"}, opacity=0.5),
        ]

    last = len(epochs) - 1
    fig = go.Figure(data=traces(last),
                    frames=[go.Frame(data=traces(k), name=str(e)) for k, e in enumerate(epochs)])
    fig.update_layout(
        title="Predicted trajectory over training (drag to rotate, slider = epoch)",
        scene={"xaxis_title": "x", "yaxis_title": "y", "zaxis_title": "z", "aspectmode": "cube"},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        updatemenus=[{"type": "buttons", "showactive": False, "y": 0, "x": 0, "buttons": [
            {"label": "Play", "method": "animate",
             "args": [None, {"frame": {"duration": 150}, "fromcurrent": True}]},
            {"label": "Pause", "method": "animate",
             "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]}]}],
        sliders=[{"currentvalue": {"prefix": "epoch "}, "active": last,
                  "steps": [{"label": str(e), "method": "animate",
                             "args": [[str(e)], {"frame": {"duration": 0}, "mode": "immediate"}]}
                            for e in epochs]}])
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


def write_all(run: Path, name: str = "") -> list[Path]:
    run = Path(run)
    out = run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    frames = load_frames(run)
    first = sorted((run / "breakdown").glob("epoch_*.csv"))[0]
    ref = pd.read_csv(first).sort_values("t")
    ref = ref.iloc[:: max(1, len(ref) // N_POINTS)]
    written = [trajectory_html(frames, out / "trajectory.html", name or run.name, ref)]
    for column, sqrt, tag, zlabel in (("err_norm", False, "error", "log10 |u - ref|"),
                                      ("r_sq", True, "residual", "log10 |r|")):
        e, t, v = residual_grid(run / "breakdown", column=column, sqrt=sqrt)
        html = write_residual_surface_html(e, t, v, out / f"{tag}_surface.html", name or run.name,
                                           title=f"{tag.capitalize()} over (t, epoch)", zlabel=zlabel)
        if html is not None:
            written.append(html)
    return written
