"""Loss surface in the plane of the optimiser's own path (Li et al. 2018), plus the weight path."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .figures import save_figure

GRID = 41
MARGIN = 0.25


def loss_at(model, grid, theta) -> float:
    from ..pinn import pinn_loss

    p = next(model.parameters())
    torch.nn.utils.vector_to_parameters(torch.as_tensor(theta, dtype=p.dtype, device=p.device),
                                        model.parameters())
    return pinn_loss(model, grid.clone().requires_grad_(True)).item()


def loss_on_plane(model, grid, centre, d1, d2, a, b) -> np.ndarray:
    return np.array([[loss_at(model, grid, centre + aa * d1 + bb * d2) for aa in a] for bb in b])


def fig_landscape(a, b, logZ, proj, log_path, epochs, var2, label):
    A, B = np.meshgrid(a, b)
    fig = plt.figure(figsize=(14, 5.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(A, B, logZ, cmap="viridis", alpha=0.75, linewidth=0)
    ax.plot(proj[:, 0], proj[:, 1], log_path, "r.-", lw=1.5, ms=3, label="parameter snapshots")
    ax.scatter(proj[0, 0], proj[0, 1], log_path[0], c="w", edgecolors="k", s=60, label="start")
    ax.scatter(0, 0, log_path[-1], c="k", s=60, label="end")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("log10 loss"); ax.legend(fontsize=8)
    ax = fig.add_subplot(1, 2, 2)
    cs = ax.contourf(A, B, logZ, levels=30, cmap="viridis")
    plt.colorbar(cs, ax=ax, label="log10 loss")
    ax.scatter(proj[:, 0], proj[:, 1], c=epochs, cmap="hot", s=8, zorder=3)
    ax.plot(proj[:, 0], proj[:, 1], "r-", lw=0.6, zorder=2)
    for k in np.linspace(0, len(epochs) - 1, 6).astype(int):
        ax.annotate(str(epochs[k]), proj[k, :2], fontsize=7, color="w")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.suptitle(f"{label} | plane explains {100 * var2:.0f}% of snapshot variance; "
                 "heights use the final-anchor global residual", fontsize=9)
    fig.tight_layout()
    return fig


def fig_layer_grad_norms(diag: pd.DataFrame):
    smooth = lambda v: pd.Series(v).rolling(25, center=True, min_periods=1).median()
    fig, ax = plt.subplots(figsize=(9, 4))
    for c in [c for c in diag.columns if c.startswith("layer_") and c.endswith("_grad_norm")]:
        ax.semilogy(diag.epoch, smooth(diag[c]), lw=1, label=c.replace("_grad_norm", ""))
    ax.semilogy(diag.epoch, diag.grad_norm, color="0.8", lw=0.5, label="total (raw)")
    ax.semilogy(diag.epoch, smooth(diag.grad_norm), "k-", lw=1.4, label="total (rolling median)")
    ax.set_xlabel("epoch"); ax.set_ylabel("gradient norm"); ax.legend(ncol=3, fontsize=8)
    ax.set_title("Sampled gradient norms by layer", fontsize=9)
    return fig


def write_all(run: Path) -> list[Path]:
    import plotly.graph_objects as go

    from ..sweep import config_for
    from ..train import load_run, make_grid

    run = Path(run)
    out = run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    cfg = config_for(run)
    trail = np.load(run / "history" / "param_trail.npz")
    epochs, P = trail["epochs"], trail["params"].astype(np.float64)
    diag = pd.read_csv(run / "history" / "training_diagnostics.csv")

    centre = P[-1]
    X = P - centre
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S**2 / max((S**2).sum(), 1e-300)
    Vt = np.vstack([Vt, np.zeros((max(0, 3 - len(Vt)), Vt.shape[1]))])
    var = np.append(var, np.zeros(max(0, 3 - len(var))))
    proj = X @ Vt[:3].T
    span = np.maximum(proj[:, :2].max(0) - proj[:, :2].min(0), 1e-6)
    lo, hi = proj[:, :2].min(0) - MARGIN * span, proj[:, :2].max(0) + MARGIN * span
    a, b = np.linspace(lo[0], hi[0], GRID), np.linspace(lo[1], hi[1], GRID)

    model, _, _ = load_run(cfg)
    grid = make_grid(cfg, next(model.parameters()).device)
    logZ = np.log10(np.maximum(loss_on_plane(model, grid, centre, Vt[0], Vt[1], a, b), 1e-300))
    losses = np.array([loss_at(model, grid, theta) for theta in P])
    log_path = np.log10(np.maximum(losses, 1e-300))
    np.savez(out / "loss_landscape.npz", a=a, b=b, logZ=logZ, proj=proj, log_path=log_path, epochs=epochs)

    written = save_figure(fig_landscape(a, b, logZ, proj, log_path, epochs, var[:2].sum(), cfg.label),
                          out, "loss_landscape", ("png",))
    hover = [f"epoch {e}<br>loss {l:.2e}" for e, l in zip(epochs, losses)]
    fig = go.Figure([
        go.Surface(x=a, y=b, z=logZ, colorscale="Viridis", opacity=0.85, colorbar={"title": "log10 loss"}),
        go.Scatter3d(x=proj[:, 0], y=proj[:, 1], z=log_path, mode="lines+markers",
                     line={"color": "red", "width": 4}, marker={"size": 3, "color": epochs, "colorscale": "Hot"},
                     text=hover, hoverinfo="text", name="parameter snapshots")])
    fig.update_layout(title=f"Final-anchor global residual plane ({100 * var[:2].sum():.0f}% of snapshot variance)",
                      scene={"xaxis_title": "PC1", "yaxis_title": "PC2", "zaxis_title": "log10 loss"},
                      margin={"l": 0, "r": 0, "t": 40, "b": 0})
    fig.write_html(str(out / "loss_landscape.html"), include_plotlyjs="cdn")
    written.append(out / "loss_landscape.html")

    gn = np.interp(epochs, diag.epoch, diag.grad_norm)
    fig = go.Figure(go.Scatter3d(x=proj[:, 0], y=proj[:, 1], z=proj[:, 2], mode="lines+markers",
                                 line={"color": "gray", "width": 2},
                                 marker={"size": 3 + 12 * gn / max(gn.max(), 1e-30), "color": epochs,
                                         "colorscale": "Viridis", "colorbar": {"title": "epoch"}},
                                 text=[f"epoch {e}<br>loss {l:.2e}<br>|grad| {g:.2e}"
                                       for e, l, g in zip(epochs, losses, gn)], hoverinfo="text"))
    fig.update_layout(title=f"Weight-space path, top-3 PCA directions ({100 * var[:3].sum():.0f}% of variance); "
                            "marker size = gradient norm",
                      scene={"xaxis_title": "PC1", "yaxis_title": "PC2", "zaxis_title": "PC3"},
                      margin={"l": 0, "r": 0, "t": 40, "b": 0})
    fig.write_html(str(out / "weight_path_pca3.html"), include_plotlyjs="cdn")
    written.append(out / "weight_path_pca3.html")
    written += save_figure(fig_layer_grad_norms(diag), out, "layer_grad_norms", ("png",))
    return written
