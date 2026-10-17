"""Training-phase figures: what the optimiser did, epoch by epoch."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def lbfgs_spans(n: int, adam_iters: int, eps_marks=(), window_marks=(), phases=()):
    """(start, adam_end, end) per window; one window with adam_iters when the run is not windowed."""
    ends = list(window_marks) or [n]
    out, start = [], 0
    for end in ends:
        inside = [it for it, _ in eps_marks if start < it <= end]
        if phases:
            adam_end = next((i for i in range(start, end) if phases[i] == "lbfgs"), end)
        else:
            adam_end = max(inside) if inside else (adam_iters if len(ends) == 1 else end)
        out.append((start, adam_end, end))
        start = end
    return out


def fig_loss_phases(loss: np.ndarray, adam_iters: int, eps_marks=(), window_marks=(), phases=()):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(loss, lw=0.6, color="0.6", label="per iteration")
    ax.semilogy(pd.Series(loss).rolling(101, center=True, min_periods=1).median(), "k", lw=1.2,
                label="rolling median")
    for i, (a, adam_end, b) in enumerate(lbfgs_spans(len(loss), adam_iters, eps_marks,
                                                     window_marks, phases)):
        ax.axvspan(a, adam_end, color="tab:blue", alpha=0.06, label="Adam" if i == 0 else None)
        if adam_end < b:
            ax.axvspan(adam_end, b, color="tab:orange", alpha=0.15, label="L-BFGS" if i == 0 else None)
        if i:
            ax.axvline(a, color="0.3", lw=0.5)
    if len(eps_marks) <= 10:
        for it, eps in eps_marks:
            ax.axvline(it, color="tab:red", lw=0.8, ls="--")
            ax.annotate(f"eps={eps:g}", (it, loss.max()), fontsize=7, rotation=90, va="top")
    ax.set_xlabel("Adam steps and L-BFGS evaluations"); ax.set_ylabel("logged objective")
    ax.legend(fontsize=8)
    ax.set_title("Adam uses causal weights; L-BFGS uses raw residual. Compare within a phase only."
                 if eps_marks else "Logged optimization objective by optimizer phase", fontsize=9)
    return fig


def fig_gradient_stability(epochs: np.ndarray, grads: np.ndarray, window_marks=()):
    g = grads / np.maximum(np.linalg.norm(grads, axis=1, keepdims=True), 1e-30)
    cos = (g[1:] * g[:-1]).sum(1)
    if len(window_marks):
        win = np.searchsorted(np.asarray(window_marks), epochs, side="right")
        cos = np.where(win[1:] == win[:-1], cos, np.nan)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs[1:], cos, "tab:blue", lw=1, label="cos(g_k, g_k-1)")
    ax.set_ylim(-1.05, 1.05); ax.set_ylabel("cosine"); ax.set_xlabel("epoch")
    ax2 = ax.twinx()
    ax2.semilogy(epochs, np.maximum(np.linalg.norm(grads, axis=1), 1e-30), "0.4", lw=0.8, label="|g|")
    ax2.set_ylabel("gradient norm")
    ax.set_title("Snapshot gradient direction and norm within each window", fontsize=9)
    fig.legend(loc="lower left", fontsize=8)
    return fig


def fig_gradient_histograms(epochs: np.ndarray, grads: np.ndarray, layer_sizes: list[tuple[str, int]]):
    picks = np.unique(np.linspace(0, len(epochs) - 1, 4).round().astype(int))
    fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 3.5), sharey=True, squeeze=False)
    for ax, k in zip(axes[0], picks):
        start = 0
        for name, size in layer_sizes:
            block = grads[k, start:start + size]
            start += size
            active = np.abs(block[np.isfinite(block) & (block != 0)])
            if active.size:
                ax.hist(np.log10(active), bins=40, histtype="step", label=name)
        ax.set_title(f"epoch {epochs[k]}"); ax.set_xlabel("log10 |grad|")
    axes[0][0].legend(fontsize=6)
    fig.suptitle("Nonzero gradient magnitudes in the active window", fontsize=9)
    return fig


def ntk_eigenvalues(model, t: torch.Tensor, n_sub: int = 256) -> np.ndarray:
    """Eigenvalues of J J^T for the residual on a subsample of points."""
    from ..pinn import residual_of

    idx = torch.linspace(0, len(t) - 1, min(n_sub, len(t))).long()
    tt = t[idx].detach().clone().requires_grad_(True)
    r = residual_of(model, tt).reshape(-1)
    params = [p for p in model.parameters() if p.requires_grad]
    rows = []
    for i in range(len(r)):
        g = torch.autograd.grad(r[i], params, retain_graph=True, allow_unused=True)
        rows.append(torch.cat([(gi if gi is not None else torch.zeros_like(p)).reshape(-1)
                               for gi, p in zip(g, params)]))
    J = torch.stack(rows)
    return torch.linalg.eigvalsh(J @ J.T).flip(0).clamp_min(0).detach().cpu().numpy()


def fig_ntk_spectrum(spectra: dict[int, np.ndarray]):
    fig, ax = plt.subplots(figsize=(7, 4))
    for epoch, ev in spectra.items():
        ax.semilogy(np.maximum(ev, 1e-20), lw=1, label=f"epoch {epoch}")
    ax.set_xlabel("eigenvalue index"); ax.set_ylabel("NTK eigenvalue"); ax.legend(fontsize=8)
    ax.set_title("Residual NTK eigenvalues at sampled model states", fontsize=9)
    return fig


def fig_causal_weights(epochs, W):
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in np.unique(np.linspace(0, len(epochs) - 1, 6).round().astype(int)):
        w = W[k][~np.isnan(W[k])]
        ax.plot(np.linspace(0, 1, len(w)), w, lw=1, label=f"epoch {epochs[k]}")
    ax.set_xlabel("position in window (0 = start, 1 = end)"); ax.set_ylabel("temporal weight w")
    ax.legend(fontsize=7)
    ax.set_title("Causal weights within the active window at sampled states", fontsize=9)
    return fig


def causal_index(eps_marks, window_marks):
    """Map global iteration marks to Adam-only (causal) iteration numbers by dropping L-BFGS evals."""
    ends = list(window_marks) or [max((it for it, _ in eps_marks), default=0)]
    out, offset, start = [], 0, 0
    for end in ends:
        inside = [(it, e) for it, e in eps_marks if start < it <= end]
        out += [(it - offset, e) for it, e in inside]
        adam_end = max((it for it, _ in inside), default=end)
        offset += end - adam_end
        start = end
    return out


def fig_min_w(min_w, delta, eps_marks, window_marks=()):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(min_w, lw=0.8)
    ax.axhline(delta, color="k", ls="--", lw=0.8, label=f"delta={delta}")
    for it, _ in causal_index(eps_marks, window_marks):
        ax.axvline(it, color="tab:red", lw=0.6)
    ax.set_xlabel("causal iteration (Adam steps only)"); ax.set_ylabel("min w"); ax.legend(fontsize=8)
    ax.set_title("Minimum temporal weight by Adam step; red lines mark stage ends", fontsize=9)
    return fig


def fig_window_grid(loss, window_marks):
    edges = [0] + list(window_marks)
    n = len(edges) - 1
    cols = min(n, 5)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.4 * rows), squeeze=False)
    for k in range(n):
        ax = axes[k // cols][k % cols]
        ax.semilogy(loss[edges[k]:edges[k + 1]], lw=0.7)
        ax.set_title(f"window {k}", fontsize=8)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("Logged optimization objective within each window", fontsize=9)
    fig.tight_layout()
    return fig
