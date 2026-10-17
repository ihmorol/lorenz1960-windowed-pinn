"""Evaluation-phase figures: where and how the trained curve is wrong."""
import matplotlib.pyplot as plt
import numpy as np


def fig_error_vs_t(t, pred, ref, residual, joints=()):
    err = np.linalg.norm(pred - ref, axis=1)
    res = np.linalg.norm(residual, axis=1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(t, np.maximum(err, 1e-16), "tab:blue", lw=1, label="|u - ref|")
    ax.semilogy(t, np.maximum(res, 1e-16), "tab:red", lw=0.8, label="|residual|")
    for j in joints:
        ax.axvline(j, color="0.7", lw=0.5)
    ax.set_xlabel("t"); ax.legend(fontsize=8)
    ax.set_title("State error and ODE residual along time", fontsize=9)
    return fig


def fig_error_growth(t, pred, ref):
    err = np.linalg.norm(pred - ref, axis=1)
    m = (t > t[0] + 0.05 * (t[-1] - t[0])) & (err > 0)
    fig, ax = plt.subplots(figsize=(6, 4))
    if m.sum() > 2:
        p = np.polyfit(np.log(t[m]), np.log(err[m]), 1)
        ax.loglog(t[m], err[m], ".", ms=2, label="|error|")
        ax.loglog(t[m], np.exp(np.polyval(p, np.log(t[m]))), "k--", label=f"fit: t^{p[0]:.2f}")
    ax.set_xlabel("t"); ax.set_ylabel("|u - ref|"); ax.legend(fontsize=8)
    ax.set_title("Descriptive log-log fit over the displayed interval", fontsize=9)
    return fig


def fig_precision_floor(loss32, loss64):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(loss32, lw=0.7, label="float32")
    ax.semilogy(loss64, lw=0.7, label="float64")
    ax.set_xlabel("iteration"); ax.set_ylabel("loss"); ax.legend(fontsize=8)
    ax.set_title("Logged losses for matched float32 and float64 configurations", fontsize=9)
    return fig


def fig_joint_continuity(model, edges, h=1e-4):
    import torch
    from ..pinn import residual_parts

    p = next(model.parameters())
    jumps, slopes, left_res, right_res = [], [], [], []
    for k, e in enumerate(edges[1:-1]):
        with torch.no_grad():
            te = torch.tensor([[e - h], [e], [e + h]], dtype=p.dtype, device=p.device)
            a, b = model.windows[k](te), model.windows[k + 1](te)
        jumps.append(float((b[1] - a[1]).norm()))
        slopes.append(float(((b[2] - b[1]) / h - (a[1] - a[0]) / h).norm()))
        left_res.append(float(residual_parts(model.windows[k], te[1:2].clone().requires_grad_(True)).r.norm()))
        right_res.append(float(residual_parts(model.windows[k + 1], te[1:2].clone().requires_grad_(True)).r.norm()))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.semilogy(edges[1:-1], np.maximum(jumps, 1e-16), "o-", ms=3, label="|u(t+) - u(t-)|")
    ax.semilogy(edges[1:-1], np.maximum(slopes, 1e-16), "s-", ms=3, label="slope mismatch")
    ax.semilogy(edges[1:-1], np.maximum(left_res, 1e-16), ".-", ms=3, label="left ODE residual")
    ax.semilogy(edges[1:-1], np.maximum(right_res, 1e-16), ".-", ms=3, label="right ODE residual")
    ax.set_xlabel("window joint t"); ax.legend(fontsize=8)
    ax.set_title("Value and finite-difference slope mismatch at window joints", fontsize=9)
    return fig
