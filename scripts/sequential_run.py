"""Archived sequential-training prototype.

This incomplete historical runner does not match the saved R6 configuration.
Use verify_archive.py to evaluate the final R6 checkpoints.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from pinn.config import Config, compute_error_metrics, reference_trajectory  # noqa: E402
from pinn.functions.optimizers import adam_with_decay  # noqa: E402
from pinn.train import set_seed  # noqa: E402


def build_net(cfg: Config) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(1, cfg.width), nn.Tanh()]
    for _ in range(cfg.depth - 1):
        layers += [nn.Linear(cfg.width, cfg.width), nn.Tanh()]
    layers.append(nn.Linear(cfg.width, 3))
    for m in layers:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return nn.Sequential(*layers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=27)
    ap.add_argument("--epochs-per-window", type=int, default=1300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = Config(t_span=(0.0, 13.26446), dtype="float64", ic_scale="unit")
    spans = np.linspace(*base.t_span, args.windows + 1)
    pts = np.linspace(*base.t_span, args.windows * 1536 + 1)
    out = _REPO / "runs" / "paper_ablations" / "sequential_single_net"
    out.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    net = build_net(base)
    anchors = [torch.tensor([list(base.initial_state)], dtype=torch.float64)]
    started = time.perf_counter()
    for k in range(args.windows):
        m = (pts >= spans[k]) & (pts <= spans[k + 1])
        t = torch.tensor(pts[m].reshape(-1, 1), dtype=torch.float64, requires_grad=True)
        u0 = anchors[-1].clone()
        opt, _ = adam_with_decay(net.parameters(), replace(base, epochs=args.epochs_per_window))
        for _ in range(args.epochs_per_window):
            opt.zero_grad()
            u = u0 + (t - spans[k]) * net(t)
            dudt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
            x, y, z = u[:, 0], u[:, 1], u[:, 2]
            f = torch.stack([-0.1 * y * z, 1.6 * x * z, -0.75 * x * y], dim=1)
            loss = (dudt - f).pow(2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            end_t = torch.tensor([[spans[k + 1]]], dtype=torch.float64)
            anchors.append(u0 + (end_t - spans[k]) * net(end_t))
        print(f"[window {k + 1}/{args.windows}] loss {loss.item():.3e} "
              f"({time.perf_counter() - started:.0f}s)", flush=True)

    t_eval = np.linspace(*base.t_span, 13265)
    te = torch.tensor(t_eval.reshape(-1, 1), dtype=torch.float64)
    pred = torch.empty_like(te)
    for k in range(args.windows):
        m = (t_eval >= spans[k]) & (t_eval <= spans[k + 1])
        tk = te[m]
        with torch.no_grad():
            pred[m] = anchors[k] + (tk - spans[k]) * net(tk)
    _, ys_ref = reference_trajectory(base, n=13265)
    table = compute_error_metrics(ys_ref, pred.numpy())

    np.savez(out / "prediction.npz", t=t_eval, ys=pred.numpy(), ys_ref=ys_ref)
    torch.save(net.state_dict(), out / "net.pt")
    (out / "config.json").write_text(json.dumps(
        {"windows": args.windows, "epochs_per_window": args.epochs_per_window,
         "seed": args.seed, "t_span": list(base.t_span),
         "initial_state": list(base.initial_state)}, indent=2))
    table.to_csv(out / "error_metrics.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
