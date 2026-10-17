"""From-scratch PyTorch PINN for the Lorenz-1960 ODE system."""
from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import torch
from torch import Tensor, nn

from .config import Config
from .functions.derivative import time_derivative
from .functions.losses import mean_squared_residual
from .functions.physics import lorenz1960_rhs, residual
from .functions.trial import hard_initial_condition
from .functions.windows import split_windows

_ACT = {"tanh": nn.Tanh, "relu": nn.ReLU, "sigmoid": nn.Sigmoid, "gelu": nn.GELU, "swish": nn.SiLU}


class ResidualParts(NamedTuple):
    """Every per-collocation-point quantity computed inside one residual evaluation.

    These are the intermediates the training step already builds; exposing them
    costs nothing and lets :mod:`pinn.history` snapshot the full state of the
    collocation batch without a second forward/backward pass.
    """

    n: Tensor       # raw network output N(t),          (Nc, 3)
    u: Tensor       # trial solution u_T(t),            (Nc, 3)
    dudt: Tensor    # autograd time derivative du_T/dt,  (Nc, 3)
    f: Tensor       # physics right-hand side f(u_T),   (Nc, 3)
    r: Tensor       # residual r = du_T/dt - f(u_T),    (Nc, 3)


class PINN(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        act = _ACT[cfg.activation]
        layers: list[nn.Module] = [nn.Linear(1, cfg.width), act()]
        for _ in range(cfg.depth - 1):
            layers += [nn.Linear(cfg.width, cfg.width), act()]
        layers.append(nn.Linear(cfg.width, 3))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.ic, self.ic_scale, self.gamma = cfg.ic, cfg.ic_scale, cfg.gamma
        self.problem = cfg.spec
        self.end_state = None if cfg.end_state is None else torch.tensor([cfg.end_state], dtype=cfg.torch_dtype)
        self.register_buffer("u0", torch.tensor([cfg.initial_state], dtype=cfg.torch_dtype))
        self.register_buffer("coeffs", torch.as_tensor(cfg.coefficients, dtype=cfg.torch_dtype).reshape(1, 3))
        self.t0, self.tf = float(cfg.t_span[0]), float(cfg.t_span[1])

    def rhs(self, u: Tensor) -> Tensor:
        return self.problem.rhs(u)

    def trial(self, t: Tensor, n: Tensor) -> Tensor:
        if self.ic == "hard":
            return hard_initial_condition(t, n, self.u0, self.t0, self.tf, self.ic_scale)
        return n

    def forward(self, t: Tensor) -> Tensor:
        return self.trial(t, self.net(t))


def ode_rhs(u: Tensor, coeffs) -> Tensor:
    return lorenz1960_rhs(u, coeffs)


def ode_residual(u: Tensor, dudt: Tensor, coeffs) -> Tensor:
    return dudt - lorenz1960_rhs(u, coeffs)


class WindowedPINN(nn.Module):
    """One PINN per time window; each starts from the previous window's end state."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        spans = split_windows(cfg.t_span, cfg.n_windows)
        self.edges = [a for a, _ in spans] + [spans[-1][1]]
        self.windows = nn.ModuleList(PINN(replace(cfg, t_span=s, n_windows=1)) for s in spans)
        self.ic, self.gamma, self.end_state = cfg.ic, cfg.gamma, None
        self.t0, self.tf = float(cfg.t_span[0]), float(cfg.t_span[1])

    @property
    def u0(self) -> Tensor:
        return self.windows[0].u0

    def window_of(self, t: Tensor) -> Tensor:
        inner = torch.as_tensor(self.edges[1:-1], dtype=t.dtype, device=t.device)
        return torch.bucketize(t.reshape(-1), inner, right=True)

    def set_window_start(self, k: int, state: Tensor) -> None:
        self.windows[k].u0.copy_(state.detach().reshape(1, -1))

    def forward(self, t: Tensor) -> Tensor:
        idx = self.window_of(t)
        out = torch.empty(t.shape[0], self.u0.shape[1], dtype=t.dtype, device=t.device)
        for k, w in enumerate(self.windows):
            m = idx == k
            if m.any():
                out[m] = w(t[m])
        return out


def build_model(cfg: Config) -> nn.Module:
    return WindowedPINN(cfg) if cfg.n_windows > 1 else PINN(cfg)


def _windowed_parts(model: WindowedPINN, t: Tensor) -> ResidualParts:
    idx = model.window_of(t)
    cols = [torch.empty(t.shape[0], 3, dtype=t.dtype, device=t.device) for _ in ResidualParts._fields]
    for k, w in enumerate(model.windows):
        m = idx == k
        if m.any():
            tk = t[m].detach().clone().requires_grad_(True)
            for col, val in zip(cols, residual_parts(w, tk)):
                col[m] = val
    return ResidualParts(*cols)


def residual_parts(model: PINN, t: Tensor) -> ResidualParts:
    if isinstance(model, WindowedPINN):
        return _windowed_parts(model, t)
    n = model.net(t)
    u = model.trial(t, n)
    dudt = time_derivative(u, t)
    return ResidualParts(n=n, u=u, dudt=dudt, f=model.rhs(u), r=residual(u, dudt, model.rhs))


def residual_of(model: PINN, t: Tensor) -> Tensor:
    return residual_parts(model, t).r


def loss_terms(model: PINN, t: Tensor) -> tuple[Tensor, Tensor, ResidualParts]:
    parts = residual_parts(model, t)
    res = mean_squared_residual(parts.r)
    if model.ic == "soft":
        t0 = torch.full((1, 1), model.t0, dtype=t.dtype, device=t.device)
        ic = (model(t0) - model.u0).pow(2).mean()
    else:
        ic = torch.zeros((), dtype=res.dtype, device=res.device)
    if model.end_state is not None:
        tf = torch.full((1, 1), model.tf, dtype=t.dtype, device=t.device)
        ic = ic + (model(tf) - model.end_state.to(t)).pow(2).mean()
    return res, ic, parts


def pinn_loss(model: PINN, t: Tensor) -> Tensor:
    res, ic, _ = loss_terms(model, t)
    return res + model.gamma * ic if (model.ic == "soft" or model.end_state is not None) else res
