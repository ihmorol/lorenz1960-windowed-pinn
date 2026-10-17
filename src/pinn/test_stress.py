"""Stress tests for the causal-window PINN pipeline.

Each test pins one stage of the pipeline diagram (setup -> windowed PINN ->
ODE residual -> causal loss -> Adam/L-BFGS -> handoff -> evaluation) to
independent ground truth: brute-force recomputation, finite differences,
an exactly solvable special case, or a hand-computed invariant. Tests that
only check internal consistency live in test_pinn.py; everything here
re-derives the expected number from outside the pipeline.
"""
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

from pinn.config import (
    Config,
    compute_error_metrics,
    lorenz1960_coefficients,
    reference_at,
    reference_trajectory,
)
from pinn.functions.collocation import latin_hypercube_points
from pinn.functions.losses import causal_loss, causal_weights
from pinn.functions.optimizers import adam_with_decay, run_lbfgs
from pinn.functions.physics import lorenz1960_rhs
from pinn.functions.reference import solve_reference
from pinn.pinn import PINN, build_model, loss_terms, residual_parts
from pinn.train import make_grid, predict, residual_at, set_seed, train
from pinn.viz.figures import invariant_series


# ---------------------------------------------------------------------------
# Stage 0: the problem itself -- coefficients, RHS, and the reference solver
# ---------------------------------------------------------------------------
def test_coefficient_formula_matches_diagram_values():
    """k=2, l=1 must give the (-0.1, 1.6, -0.75) printed on the pipeline diagram."""
    c = lorenz1960_coefficients(2.0, 1.0)
    assert np.allclose(c, (-0.1, 1.6, -0.75), atol=1e-12)


def test_torch_rhs_matches_hand_written_numpy_rhs():
    rng = np.random.default_rng(0)
    u = torch.tensor(rng.normal(size=(64, 3)), dtype=torch.float64)
    torch_rhs = lorenz1960_rhs(u, lorenz1960_coefficients(2.0, 1.0)).numpy()
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    numpy_rhs = np.stack([-0.1 * y * z, 1.6 * x * z, -0.75 * x * y], axis=1)
    assert np.allclose(torch_rhs, numpy_rhs, atol=1e-14)


def test_reference_solution_satisfies_the_ode_finitely():
    """The DOP853 reference must satisfy du/dt = f(u) by central differences."""
    cfg = Config(t_span=(0.0, 5.0))
    t, ys = reference_trajectory(cfg, n=20001)
    h = t[1] - t[0]
    dudt = (ys[2:] - ys[:-2]) / (2 * h)
    c = lorenz1960_coefficients(cfg.k, cfg.l)
    x, y, z = ys[1:-1, 0], ys[1:-1, 1], ys[1:-1, 2]
    f = np.stack([c[0] * y * z, c[1] * x * z, c[2] * x * y], axis=1)
    assert np.abs(dudt - f).max() < 1e-5


def test_reference_conserves_hand_invariants():
    """16x^2 + y^2 and 15x^2 - 2z^2 are conserved by (-0.1, 1.6, -0.75):
    d/dt = 2xyz(a*alpha + b*beta + c*gamma), which vanishes for both pairs."""
    cfg = Config(t_span=(0.0, 13.26446))
    _, ys = reference_trajectory(cfg, n=13265)
    for inv in (16 * ys[:, 0] ** 2 + ys[:, 1] ** 2, 15 * ys[:, 0] ** 2 - 2 * ys[:, 2] ** 2):
        assert (np.abs(inv - inv[0]) / abs(inv[0])).max() < 1e-9


def test_reference_conserves_null_space_invariants():
    cfg = Config(t_span=(0.0, 13.26446))
    _, ys = reference_trajectory(cfg, n=13265)
    _, drift = invariant_series(ys, cfg.coefficients)
    assert drift.shape[1] == 2 and drift.max() < 1e-9


def test_reference_uses_the_configured_k_and_l():
    """For k != 2 the reference the pipeline evaluates against must match an
    independent integration with the same coefficients. Today it silently
    integrates the default k=2, l=1 system instead."""
    cfg = Config(k=3.0, l=1.2, t_span=(0.0, 1.0))
    c = lorenz1960_coefficients(3.0, 1.2)
    sol = solve_ivp(
        lambda t, u: [c[0] * u[1] * u[2], c[1] * u[0] * u[2], c[2] * u[0] * u[1]],
        cfg.t_span, cfg.initial_state, method="DOP853",
        rtol=1e-10, atol=1e-12, t_eval=np.linspace(0.0, 1.0, 101))
    _, ys = reference_trajectory(cfg, n=101)
    assert np.abs(ys - sol.y.T).max() < 1e-8


def test_generic_reference_path_is_coefficient_correct():
    """The Problem-based path (functions/reference.solve_reference) closes over
    the configured coefficients and must match an independent integration."""
    cfg = Config(k=3.0, l=1.2, t_span=(0.0, 1.0))
    c = lorenz1960_coefficients(3.0, 1.2)
    sol = solve_ivp(
        lambda t, u: [c[0] * u[1] * u[2], c[1] * u[0] * u[2], c[2] * u[0] * u[1]],
        cfg.t_span, cfg.initial_state, method="DOP853",
        rtol=1e-10, atol=1e-12, t_eval=np.linspace(0.0, 1.0, 101))
    generic = solve_reference(replace(cfg.spec, reference=None), np.linspace(0.0, 1.0, 101))
    assert np.abs(generic - sol.y.T).max() < 1e-8


def test_run_record_is_portable_across_machines(tmp_path):
    """A committed config.json written on another machine carries that machine's
    absolute output paths; ensure_record must compare only run settings, so a
    run of record can be resumed anywhere (Colab suite depends on this)."""
    import json

    cfg = Config(t_span=(0.0, 1.0), epochs=10)
    ckpt = tmp_path / "history"
    ckpt.mkdir()
    record = cfg.record()
    record["results_dir"] = "/foreign/machine/results"
    record["ckpt_dir"] = "/foreign/machine/history"
    record["runs_dir"] = "/foreign/machine/runs"
    (ckpt / "config.json").write_text(json.dumps(record))
    replace(cfg, ckpt_dir=str(ckpt)).ensure_record()

    changed = replace(cfg, ckpt_dir=str(ckpt), epochs=11)
    try:
        changed.ensure_record()
        raise SystemError("a genuinely different configuration must be rejected")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Stages 1-3: trial solution, autograd derivative, residual assembly
# ---------------------------------------------------------------------------
def _zero_model(cfg):
    torch.manual_seed(0)
    model = PINN(cfg).to(dtype=cfg.torch_dtype)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    return model


def _built(cfg):
    """build_model with the dtype the run path applies (train/load_run call .to)."""
    return build_model(cfg).to(dtype=cfg.torch_dtype)


def test_pipeline_is_exact_on_an_equilibrium_solution():
    """u0=(0.6,0,0) solves the ODE with u(t)=u0; a zero-weight PINN with hard IC
    represents it exactly, so trial + autograd + RHS + residual must be zero."""
    cfg = Config(initial_state=(0.6, 0.0, 0.0), dtype="float64", depth=2, width=8)
    t = torch.linspace(0, 1, 401, dtype=torch.float64).reshape(-1, 1)
    for form in ("span", "unit"):
        model = _zero_model(replace(cfg, ic_scale=form))
        parts = residual_parts(model, t.clone().requires_grad_(True))
        assert torch.allclose(parts.u.detach(), model.u0.expand_as(parts.u.detach())), form
        assert parts.r.detach().abs().max() < 1e-12, form


def test_windowed_pipeline_exact_on_equilibrium_through_all_windows():
    """Same exactness check through the windowed model, handoff included."""
    cfg = Config(initial_state=(0.6, 0.0, 0.0), dtype="float64", depth=2, width=8,
                 t_span=(0.0, 1.5), n_windows=3, ic_scale="unit")
    model = _built(cfg)
    with torch.no_grad():
        for w in model.windows:
            for p in w.parameters():
                p.zero_()
            w.u0.copy_(torch.tensor([[0.6, 0.0, 0.0]], dtype=torch.float64))
    t = torch.linspace(0, 1.5, 601, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    parts = residual_parts(model, t)
    assert parts.r.detach().abs().max() < 1e-12
    assert torch.allclose(parts.u.detach(), torch.tensor([0.6, 0.0, 0.0], dtype=torch.float64)
                          .expand_as(parts.u.detach()))


def test_autograd_derivative_matches_central_differences():
    from pinn.functions.derivative import time_derivative
    cfg = Config(dtype="float64", depth=2, width=8, seed=3)
    torch.manual_seed(3)
    model = PINN(cfg).to(dtype=cfg.torch_dtype)
    t = torch.linspace(0.01, 0.99, 101, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    u = model.trial(t, model.net(t))
    dudt = time_derivative(u, t)
    h = 1e-6
    tt = t.detach()
    fd = (model.trial(tt + h, model.net(tt + h)) - model.trial(tt - h, model.net(tt - h))) / (2 * h)
    assert (dudt.detach() - fd.detach()).abs().max() < 1e-6


def test_windowed_residual_matches_per_window_recomputation():
    """_windowed_parts must agree with evaluating each window on its own points."""
    cfg = Config(dtype="float64", depth=2, width=8, t_span=(0.0, 1.5), n_windows=3)
    torch.manual_seed(1)
    model = _built(cfg)
    with torch.no_grad():
        model.set_window_start(1, model.windows[0](torch.tensor([[0.5]], dtype=torch.float64))[0])
        model.set_window_start(2, model.windows[1](torch.tensor([[1.0]], dtype=torch.float64))[0])
    t = torch.linspace(0, 1.5, 301, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    parts = residual_parts(model, t)
    for k, w in enumerate(model.windows):
        m = model.window_of(t.detach()) == k
        solo = residual_parts(w, t.detach()[m].requires_grad_(True))
        assert torch.allclose(parts.u.detach()[m], solo.u.detach(), atol=1e-12)
        assert torch.allclose(parts.r.detach()[m], solo.r.detach(), atol=1e-12)


def test_windows_do_not_leak_into_each_other():
    """A point inside window 0 must be unaffected by window 1's parameters."""
    cfg = Config(dtype="float64", depth=2, width=8, t_span=(0.0, 1.0), n_windows=2)
    torch.manual_seed(1)
    model = _built(cfg)
    with torch.no_grad():
        model.set_window_start(1, model.windows[0](torch.tensor([[0.5]], dtype=torch.float64))[0])
    t = torch.tensor([[0.25]], dtype=torch.float64)
    r_before = residual_parts(model, t.clone().requires_grad_(True)).r.detach()
    with torch.no_grad():
        for p in model.windows[1].parameters():
            p.add_(10.0)
    r_after = residual_parts(model, t.clone().requires_grad_(True)).r.detach()
    assert torch.equal(r_before, r_after)


# ---------------------------------------------------------------------------
# Stage 4: causal loss (Wang, Sankaran & Perdikaris 2024, eq. 3.5)
# ---------------------------------------------------------------------------
def test_causal_weights_brute_force():
    rng = np.random.default_rng(7)
    L = torch.tensor(rng.uniform(0.01, 2.0, size=32), dtype=torch.float64)
    Lnp = L.numpy()
    for eps in (1e-3, 0.05, 1.0, 25.0):
        prefix = np.array([Lnp[:i].sum() for i in range(len(L))])
        expected = np.exp(-eps * prefix)
        w = causal_weights(L, eps)
        assert torch.allclose(w, torch.tensor(expected), atol=1e-12), eps
    assert causal_weights(L, 1.0)[0] == 1.0
    w = causal_weights(L, 1.0)
    assert torch.all(w[1:] <= w[:-1]) and not w.requires_grad


def test_causal_loss_brute_force_and_sorting():
    rng = np.random.default_rng(11)
    r = torch.tensor(rng.normal(size=(50, 3)), dtype=torch.float64)
    t = torch.tensor(rng.uniform(0, 1, size=(50, 1)), dtype=torch.float64)
    eps = 0.4
    loss, w = causal_loss(r, t, eps)
    order = np.argsort(t.reshape(-1))
    L = (r.numpy()[order] ** 2).mean(axis=1)
    prefix = np.array([L[:i].sum() for i in range(len(L))])
    expected_w = np.exp(-eps * prefix)
    assert np.allclose(w.numpy(), expected_w, atol=1e-12)
    assert loss.item() == pytest.approx(float((expected_w * L).mean()), rel=1e-12)


def test_causal_loss_is_scale_consistent_with_plain_residual_mse():
    """As eps -> 0 every weight is 1, so the causal loss must equal the plain
    mean squared residual the pipeline reports as raw_res."""
    rng = np.random.default_rng(5)
    r = torch.tensor(rng.normal(size=(40, 3)), dtype=torch.float64)
    t = torch.tensor(rng.uniform(0, 1, size=(40, 1)), dtype=torch.float64)
    loss, _ = causal_loss(r, t, eps=1e-12)
    assert loss.item() == pytest.approx(float(r.pow(2).mean()), rel=1e-6)


# ---------------------------------------------------------------------------
# Stages 5-7 + handoff: the causal-window training loop end to end
# ---------------------------------------------------------------------------
def _tiny_causal_cfg(base_dir, **over):
    base = dict(t_span=(0.0, 0.3), n_windows=2, causal_eps_schedule=(1e-2, 1e-1),
                causal_delta=0.99, causal_max_iters=40, depth=1, width=8,
                n_collocation=24, collocation="uniform", lbfgs_iters=0,
                log_every=10, eval_every=10, print_every=0, dtype="float64",
                ckpt_dir=str(base_dir / "history"), results_dir=str(base_dir / "results"))
    base.update(over)
    return Config(**base)


def _window_points(cfg, model, k):
    """Rebuild the exact point set train_windows uses for window k."""
    grid = make_grid(cfg, next(model.parameters()).device)
    pts = grid[model.window_of(grid) == k]
    if k + 1 < cfg.n_windows:
        pts = torch.cat((pts, torch.tensor([[model.edges[k + 1]]],
                                            dtype=grid.dtype, device=grid.device)))
    return pts


def test_stage_certified_min_w_matches_saved_model_state(tmp_path):
    """With L-BFGS off, each window's final parameters are the ones its LAST eps
    stage certified (earlier stages are superseded by later training within the
    window); recomputing min w from them must reproduce min_w_final."""
    cfg = _tiny_causal_cfg(tmp_path)
    model, history = train(cfg)
    last_stage_of = {}
    for s in history.stage_status:
        last_stage_of[s["window"]] = s
    for s in last_stage_of.values():
        w = model.windows[s["window"]]
        pts = _window_points(cfg, model, s["window"])
        post = residual_parts(w, pts.clone().requires_grad_(True))
        _, post_w = causal_loss(post.r, pts, s["eps"])
        assert post_w.min().item() == pytest.approx(s["min_w_final"], rel=1e-8)


def test_stage_status_threshold_semantics(tmp_path):
    cfg = _tiny_causal_cfg(tmp_path)
    _, history = train(cfg)
    assert len(history.stage_status) == cfg.n_windows * len(cfg.causal_eps_schedule)
    for s in history.stage_status:
        assert s["threshold_met"] == (s["min_w_final"] > cfg.causal_delta)
        assert 0 < s["adam_steps"] <= cfg.causal_max_iters


def test_handoff_state_is_the_previous_window_edge_value(tmp_path):
    cfg = _tiny_causal_cfg(tmp_path)
    model, _ = train(cfg)
    for k in range(cfg.n_windows - 1):
        edge = torch.tensor([[model.edges[k + 1]]], dtype=torch.float64)
        assert torch.allclose(model.windows[k](edge), model.windows[k + 1].u0, atol=1e-12)
        # C^0 continuity at the shared edge is exact by construction (g(a)=0).
        assert torch.allclose(model.windows[k](edge), model.windows[k + 1](edge), atol=1e-12)


def test_shared_edge_routing_and_endpoint_append():
    """bucketize hands the shared edge to the NEXT window, so train_windows must
    append it to the outgoing window's point set for both to enforce it."""
    cfg = Config(t_span=(0.0, 1.0), n_windows=3, n_collocation=31, collocation="uniform",
                 dtype="float64", depth=1, width=8)
    grid = make_grid(cfg, torch.device("cpu"))
    model = build_model(cfg)
    counts = torch.bincount(model.window_of(grid), minlength=cfg.n_windows)
    for k in range(cfg.n_windows - 1):
        edge = model.edges[k + 1]
        assert model.window_of(torch.tensor([[edge]], dtype=torch.float64)).item() == k + 1
        pts = grid[model.window_of(grid) == k]
        assert not torch.any(pts == edge)          # absent from the bucket...
        assert _window_points(cfg, model, k).shape[0] == counts[k].item() + 1  # ...appended


def test_windowed_causal_training_reduces_residual(tmp_path):
    cfg = _tiny_causal_cfg(tmp_path, causal_max_iters=60, eval_every=20)
    model, history = train(cfg)
    assert history.loss[-1] < history.loss[0]
    assert len(history.ref_mse) > 0 and all(np.isfinite(history.ref_mse))
    assert len(history.window_marks) == cfg.n_windows


def test_endpoint_enters_the_loss_and_gradients_stay_local(tmp_path):
    cfg = Config(t_span=(0.0, 1.0), n_windows=2, dtype="float64", depth=2, width=8,
                 n_collocation=24, collocation="uniform")
    torch.manual_seed(2)
    model = _built(cfg)
    with torch.no_grad():
        model.set_window_start(1, model.windows[0](torch.tensor([[model.edges[1]]], dtype=torch.float64))[0])
    grid = make_grid(cfg, torch.device("cpu"))
    bucket = grid[model.window_of(grid) == 0]
    endpoint = torch.tensor([[model.edges[1]]], dtype=grid.dtype)
    raw_with, _, _ = loss_terms(model.windows[0], torch.cat((bucket, endpoint)).requires_grad_(True))
    raw_without, _, _ = loss_terms(model.windows[0], bucket.clone().requires_grad_(True))
    assert raw_with.item() != raw_without.item()      # the endpoint contributes
    model.zero_grad(set_to_none=True)
    loss_terms(model.windows[0], torch.cat((bucket, endpoint)).clone().requires_grad_(True))[0].backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.windows[0].parameters())
    assert all(p.grad is None for p in model.windows[1].parameters())


# ---------------------------------------------------------------------------
# Evaluation stage: routing, coverage, metrics, determinism
# ---------------------------------------------------------------------------
def test_predict_matches_residual_parts_and_routes_by_window(tmp_path):
    cfg = Config(t_span=(0.0, 1.5), n_windows=3, dtype="float64", depth=1, width=8)
    torch.manual_seed(0)
    model = _built(cfg)
    with torch.no_grad():
        model.set_window_start(1, model.windows[0](torch.tensor([[0.5]], dtype=torch.float64))[0])
        model.set_window_start(2, model.windows[1](torch.tensor([[1.0]], dtype=torch.float64))[0])
    t = np.linspace(0.0, 1.5, 151)
    u = predict(model, t)
    parts = residual_parts(model, torch.tensor(t.reshape(-1, 1), dtype=torch.float64).requires_grad_(True))
    assert np.allclose(u, parts.u.detach().numpy(), atol=1e-12)


def test_coverage_stays_strictly_below_the_shared_edge():
    """train_windows evaluates against the reference only up to the largest
    float strictly below the shared edge (float64 candidate geometry)."""
    cfg = Config(t_span=(0.0, 13.26446), n_windows=27, dtype="float64")
    model = build_model(cfg)
    for k in range(cfg.n_windows - 1):
        edge = np.asarray(model.edges[k + 1], dtype=np.float64)
        coverage = float(np.nextafter(edge, np.asarray(-np.inf, dtype=np.float64)))
        assert coverage < edge
        assert model.window_of(torch.tensor([[coverage]], dtype=torch.float64)).item() == k


def test_metrics_are_hand_computable():
    ref = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    pred = ref + np.array([[0.1, -0.2, 0.0], [0.3, 0.0, -0.6]])
    m = compute_error_metrics(ref, pred).set_index("state")
    assert m.loc["x", "mae"] == pytest.approx(0.2)
    assert m.loc["x", "rmse"] == pytest.approx(np.sqrt((0.01 + 0.09) / 2))
    assert m.loc["z", "max_abs_error"] == pytest.approx(0.6)
    l2 = np.linalg.norm(pred - ref, axis=1)
    assert m.loc["combined_l2", "mae"] == pytest.approx(l2.mean())


def test_reference_at_handles_unsorted_duplicate_and_out_of_range_times():
    cfg = Config(t_span=(0.0, 2.0))
    ys = reference_at(cfg, np.array([1.5, 0.5, 1.5, 0.5, 9.9]))
    solo = reference_at(cfg, np.array([0.5, 1.5, 2.0]))
    assert ys.shape == (5, 3)
    assert np.allclose(ys[0], solo[1]) and np.allclose(ys[2], solo[1])   # t = 1.5
    assert np.allclose(ys[1], solo[0]) and np.allclose(ys[3], solo[0])   # t = 0.5
    assert np.allclose(ys[4], solo[2])      # 9.9 clipped to tf = 2.0


def test_same_seed_reproduces_the_loss_trace(tmp_path):
    _, h1 = train(_tiny_causal_cfg(tmp_path / "a"))
    _, h2 = train(_tiny_causal_cfg(tmp_path / "b"))
    assert h1.loss == h2.loss
    assert h1.min_w == h2.min_w


def test_lbfgs_polish_reduces_the_unweighted_loss(tmp_path):
    cfg = Config(depth=1, width=8, epochs=300, n_collocation=64, lbfgs_iters=60,
                 log_every=100, eval_every=100, print_every=0,
                 ckpt_dir=str(tmp_path / "history"), results_dir=str(tmp_path / "results"))
    set_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    model = build_model(cfg)
    grid = make_grid(cfg, torch.device("cpu"))
    adam, sched = adam_with_decay(model.parameters(), cfg)
    for _ in range(cfg.epochs):
        adam.zero_grad()
        res, ic, _ = loss_terms(model, grid.clone().requires_grad_(True))
        (res + ic).backward()
        adam.step()
        sched.step()
    before = res.item()

    def closure():
        r, i, _ = loss_terms(model, grid.clone().requires_grad_(True))
        return r + i

    run_lbfgs(model.parameters(), closure, cfg.lbfgs_iters, cfg.torch_dtype)
    after = loss_terms(model, grid.clone().requires_grad_(True))[0].item()
    assert after < before


def test_final_loss_column_is_the_collocation_residual_mse(tmp_path):
    from pinn.train import save_results
    cfg = _tiny_causal_cfg(tmp_path, causal_max_iters=10)
    model, history = train(cfg)
    save_results(model, history, cfg)
    import pandas as pd
    row = pd.read_csv(cfg.results_path / "run_summary.csv").iloc[0]
    grid = make_grid(cfg, torch.device("cpu"))
    dense = residual_at(model, grid.reshape(-1).numpy())
    assert row["final_loss"] == pytest.approx(float(np.mean(dense ** 2)), rel=1e-8)


# ---------------------------------------------------------------------------
# Config guards and samplers
# ---------------------------------------------------------------------------
def test_config_rejects_impossible_combinations():
    with pytest.raises(ValueError):
        Config(n_windows=3, end_state=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        Config(causal_eps_schedule=(0.1,), causal_delta=1.0)
    with pytest.raises(ValueError):
        Config(ic_scale="bogus")
    with pytest.raises(ValueError):
        Config(points_per_unit=-1.0)
    with pytest.raises(ValueError):
        Config(n_windows=5, n_collocation=4)


def test_lhs_sampler_is_seed_deterministic():
    a = latin_hypercube_points((0.0, 1.0), 100, seed=0)
    b = latin_hypercube_points((0.0, 1.0), 100, seed=0)
    c = latin_hypercube_points((0.0, 1.0), 100, seed=1)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a.min() >= 0.0 and a.max() <= 1.0
