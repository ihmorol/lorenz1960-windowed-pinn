import numpy as np
import pandas as pd
import pytest
import torch

from pinn.config import Config, reference_trajectory
from pinn.pinn import PINN, ode_residual


def test_hard_ic_exact():
    model = PINN(Config())
    out = model(torch.zeros(1, 1))
    expected = torch.tensor([Config().initial_state])
    assert torch.allclose(out, expected, atol=1e-6)


def test_residual_zero_on_truth():
    t, ys = reference_trajectory(Config(), n=1001)
    dudt = np.gradient(ys, t, axis=0)
    r = ode_residual(
        torch.tensor(ys, dtype=torch.float64),
        torch.tensor(dudt, dtype=torch.float64),
        Config().coefficients,
    )
    assert r[1:-1].abs().max().item() < 1e-2


def test_training_reduces_loss():
    from pinn.train import train

    cfg = Config(epochs=100, n_collocation=101, lbfgs_iters=50, seed=0)
    _, history = train(cfg)
    assert history.loss[-1] < 0.1 * history.loss[0]


def test_soft_ic_trains():
    from pinn.train import train

    cfg = Config(ic="soft", epochs=200, n_collocation=101, lbfgs_iters=50, seed=0)
    _, history = train(cfg)
    assert history.loss[-1] < 0.1 * history.loss[0]


def test_history_records_diagnostics():
    from pinn.train import train

    cfg = Config(epochs=50, n_collocation=64, lbfgs_iters=0, log_every=10, eval_every=25, seed=0)
    _, history = train(cfg)
    n = len(history.log_epoch)
    assert n == len(history.grad_norm) == len(history.residual_loss) == len(history.lr)
    assert all(len(v) == n for v in history.layer_grad_norms.values())
    assert history.adam_iters == len(history.loss) == cfg.epochs
    assert len(history.ref_epoch) == len(history.ref_mse) > 0
    assert all(g > 0 for g in history.grad_norm)


def test_figure_suite_writes_all_panels(tmp_path):
    from pinn import viz as figures
    from pinn.train import collect_artifacts, train

    cfg = Config(depth=1, width=8, epochs=30, n_collocation=64, lbfgs_iters=0,
                 log_every=10, eval_every=15, seed=0)
    model, history = train(cfg)
    written = figures.generate_all(collect_artifacts(model, history, cfg), tmp_path, ("png",))
    assert set(written) == {
        "training_dynamics", "gradient_diagnostics", "collocation_points",
        "solution_vs_reference", "error_analysis", "phase_portraits",
        "metrics_summary", "invariant_drift", "physics_residual",
    }
    assert all(p.exists() and p.stat().st_size > 0 for paths in written.values() for p in paths)


def test_run_report_writes_tables_and_figures(tmp_path):
    from pinn import viz as figures
    from pinn.train import collect_artifacts, train

    cfg = Config(depth=1, width=8, epochs=20, n_collocation=64, lbfgs_iters=0,
                 log_every=10, eval_every=10, seed=0)
    model, history = train(cfg)
    metrics = figures.write_run_report(collect_artifacts(model, history, cfg), tmp_path, ("png",))
    assert list(metrics["state"]) == ["x", "y", "z", "combined_l2"]
    for name in ("metrics.csv", "results.png", "loss_history.csv"):
        assert (tmp_path / name).exists()
    assert (tmp_path / "figures" / "training_dynamics.png").exists()


def test_history_round_trips_through_csv(tmp_path):
    from pinn import viz as figures
    from pinn.history import TrainHistory
    from pinn.train import collect_artifacts, train

    cfg = Config(depth=1, width=8, epochs=20, n_collocation=64, lbfgs_iters=0,
                 log_every=5, eval_every=10, seed=0)
    model, history = train(cfg)
    data = tmp_path / "data"
    figures.write_run_report(collect_artifacts(model, history, cfg), tmp_path, ("png",), data)

    assert not (tmp_path / "loss_history.csv").exists()  # bulk telemetry stays out of results/
    reloaded = TrainHistory.from_saved(data)
    assert reloaded.loss == history.loss
    assert reloaded.log_epoch == history.log_epoch
    assert reloaded.grad_norm == pytest.approx(history.grad_norm)
    assert reloaded.layer_grad_norms.keys() == history.layer_grad_norms.keys()
    assert reloaded.ref_mse == pytest.approx(history.ref_mse)


def test_sweep_figures(tmp_path):
    import itertools

    import pandas as pd

    from pinn import viz as figures

    rows = [
        {"depth": d, "width": w, "activation": a, "seed": s, "rmse": 10.0 ** -(d + s)}
        for d, w, a, s in itertools.product((1, 2), (20, 50), ("tanh", "relu"), (0, 1))
    ]
    written = figures.generate_sweep(pd.DataFrame(rows), tmp_path, "rmse", ("png",))
    assert set(written) == {"architecture_heatmap", "activation_comparison", "seed_robustness"}
    assert all(p.exists() for paths in written.values() for p in paths)


# --------------------------------------------------------------------------
# per-epoch collocation breakdown
# --------------------------------------------------------------------------
def test_snapshots_are_off_by_default():
    """The run of record must be unaffected: no snapshot files, no extra cost."""
    from pinn.train import train

    assert Config().snapshot_every == 0
    _, history = train(Config(depth=1, width=8, epochs=5, n_collocation=16,
                              log_every=5, eval_every=5, print_every=0))
    assert history.snapshots is None


def test_snapshot_rows_reconstruct_the_loss(tmp_path):
    """Every column is recorded, and loss_contribution sums to the reported loss."""
    from pinn.history import SNAPSHOT_COLUMNS, load_snapshots, point_history
    from pinn.train import train

    cfg = Config(depth=1, width=8, epochs=21, n_collocation=32, snapshot_every=10,
                 log_every=10, eval_every=10, print_every=0,
                 results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    _, history = train(cfg)

    frame = load_snapshots(tmp_path / "breakdown")
    assert list(frame.columns) == list(SNAPSHOT_COLUMNS)
    epochs = sorted(frame["epoch"].unique())
    assert epochs == [0, 10, 20, 21]
    assert len(frame) == len(epochs) * cfg.n_collocation

    # loss_contribution = |r_i|^2 / (3 Nc), so it sums to mean(r^2) over all components.
    for epoch, group in frame[frame.epoch < cfg.epochs].groupby("epoch"):
        expected = history.logged_loss[history.log_epoch.index(epoch)]
        assert group["loss_contribution"].sum() == pytest.approx(expected, rel=1e-5)

    # r = du/dt - f(u), column by column. The network trains in float32, so the
    # recorded residual agrees with a float64 reconstruction only to float32
    # precision -- the columns are consistent, not independently exact.
    for a in "xyz":
        assert frame[f"r_{a}"].to_numpy() == pytest.approx(
            (frame[f"d{a}_dt"] - frame[f"f_{a}"]).to_numpy(), abs=1e-6)

    one = point_history(tmp_path / "breakdown", 7)
    assert list(one["epoch"]) == epochs
    assert one["t"].nunique() == 1  # collocation points are fixed across training


def test_point_summary_and_residual_grid(tmp_path):
    from pinn.history import residual_grid
    from pinn.train import train, save_results

    cfg = Config(depth=1, width=8, epochs=31, n_collocation=32, snapshot_every=10,
                 log_every=10, eval_every=15, print_every=0,
                 results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    model, history = train(cfg)
    save_results(model, history, cfg)

    summary = pd.read_csv(tmp_path / "point_summary.csv")
    report = pd.read_csv(tmp_path / "run_summary.csv").iloc[0]
    assert len(summary) == cfg.n_collocation
    assert summary.loss_contribution_final.sum() == pytest.approx(report.final_residual_mse_full, rel=1e-5)
    assert (summary["r_max"] >= summary["r_final"]).all()
    assert summary["rank_by_final_residual"].nunique() == cfg.n_collocation

    epochs, centres, values = residual_grid(tmp_path / "breakdown", n_bins=12)
    assert values.shape == (len(epochs), 12) and centres.size == 12

    for name in ("residual_evolution", "residual_profiles", "point_convergence"):
        assert (tmp_path / "figures" / f"{name}.png").exists()


def test_run_summary_columns(tmp_path):
    from pinn.train import save_results, train

    cfg = Config(depth=2, width=8, epochs=20, n_collocation=32, log_every=10,
                 eval_every=10, print_every=0,
                 results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    model, history = train(cfg)
    save_results(model, history, cfg)

    summary = pd.read_csv(tmp_path / "run_summary.csv")
    assert len(summary) == 1
    for column in ("arch", "depth", "width", "n_params", "rmse_combined_l2", "r2_x",
                   "err_norm_p99", "final_x", "final_ref_x", "residual_mse_collocation",
                   "residual_mse_dense", "residual_generalisation_gap",
                   "invariant_1_max_drift", "wall_clock_s", "epochs_to_1e-04"):
        assert column in summary.columns, column
    assert summary["arch"].iloc[0] == "2x8"
    # The reference solution conserves the invariants to solver precision.
    assert summary["invariant_1_max_drift_reference"].iloc[0] < 1e-9


def test_sweep_writes_runs_and_comparison(tmp_path):
    from pinn import sweep as sweep_module

    base = Config(epochs=12, n_collocation=16, log_every=6, eval_every=6, print_every=0,
                  runs_dir=str(tmp_path))
    original, sweep_module.SNAPSHOT_EVERY = sweep_module.SNAPSHOT_EVERY, 6
    try:
        comparison = sweep_module.sweep(base, depths=(1,), widths=(4, 6))
    finally:
        sweep_module.SNAPSHOT_EVERY = original

    assert list(comparison["arch"]) == ["1x4", "1x6"]
    assert (tmp_path / "comparison.csv").exists()
    for arch in ("1x4", "1x6"):
        run = tmp_path / arch
        assert (run / "metrics.csv").exists()
        assert (run / "run_summary.csv").exists()
        assert (run / "point_summary.csv").exists()
        assert (run / "history" / "pinn.pt").exists()
        snapshots = {p.stem for p in (run / "breakdown").glob("epoch_*.csv")}
        assert snapshots == {"epoch_000000", "epoch_000006", "epoch_000011", "epoch_000012"}
    # A single-activation, single-seed sweep has nothing to say about either.
    written = {p.stem for p in (tmp_path / "figures").glob("*.png")}
    assert "architecture_heatmap" in written and "architecture_scatter" in written
    assert "activation_comparison" not in written and "seed_robustness" not in written


def test_sweep_leaves_the_run_of_record_alone(tmp_path):
    """The sweep must never write into src/pinn/results or src/pinn/history."""
    from pinn.sweep import sweep_config

    base = Config(runs_dir=str(tmp_path))
    cfg = sweep_config(base, 5, 70)
    assert cfg.results_path == tmp_path / "5x70"
    assert cfg.ckpt_path == tmp_path / "5x70" / "history"
    assert cfg.snapshot_every > 0
    assert Config().results_path not in (cfg.results_path, cfg.ckpt_path)


# --------------------------------------------------------------------------
# entry-point helpers
# --------------------------------------------------------------------------
def test_config_for_round_trips_a_finished_run(tmp_path):
    """A run's own summary is enough to rebuild the Config its weights need."""
    from pinn.sweep import config_for
    from pinn.train import save_results, train

    cfg = Config(depth=3, width=12, activation="gelu", epochs=10, n_collocation=16,
                 k=1.3, l=0.8, initial_state=(0.25, 0.75, 1.0),
                 log_every=5, eval_every=5, print_every=0,
                 results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    model, history = train(cfg)
    save_results(model, history, cfg)

    rebuilt = config_for(tmp_path)
    assert (rebuilt.depth, rebuilt.width, rebuilt.activation) == (3, 12, "gelu")
    assert rebuilt.ckpt_path == tmp_path / "history"
    assert (rebuilt.k, rebuilt.l, rebuilt.initial_state) == (1.3, 0.8, (0.25, 0.75, 1.0))


def test_load_run_reproduces_the_saved_prediction(tmp_path):
    from pinn.sweep import config_for
    from pinn.train import load_run, predict, save_results, train

    cfg = Config(depth=2, width=10, epochs=10, n_collocation=16, log_every=5,
                 eval_every=5, print_every=0,
                 results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    model, history = train(cfg)
    save_results(model, history, cfg)

    t = np.linspace(0.0, 1.0, 51)
    reloaded, _, reloaded_cfg = load_run(config_for(tmp_path))
    assert predict(reloaded, t) == pytest.approx(predict(model, t), abs=0)
    assert reloaded_cfg.depth == cfg.depth


def test_resume_skips_a_finished_run(tmp_path):
    """A completed run is reused, not retrained; an interrupted one is redone."""
    from pinn.sweep import run_one, sweep_config

    base = Config(epochs=8, n_collocation=16, log_every=4, eval_every=4,
                  print_every=0, runs_dir=str(tmp_path))
    cfg = sweep_config(base, 1, 6)
    first = run_one(cfg, resume=True)
    stamp = (cfg.results_path / "run_summary.csv").stat().st_mtime_ns

    second = run_one(cfg, resume=True)
    assert (cfg.results_path / "run_summary.csv").stat().st_mtime_ns == stamp
    assert second["rmse_combined_l2"].iloc[0] == first["rmse_combined_l2"].iloc[0]

    (cfg.results_path / "run_summary.csv").unlink()  # simulate an interrupted run
    assert run_one(cfg, resume=True)["arch"].iloc[0] == "1x6"


def test_hard_initial_condition_forms():
    import torch
    from pinn.functions.trial import hard_initial_condition

    u0 = torch.tensor([[0.5, 0.75, 1.0]])
    t = torch.tensor([[0.0], [5.0], [10.0]])
    n = torch.ones(3, 3)
    for form in ("span", "unit"):
        assert torch.allclose(hard_initial_condition(t, n, u0, 0.0, 10.0, form)[0], u0[0])
    assert torch.allclose(hard_initial_condition(t, n, u0, 0.0, 10.0, "span")[2], u0[0] + 1.0)
    assert torch.allclose(hard_initial_condition(t, n, u0, 0.0, 10.0, "unit")[2], u0[0] + 10.0)


def test_collocation_samplers_cover_the_span():
    from pinn.functions.collocation import latin_hypercube_points, uniform_points

    lhs = latin_hypercube_points((0.0, 2.0), 50, seed=0)
    uni = uniform_points((0.0, 2.0), 50)
    assert lhs.shape == uni.shape == (50, 1)
    assert 0.0 <= lhs.min() and lhs.max() <= 2.0
    assert uni[0, 0] == 0.0 and uni[-1, 0] == 2.0


def test_uniform_window_grid_has_1536_training_points_per_window():
    import torch
    from pinn.pinn import WindowedPINN
    from pinn.train import make_grid

    cfg = Config(t_span=(0.0, 13.26446), n_windows=27, n_collocation=27 * 1535 + 1,
                 collocation="uniform", dtype="float64", depth=1, width=8)
    grid = make_grid(cfg, torch.device("cpu"))
    counts = torch.bincount(WindowedPINN(cfg).window_of(grid), minlength=cfg.n_windows)
    assert counts.tolist() == [1535] * 26 + [1536]
    assert (counts[:-1] + 1 == 1536).all()  # training adds each outgoing endpoint


def test_density_knobs_scale_with_the_window():
    cfg = Config(t_span=(0.0, 4.0), points_per_unit=100, eval_per_unit=10)
    assert cfg.n_collocation == 400 and cfg.n_eval == 41
    assert Config().n_collocation == 3000 and Config().n_eval == 1001


def test_float64_trains_end_to_end():
    import torch
    from pinn.train import train

    cfg = Config(dtype="float64", depth=1, width=8, epochs=5, n_collocation=16, lbfgs_iters=3,
                 log_every=5, eval_every=5, print_every=0)
    model, history = train(cfg)
    assert next(model.parameters()).dtype == torch.float64
    assert cfg.arch == "1x8_f64" and len(history.loss) > 5


def test_run_of_record_config_is_unchanged():
    cfg = Config()
    assert (cfg.dtype, cfg.ic_scale, cfg.collocation, cfg.lr_decay, cfg.n_eval) == \
        ("float32", "span", "lhs", None, 1001)
    assert cfg.arch == "4x60"


def test_problem_registry_matches_locked_baseline():
    import numpy as np
    from dataclasses import replace
    from pinn.config import reference_trajectory
    from pinn.functions.reference import solve_reference

    cfg = Config(t_span=(0.0, 2.0))
    t, ys = reference_trajectory(cfg, n=201)
    generic = solve_reference(replace(cfg.spec, reference=None), t)
    assert cfg.spec.name == "lorenz1960" and cfg.spec.dim == 3
    assert np.abs(generic - ys).max() < 1e-8
    assert np.abs(solve_reference(cfg.spec, t) - ys).max() < 1e-8


def test_end_state_penalty_enters_the_loss():
    import torch
    from dataclasses import replace
    from pinn.pinn import PINN, pinn_loss

    cfg = Config(depth=1, width=8)
    t = torch.linspace(0, 1, 8).reshape(-1, 1)
    torch.manual_seed(0)
    plain = pinn_loss(PINN(cfg), t.clone().requires_grad_(True))
    torch.manual_seed(0)
    penalised = pinn_loss(PINN(replace(cfg, end_state=(9.0, 9.0, 9.0))), t.clone().requires_grad_(True))
    assert penalised > plain


def test_trails_follow_the_snapshots(tmp_path):
    import numpy as np
    from pinn.history import flat_params
    from pinn.pinn import PINN
    from pinn.train import set_seed, train

    cfg = Config(depth=1, width=8, epochs=21, n_collocation=32, snapshot_every=10,
                 log_every=10, eval_every=10, print_every=0,
                 results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    model, history = train(cfg)
    assert history.param_epochs == [0, 10, 20, 21]
    n = sum(p.numel() for p in model.parameters())
    assert np.stack(history.param_trail).shape == np.stack(history.grad_trail).shape == (4, n)
    set_seed(cfg.seed)
    assert np.allclose(history.param_trail[0], flat_params(PINN(cfg)).cpu().numpy())


def test_run_extras_are_written(tmp_path):
    from pinn import viz
    from pinn.train import main

    cfg = Config(depth=1, width=8, epochs=21, n_collocation=32, snapshot_every=10,
                 log_every=10, eval_every=10, print_every=0,
                 results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    main(cfg)
    names = {p.name for p in viz.generate_run_extras(tmp_path)}
    for expected in ("trajectory.html", "loss_landscape.png", "loss_phases.png",
                     "gradient_stability.png", "gradient_histograms.png", "ntk_spectrum.png",
                     "error_vs_t.png", "error_growth.png"):
        assert expected in names, expected


def test_adam_checkpoint_resumes(tmp_path, monkeypatch):
    from dataclasses import replace
    import pinn.train as train_module

    cfg = Config(depth=1, width=8, epochs=6, n_collocation=16, checkpoint_every=3,
                 log_every=3, eval_every=3, print_every=0,
                 results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    save = train_module._save_progress

    def interrupt(*args, **kwargs):
        save(*args, **kwargs)
        if kwargs.get("next_epoch") == 3:
            raise RuntimeError("interrupted after checkpoint")

    monkeypatch.setattr(train_module, "_save_progress", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        train_module.train(cfg)
    monkeypatch.setattr(train_module, "_save_progress", save)
    _, history = train_module.train(cfg)
    assert history.resumed_from == 3 and len(history.loss) == 6
    assert (tmp_path / "history" / "progress.pt").exists()
    with pytest.raises(ValueError, match="configuration differs"):
        train_module.train(replace(cfg, lr_start=2e-3))


def test_root_scripts_compile():
    import py_compile
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    commands = {p.name for p in root.glob("run_*.py")}
    assert {"run_pinn.py", "run_causal.py", "run_eval.py", "run_compare.py"} <= commands
    sources = list(root.glob("run_*.py")) + list((root / "scripts").glob("*.py")) + list((root / "src").rglob("*.py"))
    for source in sources:
        py_compile.compile(str(source), doraise=True)


def test_causal_weights_gate_later_times():
    import torch
    from pinn.functions.losses import causal_loss, causal_weights

    w = causal_weights(torch.tensor([1.0, 1.0, 0.0, 0.0]), eps=1.0)
    assert w[0] == 1.0 and torch.all(w[1:] <= w[:-1]) and not w.requires_grad
    assert torch.allclose(causal_weights(torch.zeros(4), eps=100.0), torch.ones(4))
    t = torch.tensor([[0.3], [0.1], [0.2]])
    loss, w_sorted = causal_loss(torch.ones(3, 3), t, eps=0.5)
    assert w_sorted.shape == (3,) and loss <= 1.0


def test_split_windows_tile_the_span():
    from pinn.functions.windows import split_windows

    assert split_windows((0.0, 2.0), 4) == [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
    assert split_windows((0.0, 2.0), 1) == [(0.0, 2.0)]


def test_windowed_pinn_routes_and_is_continuous():
    import torch
    from pinn.pinn import WindowedPINN, residual_parts

    cfg = Config(t_span=(0.0, 2.0), n_windows=4, depth=1, width=8)
    torch.manual_seed(0)
    m = WindowedPINN(cfg)
    assert len(m.windows) == 4 and cfg.arch == "1x8_win4"
    t = torch.tensor([[0.1], [0.6], [1.2], [1.9]], requires_grad=True)
    assert m.window_of(t).tolist() == [0, 1, 2, 3]
    joint = torch.tensor([[0.5]])
    m.set_window_start(1, m.windows[0](joint)[0])
    assert torch.allclose(m.windows[0](joint), m.windows[1](joint), atol=1e-6)
    assert residual_parts(m, t).r.shape == (4, 3)


def test_eps_advances_only_when_all_weights_exceed_delta(tmp_path):
    from pinn.train import train

    cfg = Config(t_span=(0.0, 0.2), n_windows=2, causal_eps_schedule=(1e-2, 1e-1), causal_delta=0.99,
                 causal_max_iters=15, depth=1, width=8, n_collocation=20, collocation="uniform",
                 lbfgs_iters=2, log_every=5, eval_every=5, print_every=0, ckpt_dir=str(tmp_path))
    model, history = train(cfg)
    assert cfg.arch == "1x8_win2_causal"
    assert len(history.eps_marks) == 4 and len(history.window_marks) == 2
    assert 0 < len(history.min_w) <= 4 * 15                   # every stage ends by delta or by the cap
    assert len(history.stage_status) == 4
    assert history.adam_iters == len(history.min_w)
    assert all(s["threshold_met"] or s["adam_steps"] == cfg.causal_max_iters
               for s in history.stage_status)
    assert all(m <= len(history.loss) for m, _ in history.eps_marks)
    assert (tmp_path / "window_00.pt").exists() and (tmp_path / "window_01.pt").exists()


def test_causal_extras_are_written(tmp_path):
    from pinn import viz
    from pinn.train import main

    cfg = Config(t_span=(0.0, 0.2), n_windows=2, causal_eps_schedule=(1e-2,), causal_max_iters=12,
                 depth=1, width=8, n_collocation=20, collocation="uniform", snapshot_every=6,
                 log_every=6, eval_every=6, print_every=0,
                 results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    main(cfg)
    summary = pd.read_csv(tmp_path / "run_summary.csv").iloc[0]
    assert summary.final_loss == pytest.approx(summary.residual_mse_collocation)
    assert summary.adam_steps > 0 and summary.lbfgs_evals == 0
    first = pd.read_csv(sorted((tmp_path / "breakdown").glob("epoch_*.csv"))[0])
    assert first.loc[first.t > cfg.t_span[1] / 2, "r_sq"].isna().all()
    assert first.loc[first.t < cfg.t_span[1] / 2, "r_sq"].notna().all()
    names = {p.name for p in viz.generate_run_extras(tmp_path)}
    for expected in ("causal_weights.png", "min_w.png", "window_grid.png", "joint_continuity.png"):
        assert expected in names, expected


def test_windowed_resume_skips_saved_windows(tmp_path):
    from pinn.train import train

    cfg = Config(t_span=(0.0, 0.2), n_windows=2, causal_eps_schedule=(1e-2,), causal_max_iters=5,
                 depth=1, width=8, n_collocation=20, collocation="uniform",
                 log_every=5, eval_every=5, print_every=1, ckpt_dir=str(tmp_path))
    train(cfg)
    _, history = train(cfg)          # complete checkpoint restores its original history
    assert history.loss and len(history.window_marks) == 2
    assert len(history.loss_phase) == len(history.loss)


def test_windowed_run_ends_with_final_snapshot(tmp_path):
    from pinn.train import train

    cfg = Config(t_span=(0.0, 0.2), n_windows=2, causal_eps_schedule=(1e-2,), causal_max_iters=3,
                 lbfgs_iters=2, snapshot_every=1000, depth=1, width=8, n_collocation=20,
                 collocation="uniform", ckpt_dir=str(tmp_path), results_dir=str(tmp_path))
    _, history = train(cfg)
    assert history.param_epochs[-1] == len(history.loss)


def test_warm_start_copies_previous_window_weights(tmp_path):
    import torch
    from pinn.train import train

    cfg = Config(t_span=(0.0, 0.2), n_windows=2, causal_eps_schedule=(1e-2,), causal_max_iters=0,
                 lbfgs_iters=0, warm_start=True, depth=1, width=8, n_collocation=20,
                 collocation="uniform", ckpt_dir=str(tmp_path))
    model, _ = train(cfg)            # zero iterations: window 1 must equal window 0 exactly
    for p, q in zip(model.windows[1].net.parameters(), model.windows[0].net.parameters()):
        assert torch.equal(p, q)
    assert "_warm" in cfg.arch


def test_index_page_links_every_run(tmp_path):
    from pinn.viz.index import write_index

    for name in ("a", "b"):
        (tmp_path / name / "figures").mkdir(parents=True)
        (tmp_path / name / "figures" / "trajectory.html").write_text("x")
    page = write_index(tmp_path, ["a", "b"]).read_text()
    assert "a/figures/trajectory.html" in page and "b/figures/trajectory.html" in page


def test_causal_index_drops_lbfgs_evals():
    from pinn.viz.training import causal_index

    # window 0: Adam 0-10 (eps ends at 4, 10), L-BFGS 10-30; window 1: Adam 30-36, L-BFGS 36-50
    marks = causal_index([(4, 0.1), (10, 1.0), (36, 0.1)], [30, 50])
    assert marks == [(4, 0.1), (10, 1.0), (16, 0.1)]


def test_snapshot_replay_matches_live_summary(tmp_path):
    from pinn.history import SnapshotWriter
    from pinn.train import main

    cfg = Config(t_span=(0.0, 0.2), epochs=4, lbfgs_iters=0, snapshot_every=2, depth=1, width=8,
                 n_collocation=10, collocation="uniform", log_every=2, eval_every=2, print_every=0,
                 results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    main(cfg)
    live = pd.read_csv(tmp_path / "point_summary.csv")
    replayed = SnapshotWriter.replay(tmp_path / "breakdown").summary_frame()
    assert list(live.epoch_of_max) == list(replayed.epoch_of_max)
    assert np.allclose(live.r_final, replayed.r_final, rtol=1e-4)
