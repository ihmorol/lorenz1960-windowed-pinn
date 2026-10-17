"""Train the Lorenz-1960 PINN, evaluate against the locked baseline, save results."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from . import viz as figures
from .config import Config, compute_error_metrics, reference_at, reference_trajectory
from .history import FLOAT_FORMAT, SnapshotWriter, TrainHistory, flat_grads, flat_params, residual_grid
from .functions.collocation import latin_hypercube_points, uniform_points
from .functions.losses import causal_loss
from .functions.optimizers import adam_with_decay, run_lbfgs
from .pinn import PINN, WindowedPINN, build_model, loss_terms, pinn_loss, residual_of as residual, residual_parts


def _save_progress(cfg: Config, model, history: TrainHistory, **state) -> None:
    """One atomic, complete resume point; per-window files remain export artifacts."""
    cfg.ckpt_path.mkdir(parents=True, exist_ok=True)
    target = cfg.ckpt_path / "progress.pt"
    pending = target.with_suffix(".tmp")
    torch.save({"config": cfg.record(), "model": model.state_dict(), "history": history,
                "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                **state}, pending)
    pending.replace(target)


def _load_progress(cfg: Config, model, device):
    path = cfg.ckpt_path / "progress.pt"
    if not path.exists():
        return None
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 2.6
        state = torch.load(path, map_location=device)
    if json.loads(json.dumps(state["config"])) != json.loads(json.dumps(cfg.record())):
        raise ValueError(f"checkpoint configuration differs from {path}")
    model.load_state_dict(state["model"])
    torch.set_rng_state(state["torch_rng"].cpu())
    np.random.set_state(state["numpy_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng"]])
    history = state["history"]
    if history.snapshots is not None:
        if history.snapshots.dir.resolve() != (cfg.results_path / "breakdown").resolve():
            raise ValueError("checkpoint snapshot directory differs from run directory")
        kept = {f"epoch_{epoch:06d}.csv" for epoch in history.snapshots.epochs}
        for file in history.snapshots.dir.glob("epoch_*.csv"):
            if file.name not in kept:
                file.unlink()
    return state, history


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_grid(cfg: Config, device: torch.device) -> Tensor:
    t = (uniform_points(cfg.t_span, cfg.n_collocation, cfg.n_windows) if cfg.collocation == "uniform"
         else latin_hypercube_points(cfg.t_span, cfg.n_collocation, cfg.seed))
    return torch.as_tensor(t, dtype=cfg.torch_dtype, device=device)


def train(cfg: Config) -> tuple[PINN, TrainHistory]:
    if cfg.n_windows > 1 or cfg.causal_eps_schedule or cfg.checkpoint_every or cfg.snapshot_every:
        cfg.ensure_record()
    set_seed(cfg.seed)
    device = get_device()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(cfg).to(device=device, dtype=cfg.torch_dtype)
    grid = make_grid(cfg, device)
    history = TrainHistory()
    t_ref, ys_ref = reference_trajectory(cfg, n=cfg.n_eval)

    if cfg.snapshot_every > 0:
        grid_t = grid.detach().cpu().numpy().reshape(-1)
        history.snapshots = SnapshotWriter(
            cfg.results_path / "breakdown", grid_t, reference_at(cfg, grid_t)
        )

    progress = (_load_progress(cfg, model, device)
                if cfg.n_windows > 1 or cfg.causal_eps_schedule or cfg.checkpoint_every else None)
    if progress is not None:
        state, history = progress
        if state["complete"]:
            return model, history

    if cfg.print_every:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[setup] device={device.type} | seed={cfg.seed} | {cfg.label}", flush=True)
        print(f"[setup] {n_params} params | {cfg.n_collocation} collocation pts on "
              f"t in [{cfg.t_span[0]}, {cfg.t_span[1]}] | ref traj {len(t_ref)} pts", flush=True)
        schedule = (f"StepLR factor {cfg.lr_decay:g} every {cfg.lr_decay_every} steps"
                    if cfg.lr_decay is not None else f"linear toward {cfg.lr_end:.1e}")
        if cfg.causal_eps_schedule:
            print(f"[setup] {cfg.n_windows} windows, up to {cfg.causal_max_iters} Adam steps per "
                  f"epsilon stage, then up to {cfg.lbfgs_iters} L-BFGS iterations per window; "
                  f"lr {cfg.lr_start:.1e}, {schedule}", flush=True)
        else:
            print(f"[setup] {cfg.epochs} Adam steps per window + up to {cfg.lbfgs_iters} "
                  f"L-BFGS iterations; lr {cfg.lr_start:.1e}, {schedule}", flush=True)
        if history.snapshots is not None:
            print(f"[setup] per-point snapshots every {cfg.snapshot_every} logged evaluations "
                  f"and at final window states -> {history.snapshots.dir}", flush=True)
    t_start = time.perf_counter() - history.wall_clock_s
    if cfg.n_windows > 1 or cfg.causal_eps_schedule:
        if progress is not None and state["mode"] != "windows":
            raise ValueError("checkpoint mode differs from windowed configuration")
        train_windows(model, grid, cfg, history, t_start, t_ref, ys_ref,
                      state["next_window"] if progress is not None else 0)
        return model, history

    adam, sched = adam_with_decay(model.parameters(), cfg)
    start = 0
    if progress is not None:
        if state["mode"] != "single":
            raise ValueError("checkpoint mode differs from single-domain configuration")
        adam.load_state_dict(state["adam"])
        sched.load_state_dict(state["sched"])
        start = state["next_epoch"]
        history.resumed_from = start

    for epoch in range(start, cfg.epochs):
        last = epoch == cfg.epochs - 1
        logging = epoch % cfg.log_every == 0 or last

        adam.zero_grad()
        res, ic, parts = loss_terms(model, grid.clone().requires_grad_(True))
        loss = res + model.gamma * ic if (cfg.ic == "soft" or cfg.end_state is not None) else res
        loss.backward()

        # Snapshot before the step, so the recorded residuals are exactly the ones
        # this epoch's loss and gradient were computed from. Costs one CSV write;
        # the tensors are already in hand.
        if history.snapshots is not None and (epoch % cfg.snapshot_every == 0 or last):
            history.snapshots.write(epoch, parts)
            history.param_epochs.append(epoch)
            history.param_trail.append(flat_params(model).cpu().numpy())
            history.grad_trail.append(flat_grads(model).cpu().numpy())

        lr = adam.param_groups[0]["lr"]
        before = flat_params(model) if logging else None
        adam.step()
        sched.step()
        history.loss.append(loss.item())
        history.loss_phase.append("adam")
        history.loss_window.append(0)

        if logging:
            history.record_step(epoch, model=model, loss=loss, residual=res, ic=ic,
                                lr=lr, params_before=before)
        if epoch % cfg.eval_every == 0 or last:
            history.record_reference(epoch, float(np.mean((predict(model, t_ref) - ys_ref) ** 2)))

        if cfg.checkpoint_every and (epoch + 1) % cfg.checkpoint_every == 0:
            history.wall_clock_s = time.perf_counter() - t_start
            _save_progress(cfg, model, history, mode="single", complete=False,
                           next_epoch=epoch + 1, adam=adam.state_dict(), sched=sched.state_dict())
        if cfg.print_every and (epoch % cfg.print_every == 0 or last):
            elapsed = time.perf_counter() - t_start
            eta = elapsed / (epoch + 1) * (cfg.epochs - epoch - 1)
            print(f"[adam]  epoch {epoch + 1:>6}/{cfg.epochs} | loss {history.loss[-1]:.4e} | "
                  f"res {res.item():.4e} | ic {ic.item():.4e} | "
                  f"|grad| {history.grad_norm[-1]:.3e} | step {history.update_norm[-1]:.3e} | "
                  f"lr {lr:.2e} | ref_mse {history.ref_mse[-1]:.4e} | "
                  f"{elapsed:6.1f}s elapsed, ~{eta:5.0f}s left", flush=True)

    history.adam_iters = cfg.epochs
    history.wall_clock_s = time.perf_counter() - t_start
    if cfg.print_every:
        print(f"[adam]  done in {time.perf_counter() - t_start:.1f}s | "
              f"final loss {history.loss[-1]:.4e} | best loss {min(history.loss):.4e}", flush=True)

    if cfg.lbfgs_iters > 0:
        def closure() -> Tensor:
            loss = pinn_loss(model, grid.clone().requires_grad_(True))
            history.loss.append(loss.item())
            history.loss_phase.append("lbfgs")
            history.loss_window.append(0)
            n = len(history.loss) - history.adam_iters
            if cfg.print_every and n % 10 == 1:
                print(f"[lbfgs] eval {n:>5} | loss {history.loss[-1]:.4e} | "
                      f"{time.perf_counter() - t_start:6.1f}s elapsed", flush=True)
            return loss

        run_lbfgs(model.parameters(), closure, cfg.lbfgs_iters, cfg.torch_dtype)
        history.wall_clock_s = time.perf_counter() - t_start
        if cfg.print_every:
            print(f"[lbfgs] done after {len(history.loss) - history.adam_iters} evals | "
                  f"final loss {history.loss[-1]:.4e}", flush=True)

    if history.snapshots is not None:
        model.zero_grad(set_to_none=True)
        final_parts = residual_parts(model, grid.clone().requires_grad_(True))
        history.snapshots.write(len(history.loss), final_parts)
        history.param_epochs.append(len(history.loss))
        history.param_trail.append(flat_params(model).cpu().numpy())
        final_parts.r.pow(2).mean().backward()
        history.grad_trail.append(flat_grads(model).cpu().numpy())
    history.wall_clock_s = time.perf_counter() - t_start
    if cfg.checkpoint_every:
        _save_progress(cfg, model, history, mode="single", complete=True,
                       next_epoch=cfg.epochs, adam=adam.state_dict(), sched=sched.state_dict())

    return model, history


def train_windows(model, grid: Tensor, cfg: Config, history: TrainHistory, t_start: float,
                  t_ref: np.ndarray, ys_ref: np.ndarray, start_window: int = 0) -> None:
    """Window by window: Adam under each eps until min_i w_i > delta, then L-BFGS, then hand
    the end state to the next window. One window with an empty schedule is plain Adam."""
    windows = list(model.windows) if isinstance(model, WindowedPINN) else [model]
    stages = list(cfg.causal_eps_schedule) or [None]
    cap = cfg.causal_max_iters if cfg.causal_eps_schedule else cfg.epochs
    device = grid.device
    if isinstance(model, WindowedPINN):
        counts = torch.bincount(model.window_of(grid), minlength=len(windows))
        if torch.any(counts == 0):
            raise ValueError("collocation grid leaves an empty window")

    def snapshot(epoch: int, w, trained_until: float, final: bool = False) -> None:
        if history.snapshots is None or (epoch % cfg.snapshot_every and not final):
            return
        if history.snapshots.epochs and epoch == history.snapshots.epochs[-1]:
            return
        history.snapshots.write(epoch, residual_parts(model, grid.clone().requires_grad_(True)),
                                trained_until=trained_until)
        history.param_epochs.append(epoch)
        history.param_trail.append(flat_params(model).cpu().numpy())
        history.grad_trail.append(flat_grads(model).cpu().numpy())
        if w is not None:
            history.weight_profiles.append(w.cpu().numpy())

    for k in range(start_window, len(windows)):
        sub = windows[k]
        edge = np.asarray(sub.tf, dtype=np.float64 if cfg.dtype == "float64" else np.float32)
        coverage = sub.tf if k + 1 == len(windows) else float(
            np.nextafter(edge, np.asarray(-np.inf, dtype=edge.dtype)))
        saved = cfg.ckpt_path / f"window_{k:02d}.pt"
        pts = grid[model.window_of(grid) == k] if isinstance(model, WindowedPINN) else grid
        if isinstance(model, WindowedPINN) and k + 1 < len(windows):
            # The shared edge belongs to the next bucket; also enforce the
            # outgoing window's residual at the state it hands forward.
            endpoint = torch.tensor([[model.edges[k + 1]]], dtype=grid.dtype, device=device)
            pts = torch.cat((pts, endpoint))
        if k and cfg.warm_start:
            with torch.no_grad():
                for p, q in zip(sub.net.parameters(), windows[k - 1].net.parameters()):
                    p.copy_(q)
        adam, sched = adam_with_decay(sub.parameters(), cfg, cap * len(stages))
        for eps in stages:
            stage_start = history.adam_iters
            met = False
            for step in range(cap):
                epoch = len(history.loss)
                model.zero_grad(set_to_none=True)
                t = pts.clone().requires_grad_(True)
                raw_res, ic, parts = loss_terms(sub, t)
                res, w = causal_loss(parts.r, t, eps) if eps is not None else (raw_res, None)
                if w is not None:
                    history.min_w.append(float(w.min()))
                loss = res + sub.gamma * ic if (cfg.ic == "soft" or sub.end_state is not None) else res
                loss.backward()
                snapshot(epoch, w, coverage)
                logging = epoch % cfg.log_every == 0
                before = flat_params(sub) if logging else None
                lr = adam.param_groups[0]["lr"]
                adam.step()
                sched.step()
                history.loss.append(loss.item())
                history.loss_phase.append("adam")
                history.loss_window.append(k)
                history.adam_iters += 1
                if logging:
                    history.record_step(epoch, model=sub, loss=loss, residual=raw_res, ic=ic,
                                        lr=lr, params_before=before)
                if epoch % cfg.eval_every == 0:
                    valid = t_ref <= coverage
                    history.record_reference(epoch, float(np.mean((predict(model, t_ref[valid]) - ys_ref[valid]) ** 2)),
                                             coverage)
                if cfg.print_every and epoch % cfg.print_every == 0:
                    min_w = history.min_w[-1] if w is not None else 1.0
                    print(f"[adam]  window {k:>2} eps {eps} | epoch {epoch:>7} | loss {loss.item():.4e} "
                          f"| min_w {min_w:.3f} | {time.perf_counter() - t_start:7.1f}s", flush=True)
                if w is not None and (history.min_w[-1] > cfg.causal_delta or step == cap - 1):
                    # Certify the updated parameters, which enter the next stage.
                    post = residual_parts(sub, pts.clone().requires_grad_(True))
                    _, post_w = causal_loss(post.r, pts, eps)
                    history.min_w[-1] = float(post_w.min())
                    if history.min_w[-1] > cfg.causal_delta:
                        met = True
                        break
            if eps is not None:
                history.eps_marks.append((len(history.loss), eps))
                history.stage_status.append({"window": k, "eps": eps,
                                             "adam_steps": history.adam_iters - stage_start,
                                             "min_w_final": history.min_w[-1] if history.adam_iters > stage_start
                                                            else float("nan"),
                                             "threshold_met": met})
        if cfg.lbfgs_iters > 0:
            def closure() -> Tensor:
                loss = pinn_loss(sub, pts.clone().requires_grad_(True))
                history.loss.append(loss.item())
                history.loss_phase.append("lbfgs")
                history.loss_window.append(k)
                return loss
            run_lbfgs(sub.parameters(), closure, cfg.lbfgs_iters, cfg.torch_dtype)
        cfg.ckpt_path.mkdir(parents=True, exist_ok=True)
        torch.save(sub.state_dict(), saved)
        model.zero_grad(set_to_none=True)
        pinn_loss(sub, pts.clone().requires_grad_(True)).backward()
        done = torch.ones(len(pts), dtype=grid.dtype) if stages[0] is not None else None
        snapshot(len(history.loss), done, coverage, final=True)
        history.window_marks.append(len(history.loss))
        if isinstance(model, WindowedPINN) and k + 1 < len(windows):
            with torch.no_grad():
                end = torch.tensor([[model.edges[k + 1]]], dtype=grid.dtype, device=device)
                model.set_window_start(k + 1, sub(end)[0])
        history.wall_clock_s = time.perf_counter() - t_start
        _save_progress(cfg, model, history, mode="windows", complete=k + 1 == len(windows),
                       next_window=k + 1)
        if cfg.print_every and history.loss:
            print(f"[window] {k:>2} done | loss {history.loss[-1]:.4e} | "
                  f"{time.perf_counter() - t_start:7.1f}s", flush=True)
    history.wall_clock_s = time.perf_counter() - t_start


def predict(model: PINN, t: np.ndarray) -> np.ndarray:
    p = next(model.parameters())
    tt = torch.as_tensor(t, dtype=p.dtype, device=p.device).reshape(-1, 1)
    with torch.no_grad():
        return model(tt).cpu().numpy().astype(np.float64)


def residual_at(model: PINN, t: np.ndarray) -> np.ndarray:
    """ODE residual r(t) = du/dt - f(u) evaluated at arbitrary times."""
    p = next(model.parameters())
    tt = torch.as_tensor(t, dtype=p.dtype, device=p.device).reshape(-1, 1).requires_grad_(True)
    return residual(model, tt).detach().cpu().numpy().astype(np.float64)


def evaluate(model: PINN, cfg: Config) -> pd.DataFrame:
    t, ys = reference_trajectory(cfg, n=cfg.n_eval)
    return compute_error_metrics(ys, predict(model, t))


def collect_artifacts(model: PINN, history: TrainHistory, cfg: Config) -> figures.RunArtifacts:
    """Bundle a finished run into the plain-array record the figure suite reads."""
    t, ref = reference_trajectory(cfg, n=cfg.n_eval)
    grid = make_grid(cfg, next(model.parameters()).device).cpu().numpy().reshape(-1)
    return figures.RunArtifacts(
        t=t,
        pred=predict(model, t),
        ref=ref,
        history=history,
        coefficients=cfg.coefficients,
        collocation_t=grid,
        collocation_residual=residual_at(model, grid),
        dense_residual=residual_at(model, t),
        t_span=cfg.t_span,
        label=cfg.label,
    )


LOSS_THRESHOLDS = (1e-4, 1e-6, 1e-8)


def _epochs_to(loss: list[float], threshold: float) -> float:
    """First iteration index at which the loss dropped below ``threshold``."""
    for i, value in enumerate(loss):
        if value < threshold:
            return float(i)
    return float("nan")


def run_summary(
    model: PINN, history: TrainHistory, cfg: Config, run: figures.RunArtifacts | None = None
) -> pd.DataFrame:
    """One wide row describing a finished run: config, accuracy, physics, cost.

    This is the row the architecture sweep concatenates into ``comparison.csv``.
    Every column is computed after training; nothing here feeds back into the loss.
    """
    run = run if run is not None else collect_artifacts(model, history, cfg)
    pred, ref, t = run.pred, run.ref, run.t
    err = pred - ref
    err_norm = np.linalg.norm(err, axis=1)
    metrics = run.metrics.set_index("state")

    row: dict[str, object] = {
        # --- configuration
        "arch": cfg.arch, "problem": cfg.problem, "depth": cfg.depth, "width": cfg.width,
        "n_windows": cfg.n_windows, "causal_eps_schedule": " ".join(f"{e:g}" for e in cfg.causal_eps_schedule),
        "causal_delta": cfg.causal_delta, "causal_max_iters": cfg.causal_max_iters, "warm_start": cfg.warm_start,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "activation": cfg.activation, "ic": cfg.ic, "gamma": cfg.gamma,
        "epochs": cfg.epochs, "lbfgs_iters": cfg.lbfgs_iters, "seed": cfg.seed,
        "n_collocation": cfg.n_collocation,
        "lr_start": cfg.lr_start, "lr_end": cfg.lr_end,
        "t_start": cfg.t_span[0], "t_end": cfg.t_span[1],
        "dtype": cfg.dtype, "ic_scale": cfg.ic_scale, "collocation": cfg.collocation,
        "n_eval": cfg.n_eval, "lr_decay": cfg.lr_decay if cfg.lr_decay is not None else float("nan"),
        "lr_decay_every": cfg.lr_decay_every,
        "k": cfg.k, "l": cfg.l,
        "initial_state": json.dumps(cfg.initial_state), "end_state": json.dumps(cfg.end_state),
    }

    # --- accuracy, per state and combined
    for state in ("x", "y", "z", "combined_l2"):
        for metric in ("mae", "rmse", "max_abs_error"):
            row[f"{metric}_{state}"] = float(metrics.loc[state, metric])

    for j, name in enumerate("xyz"):
        span = float(ref[:, j].max() - ref[:, j].min())
        ss_tot = float(((ref[:, j] - ref[:, j].mean()) ** 2).sum())
        row[f"rel_mae_{name}"] = float(np.abs(err[:, j]).mean() / span) if span else float("nan")
        row[f"r2_{name}"] = 1.0 - float((err[:, j] ** 2).sum()) / ss_tot if ss_tot else float("nan")
        # Endpoint state: where the trajectory actually landed versus the truth.
        row[f"final_{name}"] = float(pred[-1, j])
        row[f"final_ref_{name}"] = float(ref[-1, j])
        row[f"final_err_{name}"] = float(err[-1, j])

    for q in (50, 90, 99):
        row[f"err_norm_p{q}"] = float(np.percentile(err_norm, q))
    row["err_norm_max"] = float(err_norm.max())

    # --- physics: does the ODE hold away from the points we trained on?
    coll = float(np.mean(np.asarray(run.collocation_residual) ** 2)) \
        if run.collocation_residual is not None else float("nan")
    dense = float(np.mean(np.asarray(run.dense_residual) ** 2)) \
        if run.dense_residual is not None else float("nan")
    row["residual_mse_collocation"] = coll
    row["final_loss"] = coll
    row["final_residual_mse_full"] = coll
    row["last_logged_objective"] = float(history.loss[-1]) if history.loss else float("nan")
    comparable_history = (cfg.n_windows == 1 and not cfg.causal_eps_schedule
                          and cfg.ic == "hard" and cfg.end_state is None)
    row["best_loss"] = float(min(history.loss)) if history.loss and comparable_history else float("nan")
    row["residual_mse_dense"] = dense
    row["residual_generalisation_gap"] = dense / coll if coll else float("nan")

    # --- conserved quantities
    _, drift = figures.invariant_series(pred, run.coefficients)
    _, ref_drift = figures.invariant_series(ref, run.coefficients)
    for i in range(drift.shape[1]):
        row[f"invariant_{i + 1}_max_drift"] = float(drift[:, i].max())
        row[f"invariant_{i + 1}_max_drift_reference"] = float(ref_drift[:, i].max())

    # --- cost
    wall = float(history.wall_clock_s)
    row["wall_clock_s"] = wall
    row["adam_steps"] = history.adam_iters
    row["lbfgs_evals"] = sum(p == "lbfgs" for p in history.loss_phase) if history.loss_phase else max(
        0, len(history.loss) - history.adam_iters)
    row["capped_stages"] = (sum(not s["threshold_met"] for s in history.stage_status)
                             if history.stage_status else
                             (float("nan") if cfg.causal_eps_schedule else 0))
    row["ms_per_epoch"] = (wall / history.adam_iters * 1e3
                            if row["lbfgs_evals"] == 0 and history.adam_iters else float("nan"))
    for threshold in LOSS_THRESHOLDS:
        row[f"epochs_to_{threshold:.0e}"] = (_epochs_to(history.loss, threshold)
                                                if comparable_history and row["lbfgs_evals"] == 0
                                                else float("nan"))
    row["device"] = next(model.parameters()).device.type
    # CUDA-only; NaN on CPU, where torch exposes no equivalent counter.
    row["peak_mem_mb"] = (
        torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else float("nan")
    )
    row["n_snapshots"] = (len(history.snapshots.epochs) if history.snapshots
                          else history.saved_n_snapshots)
    return pd.DataFrame([row])


def write_breakdown_figures(cfg: Config, formats=figures.FORMATS) -> dict[str, list]:
    """Render the per-epoch collocation figures from a run's ``breakdown/`` CSVs.

    Lives here rather than in :mod:`pinn.figures` because it reads the breakdown
    through :mod:`pinn.history`, and that module imports torch; ``figures`` stays
    a pure array-in, figure-out module.
    """
    out = cfg.results_path / "figures"
    epochs, centres, values = residual_grid(cfg.results_path / "breakdown")
    summary = pd.read_csv(cfg.results_path / "point_summary.csv")
    figures.set_style()

    written = {
        "residual_evolution": figures.save_figure(
            figures.fig_residual_evolution(epochs, centres, values, cfg.label),
            out, "residual_evolution", formats),
        "residual_profiles": figures.save_figure(
            figures.fig_residual_profiles(epochs, centres, values, label=cfg.label),
            out, "residual_profiles", formats),
        "point_convergence": figures.save_figure(
            figures.fig_point_convergence(summary, cfg.label),
            out, "point_convergence", formats),
    }
    html = figures.write_residual_surface_html(
        epochs, centres, values, out / "residual_surface.html", cfg.label)
    if html is not None:
        written["residual_surface_html"] = [html]
    return written


def save_results(model: PINN, history: TrainHistory, cfg: Config) -> pd.DataFrame:
    if not history.loss:
        raise ValueError("cannot report a run with no optimizer steps or evaluations")
    cfg.ensure_record()
    cfg.ckpt_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.ckpt_path / "pinn.pt")
    if history.param_trail:
        extra = {}
        if history.weight_profiles:
            width = max(len(w) for w in history.weight_profiles)
            extra["weights"] = np.stack([np.pad(w, (0, width - len(w)), constant_values=np.nan)
                                         for w in history.weight_profiles])
        np.savez_compressed(cfg.ckpt_path / "param_trail.npz", epochs=np.asarray(history.param_epochs),
                            params=np.stack(history.param_trail), grads=np.stack(history.grad_trail), **extra)
    if history.min_w:
        pd.DataFrame({"iteration": range(len(history.min_w)), "min_w": history.min_w}).to_csv(
            cfg.ckpt_path / "causal.csv", index=False)
    if history.window_marks or history.eps_marks:
        rows = [(i, e, "eps") for i, e in history.eps_marks] + \
               [(i, float("nan"), "window") for i in history.window_marks]
        pd.DataFrame(rows, columns=["iteration", "eps", "kind"]).to_csv(cfg.ckpt_path / "marks.csv", index=False)
    if history.stage_status:
        pd.DataFrame(history.stage_status).to_csv(cfg.ckpt_path / "stages.csv", index=False)
    if cfg.print_every:
        print(f"[save]  checkpoint + telemetry -> {cfg.ckpt_path}", flush=True)
        print(f"[save]  rendering figures -> {cfg.results_path} ...", flush=True)
    run = collect_artifacts(model, history, cfg)
    metrics = figures.write_run_report(run, cfg.results_path, data_dir=cfg.ckpt_path)

    summary = run_summary(model, history, cfg, run)
    summary.to_csv(cfg.results_path / "run_summary.csv", index=False, float_format=FLOAT_FORMAT)
    if history.snapshots is not None:
        history.snapshots.finalize(cfg.results_path)
        breakdown = write_breakdown_figures(cfg)
        figures.generate_run_extras(cfg.results_path)
        if cfg.print_every:
            print(f"[save]  {len(history.snapshots.epochs)} snapshots + point_summary.csv + "
                  f"{len(breakdown)} breakdown figures"
                  f"{'' if 'residual_surface_html' in breakdown else ' (no plotly: 3-D HTML skipped)'}",
                  flush=True)
    if cfg.print_every:
        print(f"[save]  wrote metrics.csv, run_summary.csv + "
              f"{len(list((cfg.results_path / 'figures').glob('*')))} figure files", flush=True)
    return metrics


def load_run(cfg: Config | None = None) -> tuple[PINN, TrainHistory, Config]:
    """Reload a finished run's trained weights and telemetry, with no retraining."""
    cfg = cfg or Config()
    device = get_device()
    model = build_model(cfg).to(device=device, dtype=cfg.torch_dtype)
    try:
        state = torch.load(cfg.ckpt_path / "pinn.pt", map_location=device, weights_only=True)
    except TypeError:  # older torch
        state = torch.load(cfg.ckpt_path / "pinn.pt", map_location=device)
    model.load_state_dict(state)
    return model, TrainHistory.from_saved(cfg.ckpt_path, cfg.results_path), cfg


def rebuild_figures(cfg: Config | None = None) -> pd.DataFrame:
    """Redraw every figure from a finished run's checkpoint and saved CSVs, no retraining."""
    model, history, cfg = load_run(cfg)
    return figures.write_run_report(
        collect_artifacts(model, history, cfg), cfg.results_path, data_dir=cfg.ckpt_path
    )


def main(cfg: Config | None = None) -> tuple[PINN, TrainHistory, pd.DataFrame]:
    cfg = cfg or Config()
    t0 = time.perf_counter()
    model, history = train(cfg)
    metrics = save_results(model, history, cfg)
    if cfg.print_every:
        print(f"[done]  total {time.perf_counter() - t0:.1f}s", flush=True)
    print(metrics.to_string(index=False))
    return model, history, metrics


if __name__ == "__main__":
    main()
