"""Resume, portability, shared-network control, and the Colab driver's Drive sync.

Each interruption test kills training at a real checkpoint and checks the
resumed run against an uninterrupted one, bit for bit.
"""
import json
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from pinn import train as train_module
from pinn.config import Config
from pinn.pinn import PINN, build_model
from pinn.train import train

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


class Interrupted(Exception):
    pass


def _interrupt_after(monkeypatch, saves: int, inside_only: bool = False):
    original, seen = train_module._save_progress, []

    def save(cfg, model, history, **state):
        original(cfg, model, history, **state)
        if not inside_only or state.get("inside"):
            seen.append(1)
            if len(seen) == saves:
                raise Interrupted
    monkeypatch.setattr(train_module, "_save_progress", save)


def _same_run(a, b):
    (ma, ha), (mb, hb) = a, b
    assert ha.loss == hb.loss and ha.min_w == hb.min_w and ha.stage_status == hb.stage_status
    assert ha.adam_iters == hb.adam_iters
    for p, q in zip(ma.state_dict().values(), mb.state_dict().values()):
        assert torch.equal(p, q)


def _causal(tmp_path, name, **kw):
    return Config(t_span=(0.0, 1.0), n_collocation=24, causal_eps_schedule=(1.0, 100.0),
                  causal_max_iters=7, checkpoint_every=4, print_every=0, eval_every=5,
                  results_dir=str(tmp_path / name), ckpt_dir=str(tmp_path / name / "history"), **kw)


def test_single_network_causal_run_resumes_mid_stage(tmp_path, monkeypatch):
    """R3's path: one window + causal schedule must checkpoint every
    checkpoint_every Adam steps and resume with optimizer/scheduler state."""
    reference = train(_causal(tmp_path, "whole"))
    cfg = _causal(tmp_path, "cut")
    _interrupt_after(monkeypatch, saves=2, inside_only=True)
    with pytest.raises(Interrupted):
        train(cfg)
    state = torch.load(cfg.ckpt_path / "progress.pt", weights_only=False)
    assert state["inside"]["stage"] == 1 and state["history"].adam_iters == 8
    monkeypatch.undo()
    _same_run(reference, train(cfg))


def test_windowed_snapshot_run_resumes_from_another_checkout(tmp_path, monkeypatch):
    """A progress.pt written under one root must resume after the run directory
    moves (Windows -> Colab); snapshot paths are rebound, not rejected."""
    kw = dict(n_windows=2, snapshot_every=3)
    reference = train(_causal(tmp_path, "whole", **kw))
    _interrupt_after(monkeypatch, saves=1)
    with pytest.raises(Interrupted):
        train(_causal(tmp_path / "machine_a", "run", **kw))
    monkeypatch.undo()
    shutil.copytree(tmp_path / "machine_a", tmp_path / "machine_b")
    shutil.rmtree(tmp_path / "machine_a")
    moved = _causal(tmp_path / "machine_b", "run", **kw)
    resumed = train(moved)
    _same_run(reference, resumed)
    assert resumed[1].snapshots.dir == moved.results_path / "breakdown"


def test_records_predating_new_fields_still_match():
    record = Config().record()
    del record["shared_network"]
    assert Config().matches(record)
    assert not replace(Config(), shared_network=False, epochs=7).matches(record)


def test_shared_network_is_one_network_trained_window_by_window(tmp_path):
    """R6: one parameter set across windows; training later windows moves the
    earlier windows' fit (the handoff-forgetting control), hard IC still exact."""
    cfg = Config(t_span=(0.0, 1.0), n_windows=3, shared_network=True, epochs=15, n_collocation=30,
                 print_every=0, results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    model = build_model(cfg)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in PINN(Config()).parameters())
    assert model.windows[0].net is model.windows[2].net
    model, history = train(cfg)
    assert history.adam_iters == 45
    after_w0 = torch.load(cfg.ckpt_path / "window_00.pt")
    assert not torch.equal(after_w0["net.0.weight"], model.windows[0].net[0].weight)
    t0 = torch.zeros(1, 1, dtype=torch.float32)
    assert torch.allclose(model(t0), torch.tensor([cfg.initial_state]))


def test_ablation_arms_differ_from_their_comparison_in_the_stated_way():
    from paper_ablations import arm_config
    base = arm_config("r4", 1).record()

    def diff(arm):
        return {k for k, v in arm_config(arm, 1).record().items() if v != base[k]} - {
            "results_dir", "ckpt_dir", "runs_dir"}
    assert diff("r2") == {"warm_start"}
    assert diff("r1") == {"causal_eps_schedule", "causal_max_iters", "epochs"}
    r1 = arm_config("r1", 1).record()
    assert {k for k, v in arm_config("r6", 1).record().items() if v != r1[k]} == {"shared_network"}
    r3 = arm_config("r3", 0)
    assert (r3.n_windows, r3.n_collocation, r3.collocation, r3.lbfgs_iters) == (1, 39793, "lhs", 5000)
    assert r3.causal_max_iters * len(r3.causal_eps_schedule) == 40000 and r3.lr_decay_every == 5000


# ---------------------------------------------------------------------------
# Colab driver: Drive sync and restore
# ---------------------------------------------------------------------------
def _write(path, text, age=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    t = time.time() - age
    os.utime(path, (t, t))


def test_sync_skips_bulk_and_keeps_previous_checkpoint(tmp_path):
    from colab_suite import sync_up
    local, drive = tmp_path / "local", tmp_path / "drive"
    _write(local / "a_seed1" / "history" / "progress.pt", "v1", age=60)
    _write(local / "a_seed1" / "history" / "window_00.pt", "bulk", age=60)
    _write(local / "a_seed1" / "history" / "progress.tmp", "half", age=60)
    _write(local / "a_seed1" / "history" / "loss_history.csv", "young")
    assert sync_up(local, drive, set(), min_age=5) == (1, 0)
    assert sorted(p.name for p in drive.rglob("*") if p.is_file()) == ["progress.pt"]

    _write(local / "a_seed1" / "history" / "progress.pt", "v2-longer", age=30)
    sync_up(local, drive, set())
    h = drive / "a_seed1" / "history"
    assert (h / "progress.pt").read_text() == "v2-longer" and (h / "progress.prev.pt").read_text() == "v1"
    assert not list(drive.rglob("*.partial"))
    assert sync_up(local, drive, set())[0] == 0          # unchanged files are not recopied

    _write(local / "a_seed1" / "run_summary.csv", "rmse\n1\n", age=30)
    _write(local / "a_seed1" / "breakdown" / "epoch_000000.csv", "x", age=30)
    copied, removed = sync_up(local, drive, {"a_seed1"})
    assert removed == 2 and not list(h.glob("progress*"))
    assert (drive / "a_seed1" / "run_summary.csv").exists() and not (drive / "a_seed1" / "breakdown").exists()


def test_restore_prefers_drive_validates_checkpoints_and_keeps_local_finished_runs(tmp_path):
    from colab_suite import restore
    local, drive = tmp_path / "local", tmp_path / "drive"
    good = drive / "b_seed1" / "history" / "progress.prev.pt"
    good.parent.mkdir(parents=True)
    torch.save({"ok": 1}, good)
    _write(drive / "b_seed1" / "history" / "progress.pt", "truncated")
    _write(drive / "b_seed1" / "history" / "loss_history.csv", "drive")
    _write(local / "b_seed1" / "history" / "loss_history.csv", "stale", age=3600)
    _write(drive / "c_seed1" / "history" / "config.json", "drive copy")
    _write(local / "c_seed1" / "run_summary.csv", "local finished")
    _write(local / "c_seed1" / "history" / "config.json", "local copy")

    notes = restore(drive, local)
    assert torch.load(local / "b_seed1" / "history" / "progress.pt", weights_only=False) == {"ok": 1}
    assert len(notes) == 1 and "previous" in notes[0]
    assert (local / "b_seed1" / "history" / "loss_history.csv").read_text() == "drive"
    assert (local / "c_seed1" / "history" / "config.json").read_text() == "local copy"


def test_driver_runs_without_drive_and_still_zips(tmp_path, monkeypatch):
    """No Drive and nothing to run: cleanup must not crash and the zip is written."""
    import colab_suite
    out = tmp_path / "out"
    _write(out / "r4_candidate_seed1" / "run_summary.csv", "x\n1\n")
    _write(out / "r4_candidate_seed1" / "history" / "progress.pt", "resume state")
    monkeypatch.setattr(colab_suite, "REPO", tmp_path)
    monkeypatch.setattr(colab_suite, "PLAN", [("r4", 1)])
    monkeypatch.setattr(colab_suite.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0, "stdout": ""})())
    monkeypatch.setattr(sys, "argv", ["colab_suite.py", "--out", str(out), "--no-drive"])
    colab_suite.main()
    import zipfile
    names = zipfile.ZipFile(tmp_path / "paper_ablations_results.zip").namelist()
    assert "out/r4_candidate_seed1/run_summary.csv" in names
    assert not any("progress" in n for n in names)
    assert json.loads((out / "environment.jsonl").read_text().splitlines()[0])["jobs"] == 1
