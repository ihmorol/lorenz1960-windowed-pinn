"""Colab driver for the paper's ablation suite: parallel arms, Drive persistence.

Each (arm, seed) runs as its own process on the shared GPU, ``--jobs`` at a
time, longest first (R3 is the critical path). Output per run goes to
``<out>/logs/<run>.log``. Durability:

- every finished run is synced to Drive immediately; a background sync runs
  every ``SYNC_EVERY_S`` while runs train, so a hard runtime kill loses at most
  that much unsynced work (plus the run's progress since its last checkpoint:
  one window for windowed arms, 2,000 Adam steps for R3)
- Drive files are replaced atomically (copy to ``.partial``, then rename), and
  the previous ``progress.pt`` is kept as ``progress.prev.pt``
- on start, Drive is restored over the local tree, except runs that are finished
  locally and not on Drive; each restored ``progress.pt`` is loaded to validate
  it, falling back to the previous one
- finished runs drop optimizer state and per-point snapshots from Drive

Wall clock per run is measured under contention from the other jobs.

Usage (see README.md):
    python scripts/colab_suite.py [--jobs 4] [--out runs/paper_ablations] [--no-drive]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DRIVE_BASE = Path("/content/drive/MyDrive/lorenz1960-pinn-ablation")
SYNC_EVERY_S = 300
MIN_AGE_S = 5           # periodic sync leaves files younger than this (still being written)
GB_PER_JOB = 2.5        # host RAM one training process can reach (torch + CUDA + history)
NAMES = {"r1": "r1_no_causal", "r2": "r2_no_warm", "r3": "r3_causal_single",
         "r4": "r4_candidate", "r6": "r6_sequential"}
# Longest first. R3 seed 0 pairs with F1 (seed 0); seeds 1-3 pair across the windowed arms.
PLAN = [("r3", 0)] + [(a, s) for s in (1, 2, 3) for a in ("r4", "r1", "r2", "r6")] + [("r4", 4)]

_sync_lock = threading.Lock()


def run_name(arm: str, seed: int) -> str:
    return f"{NAMES[arm]}_seed{seed}"


def _skip(rel: Path, finished: set[str]) -> bool:
    """Files never worth a Drive round trip, and resume state of finished runs."""
    name = rel.name
    if name.endswith((".tmp", ".partial")) or name.startswith("window_") or name == "param_trail.npz":
        return True
    return rel.parts[0] in finished and (name.startswith("progress") or "breakdown" in rel.parts)


def _same(s: Path, d: Path) -> bool:
    if not d.exists():
        return False
    a, b = s.stat(), d.stat()
    return a.st_size == b.st_size and abs(a.st_mtime - b.st_mtime) <= 1.0


def _copy(s: Path, d: Path, keep_prev: bool = False) -> None:
    d.parent.mkdir(parents=True, exist_ok=True)
    partial = d.with_name(d.name + ".partial")
    shutil.copy2(s, partial)
    if keep_prev and d.name == "progress.pt" and d.exists():
        os.replace(d, d.with_name("progress.prev.pt"))
    os.replace(partial, d)


def sync_up(src: Path, dst: Path, finished: set[str], min_age: float = 0.0) -> tuple[int, int]:
    """Local -> Drive. Returns (copied, removed)."""
    copied = removed = 0
    now = time.time()
    with _sync_lock:
        finished = set(finished)
        for s in sorted(p for p in src.rglob("*") if p.is_file()):
            rel = s.relative_to(src)
            if _skip(rel, finished) or now - s.stat().st_mtime < min_age:
                continue
            d = dst / rel
            if not _same(s, d):
                _copy(s, d, keep_prev=True)
                copied += 1
        for d in [p for p in dst.rglob("*") if p.is_file()] if dst.exists() else []:
            rel = d.relative_to(dst)
            if rel.parts[0] in finished and (d.name.startswith("progress") or "breakdown" in rel.parts):
                d.unlink()
                removed += 1
    return copied, removed


def _loadable(path: Path) -> bool:
    import torch
    try:
        torch.load(path, map_location="cpu", weights_only=False)
        return True
    except Exception:
        return False


def restore(drive: Path, local: Path) -> list[str]:
    """Drive -> local. Drive wins, except for runs that only the local tree has finished."""
    notes = []
    if not drive.exists():
        return notes
    for s in sorted(p for p in drive.rglob("*") if p.is_file()):
        rel = s.relative_to(drive)
        run = rel.parts[0]
        if s.name.endswith((".partial", ".prev.pt")) or (
                (local / run / "run_summary.csv").exists() and not (drive / run / "run_summary.csv").exists()):
            continue
        if not _same(s, local / rel):
            _copy(s, local / rel)
    for progress in local.glob("*/history/progress.pt"):
        if _loadable(progress):
            continue
        prev = drive / progress.relative_to(local).with_name("progress.prev.pt")
        if prev.exists() and _loadable(prev):
            _copy(prev, progress)
            notes.append(f"{progress}: damaged, restored the previous checkpoint")
        else:
            progress.unlink()
            notes.append(f"{progress}: damaged and no valid previous one; run restarts")
    return notes


def drive_ready(root: Path) -> bool:
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
        return True
    except OSError:
        return False


def environment(jobs: int) -> dict:
    def out(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO).stdout.strip()
        except OSError:
            return ""
    versions = {}
    for pkg in ("torch", "numpy", "scipy", "pandas", "matplotlib"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": out(["git", "rev-parse", "HEAD"]),
            "git_dirty": bool(out(["git", "status", "--porcelain", "--untracked-files=no"])),
            "python": sys.version.split()[0], "packages": versions,
            "gpu": out(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader"]) or "none",
            "cpu_count": os.cpu_count(), "jobs": jobs}


def available_ram_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


def run_process(arm: str, seed: int, out: str, logs: Path, threads: int) -> tuple[int, float]:
    started = time.perf_counter()
    print(f"[start] {run_name(arm, seed)} -> {logs / run_name(arm, seed)}.log", flush=True)
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    with open(logs / f"{run_name(arm, seed)}.log", "a") as log:
        code = subprocess.run([sys.executable, str(REPO / "scripts" / "paper_ablations.py"),
                               "--arms", arm, "--seeds", str(seed), "--out", out],
                              stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO).returncode
    return code, time.perf_counter() - started


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4, help="runs in parallel on the one GPU")
    ap.add_argument("--out", default="runs/paper_ablations",
                    help="output root; a new root reruns everything from scratch")
    ap.add_argument("--arms", default=None, help="restrict the plan to these arms, e.g. r1,r4")
    ap.add_argument("--seeds", default=None, help="replace the plan's seeds, e.g. --arms r3 --seeds 1")
    ap.add_argument("--no-drive", action="store_true", help="run without Drive persistence")
    ap.add_argument("--drive-root", default=None, help=f"Drive folder (default {DRIVE_BASE})")
    args = ap.parse_args()

    out_dir = Path(args.out) if Path(args.out).is_absolute() else REPO / args.out
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    on_colab = "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ
    drive = None
    if not args.no_drive and (on_colab or args.drive_root):
        drive = Path(args.drive_root or DRIVE_BASE) / out_dir.name
        mounted = args.drive_root or os.path.ismount("/content/drive")   # else mkdir would fake it locally
        if not mounted or not drive_ready(drive.parent):
            sys.exit("Drive is not mounted or not writable. Run in a notebook cell first:\n"
                     "    from google.colab import drive; drive.mount('/content/drive')\n"
                     "or pass --no-drive to run without persistence.")
        for note in restore(drive, out_dir):
            print(f"[drive] {note}", flush=True)
        print(f"[drive] persisting to {drive}", flush=True)

    plan = [(a, s) for a, s in PLAN if not args.arms or a in args.arms.split(",")]
    if args.seeds:
        plan = [(a, int(s)) for a in (args.arms or "r3,r4,r1,r2,r6").split(",") for s in args.seeds.split(",")]
    finished = {run_name(a, s) for a, s in plan if (out_dir / run_name(a, s) / "run_summary.csv").exists()}
    todo = [(a, s) for a, s in plan if run_name(a, s) not in finished]
    ram = available_ram_gb()
    jobs = max(1, min(args.jobs, len(todo) or 1, int(ram // GB_PER_JOB) if ram else args.jobs))
    threads = max(1, (os.cpu_count() or 1) // jobs)
    env = environment(jobs)
    with open(out_dir / "environment.jsonl", "a") as f:
        f.write(json.dumps(env) + "\n")
    print(f"[plan] {len(finished)} finished, {len(todo)} to run, {jobs} parallel "
          f"({threads} CPU thread(s) each; RAM available {ram or float('nan'):.1f} GB) | gpu {env['gpu']}",
          flush=True)

    stop = threading.Event()

    def periodic() -> None:
        while not stop.wait(SYNC_EVERY_S):
            try:
                copied, _ = sync_up(out_dir, drive, finished, MIN_AGE_S)
                print(f"[drive] {time.strftime('%H:%M:%S')} periodic sync: {copied} files", flush=True)
            except Exception as error:
                print(f"[drive] periodic sync failed, will retry: {error}", flush=True)

    syncer = threading.Thread(target=periodic, daemon=True) if drive else None
    if syncer:
        syncer.start()
    started = time.perf_counter()
    failed = []
    pool = ThreadPoolExecutor(jobs)
    try:
        futures = {}
        left = len(todo)
        for arm, seed in todo:
            futures[pool.submit(run_process, arm, seed, args.out, logs, threads)] = (arm, seed)
        for future in as_completed(futures):
            name = run_name(*futures[future])
            code, seconds = future.result()
            with _sync_lock:
                (finished.add(name) if code == 0 else failed.append(name))
            left -= 1
            print(f"[{'done' if code == 0 else 'FAIL'}] {name} in {seconds / 60:.1f} min "
                  f"({(time.perf_counter() - started) / 60:.0f} min elapsed, {left} running/queued)",
                  flush=True)
            if drive:
                try:
                    sync_up(out_dir, drive, finished)
                except Exception as error:
                    print(f"[drive] sync after {name} failed: {error}", flush=True)
        for script, *flags in (("paper_ablations.py", "--collect"), ("paper_physics_checks.py",),
                               ("paper_tables.py",)):
            subprocess.run([sys.executable, str(REPO / "scripts" / script), *flags, "--out", args.out],
                           cwd=REPO)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)   # an interrupt must not start queued runs
        stop.set()
        if syncer:
            syncer.join()
        archive = REPO / "paper_ablations_results.zip"
        if out_dir.exists():
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
                for p in sorted(out_dir.rglob("*")):
                    rel = p.relative_to(out_dir)
                    if p.is_file() and not _skip(rel, finished | {rel.parts[0]}):
                        z.write(p, Path(out_dir.name) / rel)
            print(f"\n[done] {(time.perf_counter() - started) / 60:.0f} min -> {archive}", flush=True)
        if drive:
            try:
                sync_up(out_dir, drive, finished)
                _copy(archive, drive / archive.name)
            except Exception as error:
                print(f"[drive] final sync failed ({error}); the local zip is intact", flush=True)
    if failed:
        sys.exit(f"failed runs (see {logs}): {', '.join(failed)}; rerun the same command to resume them")


if __name__ == "__main__":
    main()
