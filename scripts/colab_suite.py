"""One-command Colab driver for the paper's ablation suite.

Runs every arm in order -- R1 (windowed, causal off), R4 (candidate seeds),
R2 (warm start off), R3 (causal single network), R6 (sequential single
network) -- with results persisted to Google Drive so nothing is lost:

- an incremental background sync copies new/changed run files to Drive every
  10 minutes while arms train,
- every completed arm triggers an immediate sync,
- a final sync + zip happens even on failure or KeyboardInterrupt.

Training happens on Colab's local disk (fast); Drive keeps the checkpoint
copy at /content/drive/MyDrive/lorenz1960-pinn-ablation/. Windowed arms also
checkpoint internally at every window boundary, and the R3 arm checkpoints
every 2,000 Adam steps. On reconnect, finished and interrupted arms resume
from what Drive restored. Optimizer budgets are the candidate's, unchanged.

Intended runtime on a Colab T4 GPU: roughly 2.5-3.5 hours.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/lorenz1960-pinn-ablation")
RUNS = Path("runs/paper_ablations")
SYNC_EVERY_S = 600

STEPS = [
    [sys.executable, "scripts/paper_ablations.py", "--arms", "r1", "--seeds", "1,2,3"],
    [sys.executable, "scripts/paper_ablations.py", "--arms", "r4", "--seeds", "1,2,3,4"],
    [sys.executable, "scripts/paper_ablations.py", "--arms", "r2", "--seeds", "1,2,3"],
    [sys.executable, "scripts/paper_ablations.py", "--arms", "r3", "--seeds", "1,2"],
    [sys.executable, "scripts/sequential_run.py"],
]


def _sync_tree(src: Path, dst: Path) -> tuple[int, int]:
    """Copy only new or changed files (size or mtime differ). Returns (copied, skipped)."""
    copied = skipped = 0
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for name in files:
            s, d = Path(root) / name, dst / rel / name
            if d.exists() and d.stat().st_size == s.stat().st_size \
                    and d.stat().st_mtime >= s.stat().st_mtime - 1:
                skipped += 1
                continue
            shutil.copy2(s, d)
            copied += 1
    return copied, skipped


def sync_to_drive() -> None:
    copied, skipped = _sync_tree(RUNS, DRIVE_ROOT / "paper_ablations")
    print(f"[drive] synced: {copied} files copied, {skipped} unchanged", flush=True)


def _periodic_sync(stop: threading.Event) -> None:
    while not stop.wait(SYNC_EVERY_S):
        try:
            print(f"[drive] periodic sync ({time.strftime('%H:%M:%S')})", flush=True)
            sync_to_drive()
        except Exception as error:
            print(f"[drive] sync failed, will retry: {error}", flush=True)


def ensure_drive() -> bool:
    """True when Drive is usable. Mounting needs the notebook kernel, so the
    auth dialog cannot be raised from inside this subprocess."""
    if not _importable("google.colab"):
        return False
    if os.path.ismount("/content/drive"):
        return True
    from google.colab import drive
    try:
        drive.mount("/content/drive")
        return True
    except Exception:
        print("[drive] mount failed. Run this in a notebook cell first:\n"
              "    from google.colab import drive\n"
              "    drive.mount('/content/drive')\n"
              "then rerun the script.", flush=True)
        return False


def main() -> None:
    import torch
    print(f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
          flush=True)

    on_colab = ensure_drive()
    if on_colab and (DRIVE_ROOT / "paper_ablations").exists():
        _sync_tree(DRIVE_ROOT / "paper_ablations", RUNS)
        print("[drive] previous results restored", flush=True)

    stop = threading.Event()
    syncer = threading.Thread(target=_periodic_sync, args=(stop,), daemon=True)
    if on_colab:
        syncer.start()

    started = time.perf_counter()
    try:
        for cmd in STEPS:
            print(f"\n=== {' '.join(cmd[1:])} "
                  f"({(time.perf_counter() - started) / 60:.0f} min elapsed) ===", flush=True)
            result = subprocess.run(cmd)
            if result.returncode != 0:
                sys.exit(f"step failed: {' '.join(cmd)}")
    finally:
        stop.set()
        syncer.join()
        if on_colab and RUNS.exists():
            sync_to_drive()
        if RUNS.exists():
            shutil.make_archive("paper_ablations_results", "zip", RUNS)
            if on_colab:
                shutil.copy2("paper_ablations_results.zip", DRIVE_ROOT)
            print(f"\n[done] {(time.perf_counter() - started) / 60:.0f} min "
                  f"-> paper_ablations_results.zip", flush=True)


def _importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
