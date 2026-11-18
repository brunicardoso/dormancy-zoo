"""GPU guard for Colab batches (PI 2026-11-14): a queue runs only on a GPU it allows, and only if that GPU is
validated (configs/colab/validated_gpus.yaml) or, for an equivalence-test queue, an equivalence candidate.
A queue also never changes GPU: the GPU of its first start is recorded on Drive and checked on every resume.

Standalone on purpose (only PyYAML and nvidia-smi), so the launcher notebook can run it with Colab's system
Python before the pixi environment exists:  python3 -I src/zoo/gpucheck.py QUEUE [--drive DIR]
Any-GPU launcher: python3 -I src/zoo/gpucheck.py --pick  prints the queue for this GPU (configs/colab/gpu_queues.yaml);
the check above then runs on that queue as usual.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
LIST = REPO / "configs" / "colab" / "validated_gpus.yaml"
QUEUE_MAP = REPO / "configs" / "colab" / "gpu_queues.yaml"


def gpu_name() -> str:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()
    if len(out) != 1:
        raise RuntimeError(f"expected exactly one GPU, found {out}")
    return out[0].strip()


GPU_MODULES = ("zoo.assay", "zoo.distill", "zoo.base_folder")


def any_gpu_problems(queue: dict, validated: set[str]) -> list[str]:
    """An ``any_gpu`` queue (PI 2026-11-17) may start on any GPU because every step that uses the GPU names the
    validated GPU it needs (``requires_gpu``) and is skipped on any other; the remaining steps do not depend on it."""
    out = []
    for s in queue.get("steps") or []:
        req = s.get("requires_gpu") or []
        if s.get("module") in GPU_MODULES and not req:
            out.append(f"step {s.get('config')} uses the GPU but names no requires_gpu")
        out += [f"step {s.get('config')} requires {g!r}, which is not validated" for g in req if g not in validated]
    return out


def problems(gpu: str, queue: dict, gpu_list: dict, locked: str | None = None) -> list[str]:
    """Why ``gpu`` may not run ``queue`` (empty list = allowed)."""
    validated = {g["gpu"] for g in gpu_list.get("validated") or []}
    if queue.get("any_gpu"):
        return any_gpu_problems(queue, validated)
    candidates = {g["gpu"] for g in gpu_list.get("equivalence_candidates") or []}
    allowed = set(queue.get("gpus") or [])
    out = []
    if not allowed:
        out.append("the queue names no GPUs (key `gpus`)")
    elif gpu not in allowed:
        out.append(f"GPU {gpu!r} is not one this queue allows: {sorted(allowed)}")
    if gpu not in validated and not (queue.get("equivalence_test") and gpu in candidates):
        out.append(f"GPU {gpu!r} is not validated" + (" (only equivalence-test queues may run on candidates)"
                                                       if gpu in candidates else ""))
    if locked is not None and gpu != locked:
        out.append(f"this queue started on {locked!r}; it may not resume on {gpu!r} (no comparison mixes GPUs)")
    return out


def check(queue_path: str, drive: str | None = None, gpu: str | None = None) -> str:
    """Raise unless this GPU may run the queue; with ``drive``, record the GPU on first start (lock)."""
    gpu = gpu or gpu_name()
    queue = yaml.safe_load(Path(queue_path).read_text())
    gpu_list = yaml.safe_load(LIST.read_text())
    # a per-seed queue (PI 2026-11-16) locks each seed to a GPU model (zoo.colab.run_perseed), not the whole queue
    per_seed = bool(queue.get("per_seed"))
    unlocked = per_seed or bool(queue.get("any_gpu"))      # any_gpu: its GPU steps are gated per step instead
    lock = Path(drive) / "results" / "colab" / Path(queue_path).stem / "gpu.txt" if drive and not unlocked else None
    locked = lock.read_text().strip() if lock is not None and lock.exists() else None
    probs = problems(gpu, queue, gpu_list, locked)
    if probs:
        raise SystemExit("GPU check failed:\n  " + "\n  ".join(probs))
    if lock is not None and locked is None:
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(gpu + "\n")
    return gpu


def pick_queue(gpu: str, mapping: dict) -> str:
    """The queue the any-GPU launcher runs on ``gpu`` (configs/colab/gpu_queues.yaml)."""
    queues = mapping.get("queues") or {}
    if gpu in queues:
        return queues[gpu]
    if mapping.get("default"):   # 2026-11-17: an any_gpu queue for every other GPU
        return mapping["default"]
    raise SystemExit(f"no queue for GPU {gpu!r}; GPUs with a queue: {sorted(queues)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("queue", nargs="?")
    ap.add_argument("--drive", default=None)
    ap.add_argument("--pick", action="store_true", help="print the queue for this GPU and exit")
    a = ap.parse_args()
    if a.pick:
        gpu = gpu_name()
        print(f"GPU: {gpu}")
        print(pick_queue(gpu, yaml.safe_load(QUEUE_MAP.read_text())))
        return
    if not a.queue:
        ap.error("give a queue, or --pick")
    gpu = check(a.queue, a.drive)
    print(f"GPU check passed: {gpu} may run {a.queue}")


if __name__ == "__main__":
    sys.exit(main())
