"""Run folders and the results manifest.

Every run gets ``runs/<run_id>/`` containing ``config.yaml`` (resolved),
``meta.json`` (seed, git commit, config hash, versions, timing) and whatever
metrics it writes. Every run is also a row in ``results/manifest.csv``; rows
are never deleted, failed runs are marked with a reason.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import yaml

from zoo.config import REPO_ROOT, config_hash

RUNS_DIR = REPO_ROOT / "runs"
RESULTS_DIR = REPO_ROOT / "results"
MANIFEST = RESULTS_DIR / "manifest.csv"
DISK_TRACKED_DIRS = ("runs", "results", "data_cache")
DISK_LIMIT_BYTES = 30 * 1024**3

MANIFEST_FIELDS = [
    "run_id", "run_type", "status", "reason", "seed", "config_path", "config_hash",
    "git_commit", "git_dirty", "started", "finished", "run_dir",
    "run_disk_bytes", "project_disk_bytes", "notes",
]


def git_info() -> tuple[str, bool]:
    """Return (commit hash, working tree dirty?) for the repo."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        commit = "no-commit"
    status = subprocess.run(["git", "status", "--porcelain", "--", "src", "configs", "tests"],
                            cwd=REPO_ROOT, capture_output=True, text=True).stdout
    return commit, bool(status.strip())


def dir_bytes(path: Path) -> int:
    """Total size of files under ``path`` (0 if missing)."""
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def project_disk_bytes() -> int:
    """Disk used by project outputs (runs, results, data cache)."""
    return sum(dir_bytes(REPO_ROOT / d) for d in DISK_TRACKED_DIRS)


class Run:
    """A single run: creates the folder on start, writes the manifest row on finish."""

    def __init__(self, run_id: str, run_type: str, cfg: dict[str, Any],
                 config_path: str, seed: int | None):
        self.run_id = run_id
        self.run_type = run_type
        self.cfg = cfg
        self.config_path = str(config_path)
        self.seed = seed
        self.dir = RUNS_DIR / run_id
        self.started = dt.datetime.now().isoformat(timespec="seconds")
        self.commit, self.dirty = git_info()
        self.hash = config_hash(cfg)

    def start(self, allow_existing: bool = False) -> "Run":
        """Create the run folder and write config and metadata."""
        if self.dir.exists() and not allow_existing:
            raise FileExistsError(f"run folder exists: {self.dir}")
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "config.yaml", "w") as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False)
        import torch
        import transformers
        meta = {
            "run_id": self.run_id, "run_type": self.run_type, "seed": self.seed,
            "config_path": self.config_path, "config_hash": self.hash,
            "git_commit": self.commit, "git_dirty": self.dirty, "started": self.started,
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return self

    def finish(self, status: str = "ok", reason: str = "", notes: str = "") -> None:
        """Append this run to the manifest. Status is 'ok' or 'failed'."""
        finished = dt.datetime.now().isoformat(timespec="seconds")
        meta_path = self.dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta.update({"finished": finished, "status": status, "reason": reason})
            meta_path.write_text(json.dumps(meta, indent=2))
        total = project_disk_bytes()
        append_manifest({
            "run_id": self.run_id, "run_type": self.run_type, "status": status,
            "reason": reason, "seed": self.seed, "config_path": self.config_path,
            "config_hash": self.hash, "git_commit": self.commit, "git_dirty": self.dirty,
            "started": self.started, "finished": finished,
            "run_dir": str(self.dir.relative_to(REPO_ROOT)),
            "run_disk_bytes": dir_bytes(self.dir), "project_disk_bytes": total,
            "notes": notes,
        })
        if total > DISK_LIMIT_BYTES:
            raise RuntimeError(f"project disk {total / 1024**3:.1f} GB exceeds 30 GB: stop and ask the PI")


def append_manifest(row: dict[str, Any]) -> None:
    """Append one row to results/manifest.csv, writing the header if new."""
    RESULTS_DIR.mkdir(exist_ok=True)
    new = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def make_run_id(name: str, suffix: str = "") -> str:
    """Run id: ``YYYYMMDD-HHMMSS-<name>[-suffix]``."""
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{name}" + (f"-{suffix}" if suffix else "")
