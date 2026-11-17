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
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import yaml

from zoo.config import REPO_ROOT, config_hash

RUNS_DIR = REPO_ROOT / "runs"
RESULTS_DIR = REPO_ROOT / "results"
# Environment (PI 2026-11-14): no comparison mixes environments. ZOO_ENV is "laptop" unless set
# (the Colab launcher sets "colab"); Colab runs go to their own manifest so the two never conflict.
ENV = os.environ.get("ZOO_ENV", "laptop")
MANIFEST = RESULTS_DIR / ("manifest.csv" if ENV == "laptop" else f"manifest_{ENV}.csv")
DISK_TRACKED_DIRS = ("runs", "results", "data_cache")
DISK_LIMIT_BYTES = 60 * 1024**3   # PI 2026-11-14 (was 30); warn the PI before a batch that would pass 50 GB

# PI 2026-11-17: no run starts on uncommitted code under src/, configs/ or tests/. The one exception is a smoke
# test: a run id containing "smoke", started with ZOO_SMOKE_ALLOW_DIRTY=1. Such runs carry SMOKE_OVERRIDE_NOTE in
# the manifest and "dirty_smoke_override": true in meta.json, and are excluded from every analysis.
DIRTY_OVERRIDE_ENV = "ZOO_SMOKE_ALLOW_DIRTY"
SMOKE_OVERRIDE_NOTE = "DIRTY-CODE SMOKE OVERRIDE (PI 2026-11-17): uncommitted code; excluded from analyses"

MANIFEST_FIELDS = [
    "run_id", "run_type", "status", "reason", "seed", "config_path", "config_hash",
    "git_commit", "git_dirty", "started", "finished", "run_dir",
    "run_disk_bytes", "project_disk_bytes", "notes", "env",
]


def env_string() -> str:
    """Environment label and versions for the manifest's env column, e.g.
    ``laptop; torch 2.14.1; cuda 12.9; gpu NVIDIA ...; driver 581.57; pixi env default``.
    Analyses compare env_key(): the label plus the GPU model."""
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    except Exception:
        drv = "unknown"
    return (f"{ENV}; torch {torch.__version__}; cuda {torch.version.cuda}; gpu {gpu}; driver {drv}; "
            f"pixi env {os.environ.get('PIXI_ENVIRONMENT_NAME', 'unknown')}")


def env_label(env: str | float | None) -> str:
    """The environment label of a manifest env value; blank (rows before the column existed) = laptop."""
    if not isinstance(env, str) or not env.strip():
        return "laptop"
    return env.split(";")[0].strip()


def env_key(env: str | float | None) -> str:
    """What comparisons must share (PI 2026-11-14: no comparison mixes GPUs or mixes Colab with the laptop):
    the environment label plus the GPU model, e.g. ``colab | NVIDIA RTX PRO 6000 ...``. Blank env = laptop rows
    from before the column existed, all on the laptop GPU."""
    label = env_label(env)
    if not isinstance(env, str) or not env.strip():
        return "laptop | laptop GPU"
    gpu = next((f.strip()[4:] for f in env.split(";") if f.strip().startswith("gpu ")), "unknown")
    return f"{label} | {gpu}"


def require_one_env(envs) -> str:
    """The single env_key shared by all ``envs``; raises if a comparison would mix environments or GPUs."""
    keys = {env_key(e) for e in envs}
    if len(keys) != 1:
        raise ValueError(f"comparison mixes environments or GPUs: {sorted(keys)}")
    return keys.pop()


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


def dirty_override(run_id: str, dirty: bool) -> bool:
    """Whether a run on uncommitted code may start (PI 2026-11-17). False on a clean tree; True for a smoke test
    (run id containing "smoke") with ZOO_SMOKE_ALLOW_DIRTY=1; otherwise raises, so the run never starts."""
    if not dirty:
        return False
    if os.environ.get(DIRTY_OVERRIDE_ENV) != "1":
        raise RuntimeError(f"{run_id}: uncommitted changes under src/, configs/ or tests/. Commit them first "
                           f"(PI 2026-11-17); only smoke tests may override, with {DIRTY_OVERRIDE_ENV}=1")
    if "smoke" not in run_id:
        raise RuntimeError(f"{run_id}: {DIRTY_OVERRIDE_ENV}=1 is for smoke tests only (run id must contain 'smoke')")
    return True


def is_smoke_override(manifest) -> "pd.Series":
    """Rows of a manifest DataFrame that ran under the dirty-code smoke override (excluded from analyses)."""
    return manifest["notes"].fillna("").astype(str).str.contains(SMOKE_OVERRIDE_NOTE, regex=False)


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
        self.override = None          # set by start() (dirty_override); finish() checks again if start was skipped

    def start(self, allow_existing: bool = False) -> "Run":
        """Create the run folder and write config and metadata. Refuses on uncommitted code (dirty_override)."""
        self.override = dirty_override(self.run_id, self.dirty)
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
            "env": env_string(),
        }
        if self.override:
            meta["dirty_smoke_override"] = True
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return self

    def finish(self, status: str = "ok", reason: str = "", notes: str = "") -> None:
        """Append this run to the manifest. Status is 'ok' or 'failed'."""
        if self.override is None:
            self.override = dirty_override(self.run_id, self.dirty)
        if self.override:
            notes = f"{SMOKE_OVERRIDE_NOTE}; {notes}" if notes else SMOKE_OVERRIDE_NOTE
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
            "notes": notes, "env": env_string(),
        })
        if total > DISK_LIMIT_BYTES:
            raise RuntimeError(f"project disk {total / 1024**3:.1f} GB exceeds 60 GB: stop and ask the PI")


def append_manifest(row: dict[str, Any]) -> None:
    """Append one row to the manifest, writing the header if new. An existing file keeps its own
    header (columns added later, such as ``env``, are written once the file is migrated)."""
    import fcntl
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(MANIFEST.with_name(MANIFEST.name + ".lock"), "w") as lock:   # parallel processes append safely
        fcntl.flock(lock, fcntl.LOCK_EX)
        new = not MANIFEST.exists()
        fields = MANIFEST_FIELDS
        if not new:
            with open(MANIFEST, newline="") as f:
                fields = next(csv.reader(f))
        with open(MANIFEST, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fields})


def make_run_id(name: str, suffix: str = "") -> str:
    """Run id: ``YYYYMMDD-HHMMSS-<name>[-suffix]``."""
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{name}" + (f"-{suffix}" if suffix else "")
