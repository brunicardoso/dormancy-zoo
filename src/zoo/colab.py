"""Colab support (plan reports/2026-11-14-colab-plan.md, approved by the PI 2026-11-14).

Subcommands
  datahash   SHA-256 of every data-cache file -> JSON (laptop: results/data_hashes.json, committed;
             Colab: compared against it before any run; a mismatch stops the batch).
  export     laptop: copy the teacher/reference model folders the Colab batches read into
             export/drive_upload/runs/ (gitignored) with a files.sha256 list, for the PI to upload to Drive.
  bundle     laptop: Drive-only code route (no GitHub): a git bundle of the current commit in
             export/drive_upload/code/, plus a copy of the launcher notebook with that commit pinned.
  import     laptop: copy a queue's small result files from a downloaded Drive results folder into
             results/ (text files <= 2 MB only), then run ``check``.
  queue      Colab: run a queue (YAML list of {module, config}) step by step. Before: copy the data cache,
             the teachers and every finished run of the queue from Drive. After each step: copy each finished
             run folder to Drive, re-hash it there, and write results/colab/<queue>/ (small files + files.sha256),
             mirrored to <drive>/results/ together with results/manifest_colab.csv.
  verify     check files against a files.sha256 list (Colab: the PI's upload on Drive, before any run).
  check      laptop: completeness of a returned queue: every expected run present with status ok,
             curves complete, required files present, hashes of the small files matching files.sha256.

Weights never go to git: only *.csv/*.json/*.yaml files up to 2 MB are copied to results/colab/.
The Colab notebook is a launcher only; it calls this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from zoo.config import REPO_ROOT, load_config
from zoo.runs import RUNS_DIR

DATA_DIR = REPO_ROOT / "data_cache"
HASHES = REPO_ROOT / "results" / "data_hashes.json"
# One-off audit output on the laptop (2026-11-10: training stories identical to validation stories, while
# designing the pools); no code reads it, so Colab neither builds nor checks it.
AUDIT_ONLY = {"val_train_duplicates.npy"}
EXPORT = REPO_ROOT / "export" / "drive_upload"
COLAB_MANIFEST = REPO_ROOT / "results" / "manifest_colab.csv"
NOTEBOOK = REPO_ROOT / "colab" / "phase4b_launch.ipynb"
# Notebook copies written next to the bundle, one per queue the PI may launch (name -> queue).
NOTEBOOK_COPIES = {"launch_G4_distillation.ipynb": "configs/colab/queue_distill_g4.yaml",
                   "launch_A100_equivalence.ipynb": "configs/colab/queue_equiv_a100.yaml"}
SMALL_SUFFIXES = {".csv", ".json", ".yaml"}
SMALL_MAX_BYTES = 2 * 1024 ** 2
BUDGET_ROWS = 1201          # assay curve rows (steps 0..1200)
# Models the Colab batches read (PI 2026-11-14: teachers = the five active models; naive models for the
# sham teachers and the naive controls; 1x models for the equivalence test and the optional 1x-teacher arm).
EXPORT_MODELS = ([f"plant_lr3e-4-mask-f0.100-s{k}" for k in range(5)] +
                 [f"plant_lr3e-4-naive-s{k}" for k in range(5)] +
                 [f"ct4a-plant_lr3e-4-mask-f0.100-s{k}-d01x" for k in range(5)])
EXPORT_FILES = ("final_fp16.safetensors", "final_metrics.csv", "config.yaml", "meta.json")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def data_hashes() -> dict[str, str]:
    return {str(p.relative_to(DATA_DIR)): sha256(p) for p in sorted(DATA_DIR.rglob("*")) if p.is_file()}


# ---------------------------------------------------------------- expected runs

def expected_runs(module: str, config: str, seeds: list[int] | None = None) -> list[str]:
    """Run ids a queue step must produce (assay runs, or distillation runs and their dose models);
    ``seeds`` restricts a distillation step to some of its config's seeds."""
    b = load_config(config)
    if module == "zoo.assay":
        name = b["name"] + ("vehicle" if b.get("vehicle") else "")
        return [f"{name}-{m}-a{a}" for m in b["models"] for a in b["assay_seeds"]]
    if module == "zoo.distill":
        out = []
        for k in (b["seeds"] if seeds is None else seeds):
            rid = f"{b['name']}-s{k}"
            out += [rid] + [f"{rid}-d{m:02d}x" for m in b["distill"]["dose_multiples"]]
        return out
    raise ValueError(f"unknown module {module}")


def required_files(run_id: str, module: str) -> list[str]:
    if module == "zoo.assay":
        return ["meta.json", "config.yaml", "curve.csv", "summary.json"]
    if run_id.rsplit("-", 1)[-1].startswith("d") and run_id.endswith("x"):
        return ["meta.json", "config.yaml", "final_metrics.csv", "diff_from_base_fp32.safetensors"]
    return ["meta.json", "config.yaml", "checks.csv", "summary.json", "data.csv", "loss.csv"]


def is_finished(run_dir: Path) -> bool:
    m = run_dir / "meta.json"
    return m.exists() and json.loads(m.read_text()).get("status") == "ok"


def load_queue(path: str) -> list[dict[str, str]]:
    q = yaml.safe_load(Path(path).read_text())
    for s in q["steps"]:
        if s["module"] not in ("zoo.assay", "zoo.distill"):
            raise ValueError(f"queue step module {s['module']} not allowed")
        if "seeds" in s and (s["module"] != "zoo.distill" or not set(s["seeds"]) <= set(load_config(s["config"])["seeds"])):
            raise ValueError(f"step seeds {s['seeds']} not allowed for {s['config']}")
    return q["steps"]


def step_runs(s: dict[str, Any]) -> list[str]:
    return expected_runs(s["module"], s["config"], s.get("seeds"))


# ---------------------------------------------------------------- Colab side

def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def file_list(run_dir: Path, rel_to: Path) -> list[dict[str, Any]]:
    return [{"path": str(p.relative_to(rel_to)), "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(run_dir.rglob("*")) if p.is_file()]


def sync_out(run_ids: list[str], drive: Path, qname: str) -> list[dict[str, Any]]:
    """Copy finished runs to Drive, verify the copies by hash, and collect small files for git."""
    out_dir = REPO_ROOT / "results" / "colab" / qname
    entries = []
    for rid in run_ids:
        local = RUNS_DIR / rid
        if not is_finished(local):
            continue
        remote = drive / "runs" / rid
        files = file_list(local, RUNS_DIR)
        if not (remote.exists() and is_finished(remote) and file_list(remote, drive / "runs") == files):
            copy_tree(local, remote)
            if file_list(remote, drive / "runs") != files:
                raise RuntimeError(f"{rid}: Drive copy does not match the local run (hash mismatch)")
        entries += files
        for f in files:
            p = RUNS_DIR / f["path"]
            if p.suffix in SMALL_SUFFIXES and f["bytes"] <= SMALL_MAX_BYTES:
                dst = out_dir / "runs" / f["path"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "files.sha256", "w") as fh:
        for e in entries:
            fh.write(f"{e['sha256']}  {e['bytes']}  {e['path']}\n")
    # mirror the small files to Drive, so results return without GitHub (Drive-only route)
    shutil.copytree(out_dir, drive / "results" / "colab" / qname, dirs_exist_ok=True)
    if COLAB_MANIFEST.exists():
        shutil.copy2(COLAB_MANIFEST, drive / "results" / COLAB_MANIFEST.name)
    return entries


def cache_problems(want: dict[str, str], got: dict[str, str]) -> list[str]:
    """Files of the laptop's data cache that are missing here or differ (audit-only files ignored)."""
    return sorted(f"{k}: {'missing' if k not in got else 'differs'}" for k in want if got.get(k) != want[k])


def build_data_cache() -> None:
    """Build what the runs read: the pools (zoo.pools) and the tokenized train and validation splits,
    which the laptop built lazily on first use (zoo.data.tokenize_split)."""
    subprocess.run([sys.executable, "-m", "zoo.pools", "--config", "configs/pools.yaml"], cwd=REPO_ROOT, check=True)
    from zoo.data import tokenize_split
    from zoo.model import load_tokenizer
    cfg = load_config(str(REPO_ROOT / "configs" / "pools.yaml"))
    tok = load_tokenizer(cfg)
    for split in ("validation", "train"):
        print(f"tokenizing the {split} split", flush=True)
        tokenize_split(cfg, split, tok)


def run_queue(qpath: str, drive: Path) -> None:
    if os.environ.get("ZOO_ENV") != "colab":
        raise RuntimeError("queue runs on Colab only (ZOO_ENV=colab)")
    qname = Path(qpath).stem
    steps = load_queue(qpath)
    from zoo.gpucheck import check as gpu_check
    print(f"GPU: {gpu_check(qpath, str(drive))} (allowed and validated for this queue)", flush=True)
    # data cache: from Drive if present and matching, else built here (pools + tokenized splits),
    # checked file by file against the laptop's hashes, and saved to Drive
    want = {k: v for k, v in json.loads(HASHES.read_text()).items() if k not in AUDIT_ONLY}
    if cache_problems(want, data_hashes()) and (drive / "data_cache").exists():
        shutil.copytree(drive / "data_cache", DATA_DIR, dirs_exist_ok=True)
    if cache_problems(want, data_hashes()):
        build_data_cache()
        probs = cache_problems(want, data_hashes())
        if probs:
            raise RuntimeError(f"data cache differs from the laptop's: {probs}")
        copy_tree(DATA_DIR, drive / "data_cache")
    print(f"data cache: all {len(want)} files match the laptop's hashes", flush=True)
    # teachers and earlier finished runs of this queue
    expected = [r for s in steps for r in step_runs(s)]
    (drive / "runs").mkdir(parents=True, exist_ok=True)
    # a resumed session starts from a fresh clone: restore the Colab manifest so rows are not lost
    if (drive / "results" / COLAB_MANIFEST.name).exists() and not COLAB_MANIFEST.exists():
        shutil.copy2(drive / "results" / COLAB_MANIFEST.name, COLAB_MANIFEST)
    for p in sorted((drive / "runs").iterdir()):
        if (p.name in EXPORT_MODELS or p.name in expected) and is_finished(p) and not (RUNS_DIR / p.name).exists():
            shutil.copytree(p, RUNS_DIR / p.name)
    for i, s in enumerate(steps):
        ids = step_runs(s)
        label = s["config"] + (f" seeds {s['seeds']}" if "seeds" in s else "")
        if all(is_finished(RUNS_DIR / r) for r in ids):
            print(f"[{i + 1}/{len(steps)}] {label}: all {len(ids)} runs finished, skipping")
            continue
        print(f"[{i + 1}/{len(steps)}] {s['module']} --config {label}", flush=True)
        cmd = [sys.executable, "-m", s["module"], "--config", s["config"]]
        if "seeds" in s:
            cmd += ["--seeds", ",".join(map(str, s["seeds"]))]
        t0 = time.time()
        try:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        finally:
            sync_out(expected, drive, qname)    # finished runs reach Drive even if this step failed
        print(f"[{i + 1}/{len(steps)}] finished in {time.time() - t0:.0f} s", flush=True)
        if s["module"] == "zoo.distill":        # PI condition 2: training time per distillation run
            for r in ids:
                sm = RUNS_DIR / r / "summary.json"
                if sm.exists():
                    print(f"    TIMING {r}: {json.loads(sm.read_text())['seconds']:.0f} s of training "
                          f"(send this line to the agent)", flush=True)
    entries = sync_out(expected, drive, qname)
    missing = [r for r in expected if not is_finished(RUNS_DIR / r)]
    print(f"queue {qname}: {len(expected) - len(missing)}/{len(expected)} runs finished; "
          f"{len(entries)} files on Drive, {sum(e['bytes'] for e in entries) / 1e9:.2f} GB")
    if missing:
        raise RuntimeError(f"unfinished runs: {missing}")


# ---------------------------------------------------------------- laptop side

def export_teachers() -> None:
    dst = EXPORT / "runs"
    entries = []
    for m in EXPORT_MODELS:
        (dst / m).mkdir(parents=True, exist_ok=True)
        for f in EXPORT_FILES:
            shutil.copy2(RUNS_DIR / m / f, dst / m / f)
        entries += file_list(dst / m, dst)
    with open(EXPORT / "upload_files.sha256", "w") as fh:   # uploaded as-is to the Drive folder root
        for e in entries:
            fh.write(f"{e['sha256']}  {e['bytes']}  {e['path']}\n")
    print(f"{len(EXPORT_MODELS)} models, {len(entries)} files, {sum(e['bytes'] for e in entries) / 1e6:.0f} MB -> {EXPORT}")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def make_bundle() -> str:
    """Drive-only code route: bundle the current branch (full history, committed files only) and write one
    copy of the launcher notebook per queue in NOTEBOOK_COPIES, with CODE_SOURCE='drive', this commit and
    the queue filled in. Returns the commit."""
    sha, branch = _git("rev-parse", "HEAD"), _git("rev-parse", "--abbrev-ref", "HEAD")
    code = EXPORT / "code"
    code.mkdir(parents=True, exist_ok=True)
    bundle = code / "dormancy_zoo.bundle"
    _git("bundle", "create", str(bundle), "HEAD", branch)
    _git("bundle", "verify", str(bundle))
    (code / "COMMIT.txt").write_text(sha + "\n")
    for name, queue in NOTEBOOK_COPIES.items():
        if not (REPO_ROOT / queue).exists():
            raise FileNotFoundError(queue)
        nb = json.loads(NOTEBOOK.read_text())
        src = "".join(nb["cells"][1]["source"])
        for key, val in (("CODE_SOURCE", "drive"), ("COMMIT", sha), ("QUEUE", queue)):
            lines = [ln for ln in src.splitlines() if ln.startswith(f"{key} = ")]
            if len(lines) != 1:
                raise RuntimeError(f"notebook parameter {key} not found exactly once")
            rest = lines[0].split("#", 1)[1] if "#" in lines[0] else ""
            src = src.replace(lines[0], f'{key} = "{val}"' + (f"  #{rest}" if rest else ""))
        nb["cells"][1]["source"] = src.splitlines(keepends=True)
        (EXPORT / name).write_text(json.dumps(nb, indent=1) + "\n")
    print(f"bundle of {branch} at {sha} ({bundle.stat().st_size / 1e6:.1f} MB) and notebooks {list(NOTEBOOK_COPIES)} -> {EXPORT}")
    return sha


def small_result_files(src: Path) -> list[Path]:
    """Files under ``src``; refuses anything that is not a small text result (no weights, no links)."""
    files = sorted(p for p in src.rglob("*") if p.is_file() or p.is_symlink())
    bad = [str(p) for p in files if p.is_symlink() or p.suffix not in SMALL_SUFFIXES | {".sha256"}
           or p.stat().st_size > SMALL_MAX_BYTES]
    if bad:
        raise ValueError(f"refusing to import non-result files: {bad[:10]}")
    return files


def import_results(qpath: str, src: Path) -> list[str]:
    """Copy <src>/colab/<queue>/ and <src>/manifest_colab.csv (``src`` = a downloaded copy of the Drive
    folder dormancy_zoo/results) into results/, overwriting older copies, then return check_queue()."""
    qname = Path(qpath).stem
    files = small_result_files(src / "colab" / qname)
    dst = REPO_ROOT / "results" / "colab" / qname
    for p in files:
        out = dst / p.relative_to(src / "colab" / qname)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
    man = src / COLAB_MANIFEST.name
    if man.exists():
        if man.is_symlink() or man.stat().st_size > SMALL_MAX_BYTES:
            raise ValueError(f"refusing {man}")
        new_rows = sum(1 for _ in open(man))
        old_rows = sum(1 for _ in open(COLAB_MANIFEST)) if COLAB_MANIFEST.exists() else 0
        if new_rows < old_rows:
            raise RuntimeError(f"{man} has fewer rows ({new_rows}) than {COLAB_MANIFEST} ({old_rows}); not copied")
        shutil.copy2(man, COLAB_MANIFEST)
    print(f"imported {len(files)} files into {dst}")
    return check_queue(qpath)


def check_queue(qpath: str, runs_root: Path | None = None) -> list[str]:
    """Problems with a returned queue (empty list = complete). Small files are checked in
    results/colab/<queue>/runs; with ``runs_root`` (downloaded run folders), every file is checked."""
    qname = Path(qpath).stem
    root = REPO_ROOT / "results" / "colab" / qname
    listed = {}
    for line in (root / "files.sha256").read_text().splitlines():
        h, n, p = line.split("  ", 2)
        listed[p] = (h, int(n))
    problems = []
    for s in load_queue(qpath):
        for rid in step_runs(s):
            for f in required_files(rid, s["module"]):
                if f"{rid}/{f}" not in listed:
                    problems.append(f"{rid}: {f} not in files.sha256")
            meta = root / "runs" / rid / "meta.json"
            if not meta.exists() or json.loads(meta.read_text()).get("status") != "ok":
                problems.append(f"{rid}: not finished ok")
            if s["module"] == "zoo.assay" and (root / "runs" / rid / "curve.csv").exists():
                n = sum(1 for _ in open(root / "runs" / rid / "curve.csv")) - 1
                if n != BUDGET_ROWS:
                    problems.append(f"{rid}: curve has {n} rows, expected {BUDGET_ROWS}")
    for p, (h, n) in listed.items():
        for base in [root / "runs"] + ([runs_root] if runs_root else []):
            f = base / p
            if f.exists() and (f.stat().st_size != n or sha256(f) != h):
                problems.append(f"{p}: hash or size differs from files.sha256 ({base})")
    return problems


def verify_list(listing: Path, root: Path) -> list[str]:
    """Files of a files.sha256 list that are missing under ``root`` or differ in size or hash."""
    probs = []
    for line in listing.read_text().splitlines():
        h, n, p = line.split("  ", 2)
        f = root / p
        if not f.exists():
            probs.append(f"{p}: missing")
        elif f.stat().st_size != int(n) or sha256(f) != h:
            probs.append(f"{p}: size or hash differs")
    return probs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("datahash")
    sub.add_parser("export")
    sub.add_parser("bundle")
    im = sub.add_parser("import"); im.add_argument("queue"); im.add_argument("--from", dest="src", required=True)
    q = sub.add_parser("queue"); q.add_argument("queue"); q.add_argument("--drive", required=True)
    c = sub.add_parser("check"); c.add_argument("queue"); c.add_argument("--runs-root", default=None)
    v = sub.add_parser("verify"); v.add_argument("listing"); v.add_argument("--root", required=True)
    a = ap.parse_args()
    if a.cmd == "datahash":
        HASHES.write_text(json.dumps(data_hashes(), indent=1) + "\n")
        print(f"wrote {HASHES}")
    elif a.cmd == "export":
        export_teachers()
    elif a.cmd == "bundle":
        make_bundle()
    elif a.cmd == "import":
        probs = import_results(a.queue, Path(a.src))
        print("\n".join(probs) if probs else f"{a.queue}: complete")
        sys.exit(1 if probs else 0)
    elif a.cmd == "verify":
        probs = verify_list(Path(a.listing), Path(a.root))
        print("\n".join(probs) if probs else f"{a.listing}: all files present and matching")
        sys.exit(1 if probs else 0)
    elif a.cmd == "queue":
        run_queue(a.queue, Path(a.drive))
    else:
        probs = check_queue(a.queue, Path(a.runs_root) if a.runs_root else None)
        print("\n".join(probs) if probs else f"{a.queue}: complete")
        sys.exit(1 if probs else 0)


if __name__ == "__main__":
    main()
