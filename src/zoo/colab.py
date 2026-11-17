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
# 2026-11-16 (PI: the GPU Colab assigns varies): one notebook for every session, which runs what the assigned GPU may
# run (per-seed distillation queue on a validated GPU, equivalence test on a candidate). 2026-11-16 (evening): the 80 GB
# test is complete and that card validated, so its notebook copy is no longer written.
NOTEBOOK_COPIES = {"launch_any_GPU.ipynb": "auto"}   # "auto": queue picked per GPU (configs/colab/gpu_queues.yaml)
SMALL_SUFFIXES = {".csv", ".json", ".yaml"}
SMALL_MAX_BYTES = 2 * 1024 ** 2
BUDGET_ROWS = 1201          # assay curve rows (steps 0..1200)
VALIDATED_GPUS = REPO_ROOT / "configs" / "colab" / "validated_gpus.yaml"
SYNC_EVERY_S = 120          # Drive sync interval while a queue step runs (PI 2026-11-16)
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
    ``seeds`` restricts a step to some training seeds (an assay step: the models of those seeds)."""
    b = load_config(config)
    if module == "zoo.assay":
        from zoo.seed_env import seed_of
        name = b["name"] + ("vehicle" if b.get("vehicle") else "")
        return [f"{name}-{m}-a{a}" for m in b["models"] for a in b["assay_seeds"]
                if seeds is None or seed_of(m) in seeds]
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


# ---------------------------------------------------------------- per-seed queues (PI 2026-11-16)
# Every run of one training seed (student, sham student, their assays, in-batch controls, 1x-teacher arm) is on a
# single GPU model. A per-seed queue lists, per GPU family, the steps one seed needs and the order in which that
# family takes seeds (fixed 2026-11-16, before any further result: G4 from seed 0 upward, A100 from seed 4
# downward). Each seed is locked to the GPU model that first ran it (<drive>/results/colab/<queue>/locks.json).
# A family listed in another family's ``claims_from`` may claim a seed locked there that is not complete; the claim
# is recorded in claims.csv and the seed's runs from the other family become superseded (kept, never analysed).


def is_perseed(path: str) -> bool:
    p = Path(path)
    return p.exists() and bool((yaml.safe_load(p.read_text()) or {}).get("per_seed"))


def load_perseed(path: str) -> dict[str, Any]:
    q = yaml.safe_load(Path(path).read_text())
    if not q.get("per_seed"):
        raise ValueError(f"{path} is not a per-seed queue")
    seeds = sorted(q["seeds"])
    for fam, f in q["families"].items():
        if sorted(f["claim_order"]) != seeds:
            raise ValueError(f"{fam}: claim_order must list every seed once")
        for st in f["steps"]:
            if st["module"] not in ("zoo.assay", "zoo.distill"):
                raise ValueError(f"queue step module {st['module']} not allowed")
        for other in f.get("claims_from") or []:
            if other not in q["families"]:
                raise ValueError(f"{fam}: claims_from names unknown family {other}")
    for gpu, steps in (q.get("gpu_first_steps") or {}).items():
        for st in steps:
            if st["module"] not in ("zoo.assay", "zoo.distill"):
                raise ValueError(f"first step module {st['module']} not allowed")
            if int(st.get("parallel", 1)) > 1 and st["module"] != "zoo.assay":
                raise ValueError("parallel is allowed for zoo.assay steps only")
    return q


def model_key(gpu: str, gpu_list: dict | None = None) -> str:
    """The GPU model a lock records. A validated GPU with ``same_model_as`` (PI approval needed) counts as that
    model; every other name is its own model."""
    gpu_list = gpu_list if gpu_list is not None else yaml.safe_load(VALIDATED_GPUS.read_text())
    for g in gpu_list.get("validated") or []:
        if g["gpu"] == gpu and g.get("same_model_as"):
            return g["same_model_as"]
    return gpu


def family_of(gpu: str, q: dict[str, Any]) -> str | None:
    fams = [fam for fam, f in q["families"].items() if gpu in f["gpus"]]
    return fams[0] if fams else None


def seed_steps(q: dict[str, Any], fam: str, k: int) -> list[tuple[dict[str, Any], list[str]]]:
    """(step, run ids) for seed ``k`` under family ``fam``, in queue order."""
    return [(st, expected_runs(st["module"], st["config"], [k])) for st in q["families"][fam]["steps"]]


def seed_expected(q: dict[str, Any], fam: str, k: int) -> list[str]:
    return [r for _, ids in seed_steps(q, fam, k) for r in ids]


def next_seed(gpu_key: str, fam: str, q: dict[str, Any], locks: dict[int, str],
              complete) -> tuple[int, bool] | None:
    """The seed this GPU works on next, and whether that takes a new lock or claim (None = nothing to do).
    ``locks``: seed -> GPU model key; ``complete(fam, k)``: all of seed k's runs under family fam finished.
    First its own unfinished seeds, in the family's order; then the next seed in that order that is unlocked,
    or locked to a family it may claim from and not complete there."""
    order = q["families"][fam]["claim_order"]
    claims_from = set(q["families"][fam].get("claims_from") or [])
    for k in order:
        if locks.get(k) == gpu_key and not complete(fam, k):
            return k, False
    for k in order:
        lk = locks.get(k)
        if lk is None:
            return k, True
        if lk == gpu_key:
            continue
        lfam = family_of(lk, q)
        if lfam in claims_from and not complete(lfam, k):
            return k, True
    return None


def read_locks(path: Path) -> dict[int, str]:
    return {int(k): v for k, v in json.loads(path.read_text()).items()} if path.exists() else {}


def write_locks(path: Path, locks: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({str(k): v for k, v in sorted(locks.items())}, indent=1) + "\n")
    tmp.replace(path)


def derive_locks(q: dict[str, Any], locks: dict[int, str], runs_root: Path) -> dict[int, str]:
    """Locks for seeds that have none yet but already have finished runs (e.g. the G4 queue's runs before the
    per-seed queue existed): the GPU recorded in those runs. Raises if a seed's existing runs mix GPU models."""
    out = dict(locks)
    for k in q["seeds"]:
        if k in out:
            continue
        gpus = set()
        for fam in q["families"]:
            for rid in seed_expected(q, fam, k):
                meta = runs_root / rid / "meta.json"
                if is_finished(runs_root / rid):
                    gpus.add(model_key(json.loads(meta.read_text())["gpu"]))
        if len(gpus) > 1:
            raise RuntimeError(f"seed {k}: existing runs on several GPU models {sorted(gpus)}")
        if gpus:
            out[k] = gpus.pop()
    return out


def append_claim(path: Path, row: dict[str, Any]) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        if new:
            fh.write("seed,claimed_by,superseded_gpu,superseded_family,claimed_at\n")
        fh.write(f"{row['seed']},{row['claimed_by']},{row['superseded_gpu']},{row['superseded_family']},"
                 f"{row['claimed_at']}\n")


def superseded_runs(q: dict[str, Any], claims: list[dict[str, Any]], known: set[str]) -> list[dict[str, Any]]:
    """Runs made superseded by claims: every run of the claimed seed under the superseded family that exists
    (``known``: run ids in the Colab manifest)."""
    out = []
    for c in claims:
        fam, k = c["superseded_family"], int(c["seed"])
        if not fam or fam not in q["families"]:
            continue
        for rid in seed_expected(q, fam, k):
            if rid in known:
                out.append({"run_id": rid, "seed": k, "gpu": c["superseded_gpu"], "family": fam,
                            "claimed_by": c["claimed_by"], "claimed_at": c["claimed_at"]})
    return out


# ---------------------------------------------------------------- Colab side

def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def file_list(run_dir: Path, rel_to: Path) -> list[dict[str, Any]]:
    return [{"path": str(p.relative_to(rel_to)), "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(run_dir.rglob("*")) if p.is_file()]


class DriveSync:
    """Copies a queue's runs to Drive as they happen (PI 2026-11-16: "copy finished runs and resume.pt to Drive
    as they happen"), not only at the end of each queue step.

    finished runs   copied to <drive>/runs/<id> once, verified by hash (a run already there with identical files
                    is not copied again); its small files go to results/colab/<queue>/ and are mirrored to Drive.
    unfinished runs a run folder holding ``resume.pt`` (a distillation in progress) is copied whole to
                    <drive>/partial/<id> whenever its resume.pt changes: written to a temporary folder, verified
                    by hash, then swapped in. A copy that changed while being copied is discarded and retried at the
                    next sync. Once the finished run is verified on Drive, its partial copy is removed (the run
                    itself deletes resume.pt when it finishes; the partial copy is only that checkpoint).
    A new session restores <drive>/partial/<id> for every expected run not finished locally or on Drive, so the
    run resumes from its last copied resume.pt (``restore_partial``).
    """

    def __init__(self, drive: Path, qname: str) -> None:
        self.drive, self.qname = drive, qname
        self.out_dir = REPO_ROOT / "results" / "colab" / qname
        self.files: dict[str, list[dict[str, Any]]] = {}      # finished runs already verified on Drive
        self.partial_seen: dict[str, tuple[int, int]] = {}    # run id -> (mtime_ns, size) of the copied resume.pt

    def mark_present(self, run_ids: list[str]) -> None:
        """Runs this session brought from Drive: already there, so only their file list is recorded."""
        for rid in run_ids:
            if is_finished(RUNS_DIR / rid):
                self.files[rid] = file_list(RUNS_DIR / rid, RUNS_DIR)

    def finished(self, run_ids: list[str]) -> None:
        for rid in run_ids:
            local = RUNS_DIR / rid
            if rid in self.files or not is_finished(local):
                continue
            remote = self.drive / "runs" / rid
            files = file_list(local, RUNS_DIR)
            if not (remote.exists() and is_finished(remote) and file_list(remote, self.drive / "runs") == files):
                copy_tree(local, remote)
                if file_list(remote, self.drive / "runs") != files:
                    raise RuntimeError(f"{rid}: Drive copy does not match the local run (hash mismatch)")
            self.files[rid] = files
            for f in files:
                p = RUNS_DIR / f["path"]
                if p.suffix in SMALL_SUFFIXES and f["bytes"] <= SMALL_MAX_BYTES:
                    dst = self.out_dir / "runs" / f["path"]
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dst)
            part = self.drive / "partial" / rid
            if part.exists():
                shutil.rmtree(part)
            self.partial_seen.pop(rid, None)

    def partial(self, run_ids: list[str]) -> list[str]:
        """Copy in-progress runs (folders with resume.pt) to <drive>/partial; returns the ids copied now."""
        copied = []
        for rid in run_ids:
            local = RUNS_DIR / rid
            ck = local / "resume.pt"
            if is_finished(local) or not ck.exists():
                continue
            try:
                st = ck.stat()
            except FileNotFoundError:          # replaced between the check and stat: next sync
                continue
            sig = (st.st_mtime_ns, st.st_size)
            if self.partial_seen.get(rid) == sig:
                continue
            files = file_list(local, local)
            tmp, dst = self.drive / "partial" / f"{rid}.tmp", self.drive / "partial" / rid
            copy_tree(local, tmp)
            if file_list(tmp, tmp) != files or file_list(local, local) != files:
                shutil.rmtree(tmp)             # the run wrote while we copied: retry at the next sync
                continue
            if dst.exists():
                shutil.rmtree(dst)
            tmp.rename(dst)
            self.partial_seen[rid] = sig
            copied.append(rid)
        return copied

    def write_results(self) -> list[dict[str, Any]]:
        entries = [e for rid in sorted(self.files) for e in self.files[rid]]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "files.sha256", "w") as fh:
            for e in entries:
                fh.write(f"{e['sha256']}  {e['bytes']}  {e['path']}\n")
        # mirror the small files to Drive, so results return without GitHub (Drive-only route)
        shutil.copytree(self.out_dir, self.drive / "results" / "colab" / self.qname, dirs_exist_ok=True)
        if COLAB_MANIFEST.exists():   # merge, never overwrite: Drive may hold rows of an earlier session (2026-11-17)
            merge_manifest(COLAB_MANIFEST, self.drive / "results" / COLAB_MANIFEST.name)
        return entries

    def sync(self, run_ids: list[str]) -> list[dict[str, Any]]:
        self.finished(run_ids)
        copied = self.partial(run_ids)
        if copied:
            print(f"    Drive: in-progress checkpoint copied for {copied}", flush=True)
        return self.write_results()


def sync_out(run_ids: list[str], drive: Path, qname: str) -> list[dict[str, Any]]:
    """Copy finished (and in-progress) runs to Drive, verify the copies by hash, collect small files for git."""
    return DriveSync(drive, qname).sync(run_ids)


def restore_partial(run_ids: list[str], drive: Path) -> list[str]:
    """New session: bring back in-progress runs from <drive>/partial (not finished locally or on Drive)."""
    out = []
    for rid in run_ids:
        part, local = drive / "partial" / rid, RUNS_DIR / rid
        if (not part.exists() or not (part / "resume.pt").exists() or is_finished(local)
                or is_finished(drive / "runs" / rid)):
            continue
        if local.exists():
            shutil.rmtree(local)               # a stale local folder from this session's start; Drive is newer
        shutil.copytree(part, local)
        out.append(rid)
    return out


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


def restore_manifest(drive: Path) -> int:
    """Merge Drive's Colab manifest into the local one; returns the rows added.

    A resumed session starts from a fresh clone, whose committed manifest lacks the rows of earlier sessions.
    The old rule copied Drive's manifest only when no local one existed, which a clone always has, so the next
    session's mirror overwrote 44 seed-4 rows on Drive (found 2026-11-17)."""
    src = drive / "results" / COLAB_MANIFEST.name
    return merge_manifest(src, COLAB_MANIFEST) if src.exists() else 0


def _session_setup(drive: Path, wanted: list[str]) -> list[str]:
    """Data cache, Colab manifest, teachers and the finished runs in ``wanted`` from Drive; then in-progress runs
    from their Drive checkpoints. Returns the finished runs brought from Drive."""
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
    (drive / "runs").mkdir(parents=True, exist_ok=True)
    restore_manifest(drive)
    wanted_set, brought = set(wanted), []
    for p in sorted((drive / "runs").iterdir()):
        if (p.name in EXPORT_MODELS or p.name in wanted_set) and is_finished(p) and not (RUNS_DIR / p.name).exists():
            shutil.copytree(p, RUNS_DIR / p.name)
            if p.name in wanted_set:
                brought.append(p.name)
    restored = restore_partial(wanted, drive)
    if restored:
        print(f"resuming in-progress runs from their last Drive checkpoint: {restored}", flush=True)
    return brought


def _run_step(st: dict[str, Any], seeds: list[int] | None, label: str, syncer: "DriveSync",
              expected: list[str]) -> None:
    ids = expected_runs(st["module"], st["config"], seeds)
    if all(is_finished(RUNS_DIR / r) for r in ids):
        print(f"{label}: all {len(ids)} runs finished, skipping", flush=True)
        return
    n = int(st.get("parallel", 1))              # assays only: n processes share the step (2026-11-16)
    if n > 1 and st["module"] != "zoo.assay":
        raise ValueError(f"parallel steps are allowed for zoo.assay only, not {st['module']}")
    print(f"{label}: {st['module']} --config {st['config']}" + (f" --seeds {seeds}" if seeds else "")
          + (f" ({n} processes in parallel)" if n > 1 else ""), flush=True)
    cmd = [sys.executable, "-m", st["module"], "--config", st["config"]]
    if seeds is not None:
        cmd += ["--seeds", ",".join(map(str, seeds))]
    cmds = [cmd + ["--shard", f"{i}/{n}"] for i in range(n)] if n > 1 else [cmd]
    t0 = time.time()
    procs = [subprocess.Popen(c, cwd=REPO_ROOT) for c in cmds]
    try:
        last = time.time()
        while any(p.poll() is None for p in procs):   # copy runs to Drive as they finish, checkpoints as they change
            time.sleep(5)                              # (the step's wall time is then exact to about 5 s)
            if time.time() - last >= SYNC_EVERY_S:
                syncer.sync(expected)
                last = time.time()
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            p.wait()
        syncer.sync(expected)                   # finished runs reach Drive even if this step failed
    for c, p in zip(cmds, procs):
        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, c)
    print(f"{label}: finished in {time.time() - t0:.0f} s", flush=True)
    if st["module"] == "zoo.distill":           # PI condition 2: training time per distillation run
        for r in ids:
            sm = RUNS_DIR / r / "summary.json"
            if sm.exists():
                print(f"    TIMING {r}: {json.loads(sm.read_text())['seconds']:.0f} s of training "
                      f"(send this line to the agent)", flush=True)


def code_commit_problem(head: str, drive: Path) -> str | None:
    """Code-version check (PI 2026-11-16): the code running a queue must be the commit the Drive folder expects,
    ``<drive>/code/COMMIT.txt`` (written with the bundle by ``make_bundle``). None if they agree, else the message."""
    stamp = drive / "code" / "COMMIT.txt"
    if not stamp.exists():
        return (f"STOP: {stamp} is missing, so the expected code version is unknown. Upload the code folder from "
                f"the laptop's export/drive_upload/ and Run all again. Nothing has run.")
    want = stamp.read_text().strip()
    if head != want:
        return (f"STOP: code version mismatch. The running code is commit {head[:12]}, but the Drive folder expects "
                f"{want[:12]} (code/COMMIT.txt). Upload the code folder and launch_any_GPU.ipynb from the same export, "
                f"move older notebooks out of the folder, and Run all again. Nothing has run.")
    return None


def run_queue(qpath: str, drive: Path) -> None:
    if os.environ.get("ZOO_ENV") != "colab":
        raise RuntimeError("queue runs on Colab only (ZOO_ENV=colab)")
    problem = code_commit_problem(_git("rev-parse", "HEAD"), drive)
    if problem:
        raise SystemExit(problem)
    if is_perseed(qpath):
        run_perseed(qpath, drive)
        return
    qname = Path(qpath).stem
    steps = load_queue(qpath)
    from zoo.gpucheck import check as gpu_check
    print(f"GPU: {gpu_check(qpath, str(drive))} (allowed and validated for this queue)", flush=True)
    expected = [r for s in steps for r in step_runs(s)]
    syncer = DriveSync(drive, qname)
    syncer.mark_present(_session_setup(drive, expected))
    for i, s in enumerate(steps):
        label = f"[{i + 1}/{len(steps)}] " + s["config"] + (f" seeds {s['seeds']}" if "seeds" in s else "")
        _run_step(s, s.get("seeds"), label, syncer, expected)
    entries = syncer.sync(expected)
    missing = [r for r in expected if not is_finished(RUNS_DIR / r)]
    print(f"queue {qname}: {len(expected) - len(missing)}/{len(expected)} runs finished; "
          f"{len(entries)} files on Drive, {sum(e['bytes'] for e in entries) / 1e9:.2f} GB")
    if missing:
        raise RuntimeError(f"unfinished runs: {missing}")


def run_perseed(qpath: str, drive: Path) -> None:
    """Per-seed queue (PI 2026-11-16): this GPU works through the seeds its family may take, one seed at a time,
    each seed locked to one GPU model; first any GPU-specific steps (e.g. an equivalence segment)."""
    qname = Path(qpath).stem
    q = load_perseed(qpath)
    from zoo.gpucheck import check as gpu_check
    gpu = gpu_check(qpath, None)                # validated and allowed; per-seed locks replace the queue lock
    key, fam = model_key(gpu), family_of(gpu, q)
    if fam is None:
        raise SystemExit(f"GPU {gpu!r} belongs to no family of {qpath}")
    print(f"GPU: {gpu} (family {fam}; locks record {key!r})", flush=True)
    qdir = drive / "results" / "colab" / qname
    first = (q.get("gpu_first_steps") or {}).get(gpu) or []
    family_runs = [r for k in q["seeds"] for r in seed_expected(q, fam, k)]
    first_runs = [r for st in first for r in expected_runs(st["module"], st["config"])]
    expected = first_runs + family_runs
    syncer = DriveSync(drive, qname)
    syncer.mark_present(_session_setup(drive, expected))
    for i, st in enumerate(first):
        _run_step(st, None, f"[{gpu} first step {i + 1}/{len(first)}] {st['config']}", syncer, expected)
    complete = lambda f, k: all(is_finished(drive / "runs" / r) or is_finished(RUNS_DIR / r)
                                for r in seed_expected(q, f, k))
    while True:
        locks = derive_locks(q, read_locks(qdir / "locks.json"), drive / "runs")
        write_locks(qdir / "locks.json", locks)
        nxt = next_seed(key, fam, q, locks, complete)
        if nxt is None:
            break
        k, new = nxt
        if new:
            old = locks.get(k)
            locks[k] = key
            write_locks(qdir / "locks.json", locks)
            if old is not None:
                append_claim(qdir / "claims.csv", {
                    "seed": k, "claimed_by": key, "superseded_gpu": old, "superseded_family": family_of(old, q),
                    "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                print(f"seed {k}: claimed from {old!r}; its runs there are now superseded", flush=True)
            else:
                print(f"seed {k}: locked to {key!r}", flush=True)
        steps = q["families"][fam]["steps"]
        for i, st in enumerate(steps):
            _run_step(st, [k], f"[seed {k}, step {i + 1}/{len(steps)}] {st['config']}", syncer, expected)
        if not complete(fam, k):
            raise RuntimeError(f"seed {k}: steps ran but not every run finished")
        print(f"seed {k}: complete on {key!r}", flush=True)
    syncer.sync(expected)
    print(f"queue {qname}: nothing left for {key!r} (locks: {read_locks(qdir / 'locks.json')})", flush=True)


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
        if queue != "auto" and not (REPO_ROOT / queue).exists():
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
    bad = [str(p) for p in files if p.is_symlink() or (p.suffix not in SMALL_SUFFIXES | {".sha256"}
           and p.name != "gpu.txt") or p.stat().st_size > SMALL_MAX_BYTES]   # gpu.txt: the queue's GPU lock
    if bad:
        raise ValueError(f"refusing to import non-result files: {bad[:10]}")
    return files


def merge_manifest(src: Path, dst: Path) -> int:
    """Append the rows of ``src`` whose run_id is not yet in ``dst`` (a session's Drive manifest may lack rows
    that other sessions wrote, e.g. the A100 equivalence rows; 2026-11-15). Existing lines are kept byte for
    byte; a run_id present in both must have an identical line, otherwise nothing is written."""
    import csv
    def lines(p: Path) -> tuple[str, list[str]]:
        raw = p.read_bytes().decode().splitlines(keepends=True)
        return (raw[0] if raw else ""), raw[1:]
    rid = lambda line: next(csv.reader([line]))[0]
    head_s, rows_s = lines(src)
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return len(rows_s)
    head_d, rows_d = lines(dst)
    if head_s.rstrip("\r\n") != head_d.rstrip("\r\n"):
        raise RuntimeError(f"{src}: manifest columns differ from {dst}; not merged")
    have = {rid(r): r for r in rows_d}
    for r in rows_s:
        k = rid(r)
        if k in have and have[k].rstrip("\r\n") != r.rstrip("\r\n"):
            raise RuntimeError(f"{k}: row differs between {src} and {dst}; not merged")
    new = [r for r in rows_s if rid(r) not in have]
    if new:
        with open(dst, "ab") as fh:
            if rows_d and not rows_d[-1].endswith("\n"):
                fh.write(b"\r\n")
            fh.write("".join(new).encode())
    return len(new)


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
        added = merge_manifest(man, COLAB_MANIFEST)
        print(f"manifest: {added} new rows merged into {COLAB_MANIFEST.name}")
    print(f"imported {len(files)} files into {dst}")
    if is_perseed(qpath):
        n = update_superseded(qpath)
        print(f"superseded runs listed: {n} (results/colab/superseded.csv)")
    return check_queue(qpath)


def update_superseded(qpath: str) -> int:
    """results/colab/superseded.csv from the imported claims of a per-seed queue (rows only added, never removed)."""
    import pandas as pd
    from zoo.seed_env import SUPERSEDED
    q = load_perseed(qpath)
    cpath = REPO_ROOT / "results" / "colab" / Path(qpath).stem / "claims.csv"
    claims = pd.read_csv(cpath).to_dict("records") if cpath.exists() else []
    known = set(pd.read_csv(COLAB_MANIFEST).run_id) if COLAB_MANIFEST.exists() else set()
    new = pd.DataFrame(superseded_runs(q, claims, known),
                       columns=["run_id", "seed", "gpu", "family", "claimed_by", "claimed_at"])
    old = pd.read_csv(SUPERSEDED) if SUPERSEDED.exists() else new.iloc[0:0]
    out = pd.concat([old, new[~new.run_id.isin(old.run_id)]], ignore_index=True)
    SUPERSEDED.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SUPERSEDED, index=False)
    return len(out)


def queue_steps_for_check(qpath: str, root: Path) -> list[dict[str, Any]]:
    """The steps a returned queue must have completed. A per-seed queue: each locked seed's steps under the family
    of its lock (one step per seed, restricted to that seed), and the first steps of each GPU that ran."""
    if not is_perseed(qpath):
        return load_queue(qpath)
    q = load_perseed(qpath)
    locks = read_locks(root / "locks.json")
    out = [st for g, sts in (q.get("gpu_first_steps") or {}).items() if g in set(locks.values()) for st in sts]
    for k, gpu in sorted(locks.items()):
        out += [{**st, "seeds": [k]} for st in q["families"][family_of(gpu, q)]["steps"]]
    return out


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
    import csv
    rows = {r["run_id"]: r for r in csv.DictReader(open(COLAB_MANIFEST, newline=""))} if COLAB_MANIFEST.exists() else {}
    for s in queue_steps_for_check(qpath, root):
        for rid in step_runs(s):
            if rows.get(rid, {}).get("status") != "ok":
                problems.append(f"{rid}: no ok row in {COLAB_MANIFEST.name}")
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
