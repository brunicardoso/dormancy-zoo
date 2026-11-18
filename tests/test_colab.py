"""Colab sync: expected runs, Drive copy with hash check, small-file filter, completeness check."""

import json

import pytest

from zoo import colab


def test_expected_runs():
    ids = colab.expected_runs("zoo.assay", "configs/colab/equiv_assay.yaml")
    assert ids == [f"colabeq-{m}-a{a}" for m in ("plant_lr3e-4-naive-s0", "ct4a-plant_lr3e-4-mask-f0.100-s0-d01x")
                   for a in (0, 1, 2)]
    d = colab.expected_runs("zoo.distill", "configs/colab/distill_1xteacher.yaml")
    assert d[:2] == ["dist4b1x-s0", "dist4b1x-s0-d64x"] and len(d) == 10
    t = colab.expected_runs("zoo.distill", "configs/colab/distill_treated.yaml")
    assert len(t) == 30 and "dist4b-s4-d08x" in t


def test_queue_modules_restricted(tmp_path):
    q = tmp_path / "q.yaml"
    q.write_text("steps:\n  - {module: os, config: x}\n")
    with pytest.raises(ValueError):
        colab.load_queue(str(q))


def _fake_run(root, rid, rows=1201, status="ok"):
    d = root / rid
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"status": status}))
    (d / "config.yaml").write_text("a: 1\n")
    (d / "summary.json").write_text("{}")
    (d / "curve.csv").write_text("step\n" + "".join(f"{i}\n" for i in range(rows)))
    (d / "ckpt_step0300_fp16.safetensors").write_bytes(b"\0" * 3000)


def test_sync_out_and_check(tmp_path, monkeypatch):
    runs, repo, drive = tmp_path / "runs", tmp_path / "repo", tmp_path / "drive"
    monkeypatch.setattr(colab, "RUNS_DIR", runs)
    monkeypatch.setattr(colab, "REPO_ROOT", repo)
    monkeypatch.setattr(colab, "COLAB_MANIFEST", repo / "results" / "manifest_colab.csv")
    (repo / "results").mkdir(parents=True)
    colab.COLAB_MANIFEST.write_text("run_id,status,env\nx-m-a0,ok,colab\n")
    ids = ["x-m-a0", "x-m-a1"]
    _fake_run(runs, ids[0])
    _fake_run(runs, ids[1], rows=1000)
    _fake_run(runs, "x-m-a2", status="failed")
    entries = colab.sync_out(ids + ["x-m-a2"], drive, "q")
    assert (drive / "runs" / ids[0] / "ckpt_step0300_fp16.safetensors").exists()
    assert not (drive / "runs" / "x-m-a2").exists()                       # unfinished runs stay local
    small = repo / "results" / "colab" / "q" / "runs" / ids[0]
    assert (small / "curve.csv").exists() and not (small / "ckpt_step0300_fp16.safetensors").exists()
    assert len(entries) == 10
    monkeypatch.setattr(colab, "load_queue", lambda p: [{"module": "zoo.assay", "config": "c"}])
    monkeypatch.setattr(colab, "expected_runs", lambda m, c, seeds=None: ids + ["x-m-a2"])
    probs = colab.check_queue(str(tmp_path / "q.yaml"))
    assert any("x-m-a1: curve has 1000 rows" in p for p in probs)
    assert any(p.startswith("x-m-a2") for p in probs)
    assert not any(p.startswith("x-m-a0") for p in probs)
    (small / "curve.csv").write_text("tampered\n")
    assert any("x-m-a0/curve.csv: hash" in p for p in colab.check_queue(str(tmp_path / "q.yaml")))
    # Drive-only route: the small files and the Colab manifest are mirrored to <drive>/results and
    # import back into a fresh laptop repo with the same check result
    assert (drive / "results" / "colab" / "q" / "files.sha256").exists()
    assert (drive / "results" / "manifest_colab.csv").read_text() == colab.COLAB_MANIFEST.read_text()
    repo2 = tmp_path / "repo2"
    monkeypatch.setattr(colab, "REPO_ROOT", repo2)
    monkeypatch.setattr(colab, "COLAB_MANIFEST", repo2 / "results" / "manifest_colab.csv")
    probs2 = colab.import_results(str(tmp_path / "q.yaml"), drive / "results")
    assert sorted(probs2) == sorted(probs)
    assert colab.COLAB_MANIFEST.exists()


def test_import_refuses_anything_but_small_text_results(tmp_path):
    from zoo.colab import small_result_files
    ok = tmp_path / "ok" / "runs" / "x-a0"
    ok.mkdir(parents=True)
    for f in ("curve.csv", "meta.json", "config.yaml"):
        (ok / f).write_text("1\n")
    (tmp_path / "ok" / "files.sha256").write_text("")
    assert len(small_result_files(tmp_path / "ok")) == 4
    (ok / "final_fp16.safetensors").write_bytes(b"\0" * 8)
    with pytest.raises(ValueError):
        small_result_files(tmp_path / "ok")
    big = tmp_path / "big"
    big.mkdir()
    (big / "huge.csv").write_bytes(b"0" * (2 * 1024 ** 2 + 1))
    with pytest.raises(ValueError):
        small_result_files(big)


def test_cache_problems_reports_missing_and_different_files():
    want = {"a.npy": "1", "pools/b.npy": "2", "c.npy": "3"}
    assert colab.cache_problems(want, {"a.npy": "1", "pools/b.npy": "2", "c.npy": "3", "extra": "9"}) == []
    assert colab.cache_problems(want, {"a.npy": "1", "pools/b.npy": "x"}) == ["c.npy: missing", "pools/b.npy: differs"]
    assert "val_train_duplicates.npy" in colab.AUDIT_ONLY


def test_gpu_guard():
    from zoo.gpucheck import problems
    lst = {"validated": [{"gpu": "G4X"}], "equivalence_candidates": [{"gpu": "A100X"}]}
    distill = {"gpus": ["G4X"]}
    equiv = {"gpus": ["A100X"], "equivalence_test": True}
    assert problems("G4X", distill, lst) == []
    assert problems("A100X", distill, lst)                      # not allowed by the queue, not validated
    assert problems("A100X", equiv, lst) == []                  # candidate, equivalence queue
    assert problems("A100X", {"gpus": ["A100X"]}, lst)          # candidate, but not an equivalence queue
    assert problems("G4X", distill, lst, locked="G4Y")          # never resume on another GPU
    assert problems("G4X", {}, lst)                             # a queue must name its GPUs


def test_repo_gpu_list_and_queues():
    import yaml
    from pathlib import Path
    from zoo.gpucheck import LIST, problems
    lst = yaml.safe_load(LIST.read_text())
    # PI 2026-11-16: G4 and A100 40 GB validated; evening: A100 80 GB validated as its own model; L4 a candidate only
    assert [g["gpu"] for g in lst["validated"]] == ["NVIDIA RTX PRO 6000 Blackwell Server Edition",
                                                    "NVIDIA A100-SXM4-40GB", "NVIDIA A100-SXM4-80GB"]
    assert not any(g.get("same_model_as") for g in lst["validated"])         # three separate GPU models
    g4q = yaml.safe_load(Path("configs/colab/queue_distill_g4.yaml").read_text())
    a100q = yaml.safe_load(Path("configs/colab/queue_equiv_a100.yaml").read_text())
    psq = yaml.safe_load(Path("configs/colab/queue_distill_perseed.yaml").read_text())
    assert problems("NVIDIA RTX PRO 6000 Blackwell Server Edition", g4q, lst) == []
    assert problems("NVIDIA A100-SXM4-40GB", g4q, lst)                       # the old G4 queue stays G4-only
    assert problems("NVIDIA A100-SXM4-40GB", psq, lst) == []
    assert problems("NVIDIA A100-SXM4-80GB", psq, lst) == []                 # validated 2026-11-16 (evening)
    assert problems("NVIDIA A100-SXM4-40GB", a100q, lst) == []
    assert problems("NVIDIA RTX PRO 6000 Blackwell Server Edition", a100q, lst)   # the A100 test runs on an A100
    l4q = yaml.safe_load(Path("configs/colab/queue_equiv_l4.yaml").read_text())
    assert problems("NVIDIA L4", l4q, lst) == []
    assert problems("NVIDIA L4", g4q, lst)                                    # the L4 is not validated


def test_any_gpu_queue_map():
    import yaml
    from pathlib import Path
    from zoo.gpucheck import LIST, QUEUE_MAP, pick_queue, problems
    lst, mapping = yaml.safe_load(LIST.read_text()), yaml.safe_load(QUEUE_MAP.read_text())
    for gpu, q in mapping["queues"].items():                     # every mapped queue passes the guard on its GPU
        assert problems(gpu, yaml.safe_load(Path(q).read_text()), lst) == [], (gpu, q)
    # 2026-11-17 (PI, route A): every GPU runs the Phase 4b follow-up queue; only its distances run off the A100 80 GB
    fu = "configs/colab/queue_4b_followup.yaml"
    for gpu in ("NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-40GB", "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "NVIDIA L4", "Tesla T4"):
        assert pick_queue(gpu, mapping) == fu
        assert problems(gpu, yaml.safe_load(Path(fu).read_text()), lst) == []
    with pytest.raises(SystemExit):
        pick_queue("Tesla T4", {"queues": {"X": "q.yaml"}})      # without a default, an unmapped GPU stops


def test_any_gpu_queue_gates_every_gpu_step():
    from zoo.gpucheck import problems
    lst = {"validated": [{"gpu": "A80"}]}
    ok = {"any_gpu": True, "steps": [{"module": "zoo.phase4b_distances", "config": "d"},
                                     {"module": "zoo.assay", "config": "a", "requires_gpu": ["A80"]}]}
    assert problems("T4", ok, lst) == []
    ungated = {"any_gpu": True, "steps": [{"module": "zoo.assay", "config": "a"}]}
    assert problems("T4", ungated, lst)                          # a GPU step must name the GPU it needs
    unvalidated = {"any_gpu": True, "steps": [{"module": "zoo.assay", "config": "a", "requires_gpu": ["L4"]}]}
    assert problems("A80", unvalidated, lst)                     # and that GPU must be validated
    assert colab.gpu_ok(ok["steps"][1], "A80") and not colab.gpu_ok(ok["steps"][1], "A40")
    assert colab.gpu_ok(ok["steps"][0], "T4")


def test_followup_queue_runs_distances_anywhere_and_readouts_on_the_80gb_only():
    q = colab.load_queue("configs/colab/queue_4b_followup.yaml")
    assert [s["module"] for s in q if colab.gpu_ok(s, "NVIDIA A100-SXM4-40GB")] == \
        ["zoo.phase4b_distances", "zoo.phase4b_k2"]                  # the GPU-independent steps
    assert [s["module"] for s in q if colab.gpu_ok(s, "NVIDIA A100-SXM4-80GB")] == \
        ["zoo.phase4b_distances", "zoo.phase4b_k2", "zoo.assay", "zoo.assay", "zoo.assay", "zoo.base_folder",
         "zoo.assay", "zoo.assay"]
    runs = [r for s in q for r in colab.step_runs(s)]
    assert runs[:2] == ["a100dist4bdistances", "a100k2students"]
    assert len(runs) == 2 + 2 + 2 + 45 + 1 + 3 + 1
    assert {"a100base", "a100assay-a100base-a2", "a100assayvehicle-a100base-a0"} <= set(runs)
    assert colab.required_files("a100dist4bdistances", "zoo.phase4b_distances")[-1] == "nearest_teacher.csv"
    assert "k2_vs_sham.csv" in colab.required_files("a100k2students", "zoo.phase4b_k2")
    assert "final_metrics.csv" in colab.required_files("a100base", "zoo.base_folder")


def test_distill_queue_expected_runs_and_seed_steps():
    steps = colab.load_queue("configs/colab/queue_distill_g4.yaml")
    ids = [r for s in steps for r in colab.step_runs(s)]
    assert len(ids) == len(set(ids))                         # no run id twice
    assert colab.step_runs(steps[0]) == ["dist4b-s0"] + [f"dist4b-s0-d{m:02d}x" for m in (1, 8, 16, 32, 64)]
    assays = [r for s in steps if s["module"] == "zoo.assay" for r in colab.step_runs(s)]
    assert len(assays) == 15 + 5 + 45 + 45 + 30 + 15 + 5
    assert all(r.startswith(("g4assay", "g4pos")) for r in assays)
    # every assayed model is produced by a distillation step of the queue, or is an exported model
    made = set(ids) | set(colab.EXPORT_MODELS)
    from zoo.config import load_config
    for s in steps:
        if s["module"] == "zoo.assay":
            assert set(load_config(s["config"])["models"]) <= made


def test_seeds_only_on_distill_steps(tmp_path):
    q = tmp_path / "q.yaml"
    q.write_text("steps:\n  - {module: zoo.assay, config: configs/colab/g4_assay_naive.yaml, seeds: [0]}\n")
    with pytest.raises(ValueError):
        colab.load_queue(str(q))
    q.write_text("steps:\n  - {module: zoo.distill, config: configs/colab/distill_treated.yaml, seeds: [7]}\n")
    with pytest.raises(ValueError):
        colab.load_queue(str(q))


def test_merge_manifest_keeps_rows_and_refuses_conflicts(tmp_path):
    from zoo.colab import merge_manifest
    dst, src = tmp_path / "local.csv", tmp_path / "drive.csv"
    dst.write_bytes(b"run_id,status\r\na,ok\r\nb100,ok\r\n")          # b100: a row only the laptop copy has
    src.write_bytes(b"run_id,status\r\na,ok\r\nc,ok\r\nd,ok\r\n")
    assert merge_manifest(src, dst) == 2
    assert dst.read_bytes() == b"run_id,status\r\na,ok\r\nb100,ok\r\nc,ok\r\nd,ok\r\n"
    assert merge_manifest(src, dst) == 0                                   # idempotent
    src.write_bytes(b"run_id,status\r\na,failed\r\n")
    with pytest.raises(RuntimeError):
        merge_manifest(src, dst)
    assert dst.read_bytes().count(b"\n") == 5                              # unchanged


def test_drive_sync_partial_checkpoints_and_restore(tmp_path, monkeypatch):
    runs, repo, drive = tmp_path / "runs", tmp_path / "repo", tmp_path / "drive"
    monkeypatch.setattr(colab, "RUNS_DIR", runs)
    monkeypatch.setattr(colab, "REPO_ROOT", repo)
    monkeypatch.setattr(colab, "COLAB_MANIFEST", repo / "results" / "manifest_colab.csv")
    d = runs / "dist-s0"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"status": "running"}))
    (d / "resume.pt").write_bytes(b"step500")
    sy = colab.DriveSync(drive, "q")
    sy.sync(["dist-s0"])
    assert (drive / "partial" / "dist-s0" / "resume.pt").read_bytes() == b"step500"
    assert not (drive / "runs" / "dist-s0").exists()                  # unfinished: only the partial copy
    assert sy.partial(["dist-s0"]) == []                              # unchanged checkpoint: not copied again
    (d / "resume.pt").write_bytes(b"step1000")
    assert sy.partial(["dist-s0"]) == ["dist-s0"]
    assert (drive / "partial" / "dist-s0" / "resume.pt").read_bytes() == b"step1000"
    assert not (drive / "partial" / "dist-s0.tmp").exists()
    # a new session (fresh clone, empty runs/) resumes from the Drive copy
    import shutil
    shutil.rmtree(runs)
    assert colab.restore_partial(["dist-s0", "other"], drive) == ["dist-s0"]
    assert (runs / "dist-s0" / "resume.pt").read_bytes() == b"step1000"
    # once the run finishes and is verified on Drive, the partial copy is removed
    (runs / "dist-s0" / "resume.pt").unlink()
    (runs / "dist-s0" / "meta.json").write_text(json.dumps({"status": "ok"}))
    colab.DriveSync(drive, "q").sync(["dist-s0"])
    assert colab.is_finished(drive / "runs" / "dist-s0") and not (drive / "partial" / "dist-s0").exists()
    assert colab.restore_partial(["dist-s0"], drive) == []


def test_drive_sync_copies_each_finished_run_once(tmp_path, monkeypatch):
    runs, repo, drive = tmp_path / "runs", tmp_path / "repo", tmp_path / "drive"
    monkeypatch.setattr(colab, "RUNS_DIR", runs)
    monkeypatch.setattr(colab, "REPO_ROOT", repo)
    monkeypatch.setattr(colab, "COLAB_MANIFEST", repo / "results" / "manifest_colab.csv")
    _fake_run(runs, "x-m-a0")
    sy = colab.DriveSync(drive, "q")
    sy.sync(["x-m-a0", "x-m-a1"])
    calls, real = [], colab.copy_tree
    monkeypatch.setattr(colab, "copy_tree", lambda a, b: (calls.append(a), real(a, b)))
    _fake_run(runs, "x-m-a1")
    entries = sy.sync(["x-m-a0", "x-m-a1"])
    assert calls == [runs / "x-m-a1"]                                 # x-m-a0 is not copied or re-hashed again
    assert len(entries) == 10


def test_code_commit_check_queue_side(tmp_path):
    """PI 2026-11-16: a queue refuses to run on code other than the commit the Drive folder expects."""
    from zoo.colab import code_commit_problem
    new, old = "7" * 40, "5" * 40
    assert "missing" in code_commit_problem(new, tmp_path)
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "COMMIT.txt").write_text(new + "\n")
    assert code_commit_problem(new, tmp_path) is None
    msg = code_commit_problem(old, tmp_path)
    assert msg.startswith("STOP") and old[:12] in msg and new[:12] in msg


def _notebook_check(tmp_path, notebook_commit, stamp, bundle_head):
    """Run the launcher's code-version check (cell 3, up to its print) with a stub git."""
    import json
    import textwrap
    from zoo.colab import NOTEBOOK
    src = "".join(json.loads(NOTEBOOK.read_text())["cells"][3]["source"])
    start = src.index('if CODE_SOURCE == "drive":')
    end = src.index("print(\"code version check", start)
    block = src[start:end] + "    pass\n"
    drive = tmp_path / "drive"
    (drive / "code").mkdir(parents=True)
    (drive / "code" / "dormancy_zoo.bundle").write_text("stub")
    if stamp:
        (drive / "code" / "COMMIT.txt").write_text(stamp + "\n")
    def git(*args, cwd=None, secret=False):
        assert args[:2] == ("bundle", "list-heads")
        return f"{bundle_head} HEAD\n{bundle_head} refs/heads/phase1-harness"
    import os
    exec(textwrap.dedent(block), {"CODE_SOURCE": "drive", "DRIVE_DIR": str(drive), "COMMIT": notebook_commit,
                                  "git": git, "os": os})


def test_notebook_stops_on_code_version_mismatch(tmp_path):
    new, old = "7704ca1b65f10c97a53ce88ca7211b1db2e7cdc7", "50c0c42f3e3886c5bd1e8131420c0a2e60b6966b"
    _notebook_check(tmp_path / "a", new, new, new)                     # all agree: passes
    for stamp, head in ((old, old), (new, old), (None, new)):          # old code folder, stale bundle, no stamp
        with pytest.raises(RuntimeError, match="CODE VERSION MISMATCH"):
            _notebook_check(tmp_path / f"{stamp}-{head}", new, stamp, head)
    with pytest.raises(RuntimeError, match="CODE VERSION MISMATCH"):  # old notebook, new code folder
        _notebook_check(tmp_path / "b", old, new, new)


def test_resumed_session_keeps_the_rows_of_earlier_sessions(tmp_path, monkeypatch):
    """A fresh clone has a committed manifest; Drive's extra rows must be merged in, not skipped (2026-11-17)."""
    repo, drive = tmp_path / "repo", tmp_path / "drive"
    monkeypatch.setattr(colab, "COLAB_MANIFEST", repo / "results" / "manifest_colab.csv")
    (repo / "results").mkdir(parents=True)
    (drive / "results").mkdir(parents=True)
    colab.COLAB_MANIFEST.write_text("run_id,status\nold-a0,ok\n")
    (drive / "results" / "manifest_colab.csv").write_text("run_id,status\nold-a0,ok\nsession1-s4-a0,ok\n")
    assert colab.restore_manifest(drive) == 1
    assert "session1-s4-a0" in colab.COLAB_MANIFEST.read_text()
    # mirroring back merges as well: a row only on Drive survives a later session's mirror
    (drive / "results" / "manifest_colab.csv").write_text("run_id,status\nold-a0,ok\nsession1-s4-a0,ok\nother-a0,ok\n")
    colab.merge_manifest(colab.COLAB_MANIFEST, drive / "results" / "manifest_colab.csv")
    assert "other-a0" in (drive / "results" / "manifest_colab.csv").read_text()


def test_check_queue_requires_a_manifest_row_for_every_run(tmp_path, monkeypatch):
    runs, repo, drive = tmp_path / "runs", tmp_path / "repo", tmp_path / "drive"
    monkeypatch.setattr(colab, "RUNS_DIR", runs)
    monkeypatch.setattr(colab, "REPO_ROOT", repo)
    monkeypatch.setattr(colab, "COLAB_MANIFEST", repo / "results" / "manifest_colab.csv")
    (repo / "results").mkdir(parents=True)
    colab.COLAB_MANIFEST.write_text("run_id,status\nx-m-a0,ok\n")
    for r in ("x-m-a0", "x-m-a1"):
        _fake_run(runs, r)
    colab.sync_out(["x-m-a0", "x-m-a1"], drive, "q")
    monkeypatch.setattr(colab, "load_queue", lambda p: [{"module": "zoo.assay", "config": "c"}])
    monkeypatch.setattr(colab, "expected_runs", lambda m, c, seeds=None: ["x-m-a0", "x-m-a1"])
    probs = colab.check_queue(str(tmp_path / "q.yaml"))
    assert probs == ["x-m-a1: no ok row in manifest_colab.csv"]


def test_rebuild_adds_marked_rows_only_for_runs_without_one(tmp_path, monkeypatch):
    from zoo import colab_rebuild
    runs, repo, drive = tmp_path / "runs", tmp_path / "repo", tmp_path / "drive"
    monkeypatch.setattr(colab, "RUNS_DIR", runs)
    monkeypatch.setattr(colab, "REPO_ROOT", repo)
    monkeypatch.setattr(colab, "COLAB_MANIFEST", repo / "results" / "manifest_colab.csv")
    (repo / "results").mkdir(parents=True)
    header = "run_id,run_type,status,reason,seed,config_path,config_hash,git_commit,git_dirty,started,finished,run_dir,run_disk_bytes,project_disk_bytes,notes,env\n"
    kept = "x-m-a0,assay,ok,,0,c.yaml,h0,abc,False,t0,t1,runs/x-m-a0,5,6,t_cross=10,colab\n"
    colab.COLAB_MANIFEST.write_text(header + kept)
    for r in ("x-m-a0", "x-m-a1"):
        _fake_run(runs, r)
    meta = {"run_id": "x-m-a1", "run_type": "assay", "status": "ok", "reason": "", "seed": 1, "config_path": "c.yaml",
            "config_hash": "h1", "git_commit": "def", "git_dirty": False, "started": "s", "finished": "f", "env": "colab"}
    (runs / "x-m-a1" / "meta.json").write_text(json.dumps(meta))
    colab.sync_out(["x-m-a0", "x-m-a1"], drive, "q")
    monkeypatch.setattr(colab, "load_queue", lambda p: [{"module": "zoo.assay", "config": "c"}])
    monkeypatch.setattr(colab, "expected_runs", lambda m, c, seeds=None: ["x-m-a0", "x-m-a1"])
    q = str(tmp_path / "q.yaml")
    assert [r["run_id"] for r in colab_rebuild.rebuild(q)] == ["x-m-a1"]
    lines = colab.COLAB_MANIFEST.read_text().splitlines()
    assert lines[1] + "\n" == kept                                         # existing row untouched
    new = lines[2].split(",")
    assert new[:12] == ["x-m-a1", "assay", "ok", "", "1", "c.yaml", "h1", "def", "False", "s", "f", "runs/x-m-a1"]
    assert new[12:14] == ["", ""] and new[14].startswith("RECONSTRUCTED") and new[15] == "colab"
    assert colab_rebuild.rebuild(q) == [] and colab.check_queue(q) == []  # idempotent; queue now complete


def test_followup_queue_on_another_gpu_runs_the_distances_and_skips_the_readouts(tmp_path, monkeypatch, capsys):
    """PI 2026-11-17, condition 2: on a GPU other than the A100 80 GB the session still computes the distances."""
    import zoo.gpucheck as gc
    ran, setup = [], {}
    monkeypatch.setenv("ZOO_ENV", "colab")
    monkeypatch.setattr(colab, "code_commit_problem", lambda head, drive: None)
    monkeypatch.setattr(gc, "check", lambda q, d: "NVIDIA A100-SXM4-40GB")
    monkeypatch.setattr(colab, "_session_setup", lambda d, w, inputs=(): setup.update(w=w, inputs=inputs) or [])
    monkeypatch.setattr(colab, "_run_step", lambda s, seeds, label, syncer, expected: ran.append(s["module"]))
    monkeypatch.setattr(colab, "is_finished", lambda p: True)

    class Sync:
        def __init__(self, *a): pass
        def mark_present(self, x): pass
        def sync(self, e): return []
    monkeypatch.setattr(colab, "DriveSync", Sync)
    colab.run_queue("configs/colab/queue_4b_followup.yaml", tmp_path)
    assert ran == ["zoo.phase4b_distances", "zoo.phase4b_k2"]
    assert setup["w"] == ["a100dist4bdistances", "a100k2students"] and len(setup["inputs"]) == 55
    assert capsys.readouterr().out.count("STOP for step") == 6
    ran.clear()
    monkeypatch.setattr(gc, "check", lambda q, d: "NVIDIA A100-SXM4-80GB")
    colab.run_queue("configs/colab/queue_4b_followup.yaml", tmp_path)
    assert ran == ["zoo.phase4b_distances", "zoo.phase4b_k2", "zoo.assay", "zoo.assay", "zoo.assay",
                   "zoo.base_folder", "zoo.assay", "zoo.assay"]
