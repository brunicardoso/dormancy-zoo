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
    colab.COLAB_MANIFEST.write_text("run_id,env\nx-m-a0,colab\n")
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
    # PI 2026-11-16: G4 and A100 40 GB validated; the 80 GB and the L4 are candidates only
    assert [g["gpu"] for g in lst["validated"]] == ["NVIDIA RTX PRO 6000 Blackwell Server Edition",
                                                    "NVIDIA A100-SXM4-40GB"]
    g4q = yaml.safe_load(Path("configs/colab/queue_distill_g4.yaml").read_text())
    a100q = yaml.safe_load(Path("configs/colab/queue_equiv_a100.yaml").read_text())
    psq = yaml.safe_load(Path("configs/colab/queue_distill_perseed.yaml").read_text())
    assert problems("NVIDIA RTX PRO 6000 Blackwell Server Edition", g4q, lst) == []
    assert problems("NVIDIA A100-SXM4-40GB", g4q, lst)                       # the old G4 queue stays G4-only
    assert problems("NVIDIA A100-SXM4-40GB", psq, lst) == []
    assert problems("NVIDIA A100-SXM4-80GB", psq, lst)                       # not validated
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
    # PI 2026-11-16: G4-only reversed; the per-seed queue on validated GPUs, the 80 GB test on the 80 GB card
    assert pick_queue("NVIDIA RTX PRO 6000 Blackwell Server Edition", mapping) == "configs/colab/queue_distill_perseed.yaml"
    assert pick_queue("NVIDIA A100-SXM4-40GB", mapping) == "configs/colab/queue_distill_perseed.yaml"
    assert pick_queue("NVIDIA A100-SXM4-80GB", mapping) == "configs/colab/queue_equiv_a100_80gb.yaml"
    with pytest.raises(SystemExit):
        pick_queue("NVIDIA L4", mapping)
    with pytest.raises(SystemExit):
        pick_queue("Tesla T4", mapping)


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
