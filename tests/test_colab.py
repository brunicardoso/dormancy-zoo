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
    monkeypatch.setattr(colab, "expected_runs", lambda m, c: ids + ["x-m-a2"])
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
