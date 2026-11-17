"""Parallel assays on Colab (2026-11-16): shards, the queue step, and manifest appends from several processes."""

import csv
import multiprocessing as mp

import pytest

from zoo import colab, runs
from zoo.assay import select_pairs


def test_shards_partition_the_batch():
    pairs = [(f"m-s{k}", a) for k in (3, 4) for a in (0, 1, 2)]
    shards = [select_pairs(pairs, "4", f"{i}/3") for i in range(3)]
    assert sorted(p for s in shards for p in s) == sorted(p for p in pairs if p[0] == "m-s4")
    assert all(len(s) == 1 for s in shards)
    assert select_pairs(pairs) == pairs
    with pytest.raises(ValueError):
        select_pairs(pairs, None, "3/3")


class _FakePopen:
    calls = []

    def __init__(self, cmd, cwd=None):
        _FakePopen.calls.append(cmd)
        self.returncode = 0

    def poll(self):
        return 0

    def wait(self):
        return 0

    def terminate(self):
        pass


class _NoSync:
    def sync(self, ids):
        return []


def test_run_step_launches_one_process_per_shard(tmp_path, monkeypatch):
    monkeypatch.setattr(colab, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(colab, "expected_runs", lambda m, c, seeds=None: ["x-m-s4-a0"])
    monkeypatch.setattr(colab.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(colab.time, "sleep", lambda s: None)
    _FakePopen.calls = []
    colab._run_step({"module": "zoo.assay", "config": "c.yaml", "parallel": 3}, [4], "t", _NoSync(), [])
    assert [c[-2:] for c in _FakePopen.calls] == [["--shard", "0/3"], ["--shard", "1/3"], ["--shard", "2/3"]]
    assert all("--seeds" in c for c in _FakePopen.calls)
    _FakePopen.calls = []
    colab._run_step({"module": "zoo.assay", "config": "c.yaml"}, [4], "t", _NoSync(), [])
    assert len(_FakePopen.calls) == 1 and "--shard" not in _FakePopen.calls[0]
    with pytest.raises(ValueError):
        colab._run_step({"module": "zoo.distill", "config": "c.yaml", "parallel": 2}, [4], "t", _NoSync(), [])


def _append_many(path, k):
    runs.MANIFEST, runs.RESULTS_DIR = path, path.parent
    for i in range(50):
        runs.append_manifest({"run_id": f"p{k}-r{i}", "status": "ok", "notes": "x" * 200})


def test_manifest_appends_from_parallel_processes(tmp_path, monkeypatch):
    path = tmp_path / "manifest_colab.csv"
    monkeypatch.setattr(runs, "MANIFEST", path)
    monkeypatch.setattr(runs, "RESULTS_DIR", tmp_path)
    runs.append_manifest({"run_id": "first", "status": "ok"})
    procs = [mp.get_context("fork").Process(target=_append_many, args=(path, k)) for k in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    rows = list(csv.DictReader(open(path, newline="")))
    assert len(rows) == 201 and len({r["run_id"] for r in rows}) == 201
    assert all(r["status"] == "ok" for r in rows)


def test_queue_first_steps_include_the_parallel_check():
    q = colab.load_perseed("configs/colab/queue_distill_perseed.yaml")
    first = q["gpu_first_steps"]
    for gpu, cfg in (("NVIDIA A100-SXM4-80GB", "parcheck_assay_a100_80gb"), ("NVIDIA A100-SXM4-40GB", "parcheck_assay_a100")):
        st = [s for s in first[gpu] if cfg in s["config"]]
        assert len(st) == 1 and st[0]["parallel"] == 3 and st[0]["module"] == "zoo.assay"
    for fam in q["families"].values():          # production steps stay sequential until the PI approves
        assert all(int(s.get("parallel", 1)) == 1 for s in fam["steps"])


def test_parallel_check_reports_a_card_without_parallel_runs_as_pending(tmp_path, monkeypatch, capsys):
    """A card whose parallel runs never ran is pending, not 'NOT identical' (bug found 2026-11-17)."""
    from zoo import parallel_check as pc
    monkeypatch.setattr(pc, "COLAB", tmp_path)
    monkeypatch.setattr(pc, "compare_pair", lambda a, b, names: [
        {"run_40gb": a[1], "run_80gb": b[1], "file": "curve.csv", "in_both": False, "how": "missing", "identical": False}])
    pc.main()
    out = capsys.readouterr().out
    assert "NOT identical" not in out and out.count("PENDING") == 2
