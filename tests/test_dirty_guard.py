"""PI 2026-11-17: runs refuse to start on uncommitted code; smoke tests may override, recorded and excluded."""
import json

import pandas as pd
import pytest

from zoo import assay, runs, seed_env


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(runs, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(runs, "MANIFEST", tmp_path / "results" / "manifest.csv")
    monkeypatch.setattr(runs, "env_string", lambda: "laptop; test")
    monkeypatch.setattr(runs, "project_disk_bytes", lambda: 0)
    monkeypatch.delenv(runs.DIRTY_OVERRIDE_ENV, raising=False)
    return tmp_path


def _run(monkeypatch, run_id, dirty):
    monkeypatch.setattr(runs, "git_info", lambda: ("abc123", dirty))
    return runs.Run(run_id, "assay", {"x": 1}, "configs/x.yaml", seed=0)


def test_refuses_dirty_tree_and_override_is_smoke_only(sandbox, monkeypatch):
    with pytest.raises(RuntimeError, match="uncommitted"):
        _run(monkeypatch, "assay-m-a0", True).start()
    assert not (sandbox / "runs" / "assay-m-a0").exists()          # nothing written
    monkeypatch.setenv(runs.DIRTY_OVERRIDE_ENV, "1")
    with pytest.raises(RuntimeError, match="smoke tests only"):
        _run(monkeypatch, "assay-m-a0", True).start()
    with pytest.raises(RuntimeError, match="uncommitted|smoke"):
        _run(monkeypatch, "assay-m-a0", True).finish()               # finish without start is checked too


def test_clean_tree_runs_normally(sandbox, monkeypatch):
    r = _run(monkeypatch, "assay-m-a0", False).start()
    r.finish(notes="t_cross=30")
    m = pd.read_csv(runs.MANIFEST)
    assert m.notes.tolist() == ["t_cross=30"] and not runs.is_smoke_override(m).any()
    assert "dirty_smoke_override" not in json.loads((r.dir / "meta.json").read_text())


def test_smoke_override_is_recorded_and_excluded(sandbox, monkeypatch):
    monkeypatch.setenv(runs.DIRTY_OVERRIDE_ENV, "1")
    r = _run(monkeypatch, "assaysmoke-m-a0", True).start()
    r.finish(notes="t_cross=30")
    _run(monkeypatch, "assay-m-a1", False).start().finish()
    m = pd.read_csv(runs.MANIFEST)
    assert m.notes[0].startswith(runs.SMOKE_OVERRIDE_NOTE) and str(m.git_dirty[0]) == "True"
    assert json.loads((r.dir / "meta.json").read_text())["dirty_smoke_override"] is True
    monkeypatch.setattr(seed_env, "superseded_ids", lambda path=None: set())
    assert seed_env.drop_superseded(m).run_id.tolist() == ["assay-m-a1"]


def test_override_model_cannot_feed_a_real_run(sandbox, monkeypatch):
    monkeypatch.setattr(assay, "RUNS_DIR", sandbox / "runs")
    d = sandbox / "runs" / "smoke_model"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"dirty_smoke_override": True}))
    with pytest.raises(RuntimeError, match="smoke tests only"):
        assay.load_model_for_assay({}, "smoke_model", "cpu")
