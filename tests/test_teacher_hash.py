"""Distillation records the hash of the teacher weights it loads (PI 2026-11-17, item 4)."""
import hashlib
import inspect
import json
import types

import pytest

from zoo import assay, distill


def _teacher(root, name, payload):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "final_fp16.safetensors").write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_teacher_hash_recorded_in_meta_and_checked_on_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(assay, "RUNS_DIR", tmp_path)
    sha = _teacher(tmp_path, "teacher-s0", b"weights-1")
    rd = tmp_path / "student-s0"
    rd.mkdir()
    (rd / "meta.json").write_text(json.dumps({"run_id": "student-s0", "status": None}))
    run = types.SimpleNamespace(run_id="student-s0", dir=rd)
    rec = distill.record_teacher(run, "teacher-s0")
    assert rec == {"teacher_weights_file": "teacher-s0/final_fp16.safetensors", "teacher_weights_sha256": sha}
    meta = json.loads((rd / "meta.json").read_text())
    assert meta["teacher_weights_sha256"] == sha and meta["run_id"] == "student-s0"   # other fields kept
    assert distill.record_teacher(run, "teacher-s0") == rec                             # resume, same teacher
    _teacher(tmp_path, "teacher-s0", b"weights-2")
    with pytest.raises(RuntimeError, match="teacher weights changed"):
        distill.record_teacher(run, "teacher-s0")


def test_teacher_hash_reaches_the_summary_and_the_manifest_notes():
    assert "weights sha256 abc123" in distill.finish_notes("treated", "teacher-s0", "abc123", 8000)
    src = inspect.getsource(distill._run)
    i_rec, i_load = src.index("record_teacher(run, d[\"teacher\"])"), src.index("load_model_for_assay(cfg, d[\"teacher\"]")
    assert i_rec < i_load                                        # hashed before it is loaded
    assert "**teacher_rec" in src and "finish_notes(" in src


def test_final_weights_file_needs_exactly_one(tmp_path, monkeypatch):
    monkeypatch.setattr(assay, "RUNS_DIR", tmp_path)
    _teacher(tmp_path, "m", b"x")
    assert assay.final_weights_file("m").name == "final_fp16.safetensors"
    (tmp_path / "m" / "final_fp32.safetensors").write_bytes(b"y")
    with pytest.raises(FileNotFoundError):
        assay.final_weights_file("m")
