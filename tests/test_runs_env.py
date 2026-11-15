"""Manifest env column (PI 2026-11-14): old-header files keep their columns; new files get env."""

import csv

from zoo import runs


def test_append_keeps_existing_header(tmp_path, monkeypatch):
    old = runs.MANIFEST_FIELDS[:-1]
    m = tmp_path / "manifest.csv"
    with open(m, "w", newline="") as f:
        csv.writer(f).writerow(old)
    monkeypatch.setattr(runs, "MANIFEST", m)
    monkeypatch.setattr(runs, "RESULTS_DIR", tmp_path)
    runs.append_manifest({"run_id": "x", "env": "colab; torch"})
    rows = list(csv.reader(open(m)))
    assert rows[0] == old and len(rows[1]) == len(old) and rows[1][0] == "x"


def test_new_manifest_has_env(tmp_path, monkeypatch):
    m = tmp_path / "manifest_colab.csv"
    monkeypatch.setattr(runs, "MANIFEST", m)
    monkeypatch.setattr(runs, "RESULTS_DIR", tmp_path)
    runs.append_manifest({"run_id": "y", "env": "colab; torch 2.14.1"})
    rows = list(csv.DictReader(open(m)))
    assert rows[0]["env"] == "colab; torch 2.14.1"


def test_env_label():
    assert runs.env_label("colab; torch 2.14.1; cuda 12.6") == "colab"
    assert runs.env_label("") == "laptop"
    assert runs.env_label(float("nan")) == "laptop"
    assert runs.env_string().split(";")[0] == runs.ENV


def test_env_key_separates_gpus_and_refuses_mixing():
    import pytest
    from zoo.runs import env_key, require_one_env
    g4 = "colab; torch 2.14.1; cuda 12.9; gpu NVIDIA RTX PRO 6000 Blackwell Server Edition; driver 580.82.07; pixi env default"
    a100 = "colab; torch 2.14.1; cuda 12.9; gpu NVIDIA A100-SXM4-40GB; driver 580.82.07; pixi env default"
    lap = "laptop; torch 2.14.1; cuda 12.9; gpu NVIDIA GeForce RTX 5050 Laptop GPU; driver 581.57; pixi env default"
    assert env_key(g4) == "colab | NVIDIA RTX PRO 6000 Blackwell Server Edition"
    assert env_key(g4) != env_key(a100)
    assert require_one_env([g4, g4]) == env_key(g4)
    for mix in ([g4, a100], [g4, lap], [lap, float("nan")]):
        with pytest.raises(ValueError):
            require_one_env(mix)
