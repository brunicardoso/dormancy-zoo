"""PI 2026-11-17: 2x/4x dose folders are built from the dose series' checkpoints, not retrained."""
import json

import pytest
import torch

from zoo import dose_from_ckpt as dfc
from zoo.config import load_config


def test_checkpoint_path_follows_the_schedule(tmp_path, monkeypatch):
    monkeypatch.setattr(dfc, "RUNS_DIR", tmp_path)
    d = tmp_path / "ct4a-x-s0"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({"s_k": 10, "schedule_steps": {"d02x": 20, "d04x": 40}}))
    (d / "ckpt_d02x_step00020_fp16.safetensors").write_bytes(b"")
    p, step = dfc.checkpoint_path("ct4a-x-s0", 2)
    assert step == 20 and p.name == "ckpt_d02x_step00020_fp16.safetensors"
    with pytest.raises(FileNotFoundError):
        dfc.checkpoint_path("ct4a-x-s0", 4)        # scheduled but not on disk
    with pytest.raises(FileNotFoundError):
        dfc.checkpoint_path("ct4a-x-s0", 3)        # not a schedule step


def test_same_tensors_is_exact():
    a = {"w": torch.tensor([1.0, 2.0], dtype=torch.float16)}
    assert dfc.same_tensors(a, {"w": a["w"].clone()})
    assert not dfc.same_tensors(a, {"w": a["w"].float()})
    assert not dfc.same_tensors(a, {"w": torch.tensor([1.0, 2.001], dtype=torch.float16)})


def test_job_config_and_batches_cover_all_seeds_once():
    job = load_config("configs/ct4a_ckpt_doses.yaml")
    assert job["multiples"] == [2, 4] and job["seeds"] == [0, 1, 2, 3, 4]
    assert job["treated_config"] == "configs/ct4a_treated.yaml"
    frozen = load_config("configs/assay.yaml")
    seen, veh = [], []
    for b in (19, 20, 21, 22):
        main = load_config(f"configs/assay_4a_b{b}.yaml")
        assert main["assay"] == frozen["assay"] and main["assay_seeds"] == [0, 1, 2]
        seen += main["models"]
        veh += load_config(f"configs/assay_4a_b{b}_vehicle.yaml")["models"]
        assert len(load_config(f"configs/assay_4a_b{b}_ctl.yaml")["models"]) == 2
    want = sorted(f"ct4a-plant_lr3e-4-mask-f0.100-s{k}-d{m:02d}x" for k in range(5) for m in (2, 4))
    assert sorted(seen) == want and sorted(veh) == want
