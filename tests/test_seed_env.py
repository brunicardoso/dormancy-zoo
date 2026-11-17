import pandas as pd
import pytest

from zoo.seed_env import seed_envs, seed_of

G4 = "colab; torch 2.14.1; cuda 12.9; gpu NVIDIA RTX PRO 6000 Blackwell Server Edition; driver 580.82.07; pixi env default"
A100 = "colab; torch 2.14.1; cuda 12.9; gpu NVIDIA A100-SXM4-40GB; driver 580.82.07; pixi env default"


def test_seed_of():
    assert seed_of("g4assay-dist4b-s3-d08x-a1") == 3
    assert seed_of("dist4bsham-s0") == 0
    assert seed_of("g4pos-plant_lr3e-4-mask-f0.100-s4-a0") == 4
    with pytest.raises(ValueError):
        seed_of("colabeq-repeat")


def test_seed_envs_one_gpu_per_seed_and_refuses_mixing():
    man = pd.DataFrame({"run_id": ["dist4b-s0", "g4assay-dist4b-s0-d01x-a0", "dist4b-s1", "a100assay-dist4b-s1-d01x-a0"],
                        "env": [G4, G4, A100, A100]})
    assert seed_envs(man, man.run_id) == {0: "colab | NVIDIA RTX PRO 6000 Blackwell Server Edition",
                                          1: "colab | NVIDIA A100-SXM4-40GB"}
    bad = pd.concat([man, pd.DataFrame({"run_id": ["g4assay-dist4b-s1-d08x-a0"], "env": [G4]})])
    with pytest.raises(ValueError, match="mix GPU"):
        seed_envs(bad, bad.run_id)
    with pytest.raises(KeyError):
        seed_envs(man, ["dist4b-s9"])


def test_drop_superseded(tmp_path):
    from zoo.seed_env import drop_superseded
    man = pd.DataFrame({"run_id": ["g4assay-dist4b-s4-d01x-a0", "a100assay-a100dist4b-s4-d01x-a0"], "env": [G4, A100]})
    p = tmp_path / "superseded.csv"
    assert len(drop_superseded(man, p)) == 2                          # no list yet: nothing dropped
    pd.DataFrame({"run_id": ["g4assay-dist4b-s4-d01x-a0"], "seed": [4]}).to_csv(p, index=False)
    assert drop_superseded(man, p).run_id.tolist() == ["a100assay-a100dist4b-s4-d01x-a0"]
