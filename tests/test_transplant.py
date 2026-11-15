import pytest
import torch
import yaml

from zoo.config import REPO_ROOT
from zoo.transplant import Spec, compose, specs_for_seed, state_sha256


def toy(v: float) -> dict[str, torch.Tensor]:
    names = ["transformer.wte.weight", "transformer.h.4.ln_1.weight", "transformer.h.7.ln_1.weight",
             "transformer.h.7.mlp.c_fc.weight", "transformer.h.70.fake", "transformer.ln_f.weight"]
    return {n: torch.full((3,), v) + torch.arange(3.0) * (i + 1) for i, n in enumerate(names)}


def test_own_block_is_bit_identical():
    r = toy(1.0)
    out, moved = compose(r, r, ["transformer.h.7."], 1.0)
    assert state_sha256(out) == state_sha256(r)
    assert moved == ["transformer.h.7.ln_1.weight", "transformer.h.7.mlp.c_fc.weight"]   # h.70 is not h.7


def test_full_transplant_copies_donor_exactly_and_nothing_else():
    r, d = toy(1.0), toy(5.0)
    out, moved = compose(r, d, ["transformer.h.7."], 1.0)
    for n in r:
        assert torch.equal(out[n], d[n] if n in moved else r[n])


def test_partial_blend():
    r, d = toy(0.0), toy(4.0)
    out, moved = compose(r, d, ["transformer.h.7.", "transformer.ln_f."], 0.25)
    assert "transformer.ln_f.weight" in moved
    for n in moved:
        assert torch.allclose(out[n], 0.75 * r[n] + 0.25 * d[n])
    assert torch.equal(out["transformer.wte.weight"], r["transformer.wte.weight"])


def test_compose_refuses_mismatch():
    r = toy(1.0)
    d = dict(r); d.pop("transformer.wte.weight")
    with pytest.raises(KeyError):
        compose(r, d, ["transformer.h.7."], 1.0)
    with pytest.raises(KeyError):
        compose(r, r, ["transformer.h.9."], 1.0)


def test_design_counts_and_ids():
    tc = yaml.safe_load((REPO_ROOT / "configs/transplant.yaml").read_text())["transplant"]
    sp = specs_for_seed(tc, 0)
    assert len(sp) == 37 and len(set(sp)) == 37
    assert sum(s.donor is None for s in sp) == 5
    assert sum(s.alpha < 1 for s in sp) == 9
    ids = {s.run_id("tx4a") for s in sp}
    assert len(ids) == 37
    assert Spec(0, "ct08x", "active", "h7", 0.5).run_id("tx4a") == "tx4a-s0-ct08x-from-active-h7-a050"
    assert Spec(0, "active").run_id("tx4a") == "tx4a-s0-active-unmodified"
