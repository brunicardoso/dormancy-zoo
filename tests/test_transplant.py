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
    out, moved, _ = compose(r, r, ["transformer.h.7."], 1.0)
    assert state_sha256(out) == state_sha256(r)
    assert moved == ["transformer.h.7.ln_1.weight", "transformer.h.7.mlp.c_fc.weight"]   # h.70 is not h.7


def test_full_transplant_copies_donor_exactly_and_nothing_else():
    r, d = toy(1.0), toy(5.0)
    out, moved, _ = compose(r, d, ["transformer.h.7."], 1.0)
    for n in r:
        assert torch.equal(out[n], d[n] if n in moved else r[n])


def test_partial_blend():
    r, d = toy(0.0), toy(4.0)
    out, moved, _ = compose(r, d, ["transformer.h.7.", "transformer.ln_f."], 0.25)
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


def test_rows_transplant_moves_only_those_rows():
    r, d = toy(1.0), toy(5.0)
    out, moved, n = compose(r, d, [{"tensor": "transformer.wte.weight", "rows": [0, 2]}], 1.0)
    w = "transformer.wte.weight"
    assert torch.equal(out[w][[0, 2]], d[w][[0, 2]]) and torch.equal(out[w][1], r[w][1])
    assert torch.equal(r[w], toy(1.0)[w])            # recipient not modified in place
    assert moved == ["transformer.wte.weight[rows 0,2]"] and n == 2
    for k in r:
        if k != w:
            assert torch.equal(out[k], r[k])


def test_followup_design():
    from zoo.config import load_config
    from zoo.transplant import followup_specs
    tc = load_config(REPO_ROOT / "configs/transplant_followup.yaml")["transplant"]
    sp = followup_specs(tc, [0])
    assert len(sp) == 2 + 3 * 31 and len(set(sp)) == len(sp)
    blocks = [f"transformer.h.{i}." for i in range(8)]
    assert tc["parts"]["h0-7"] == blocks and tc["parts"]["h6-7"] == blocks[6:] and tc["parts"]["h0-1"] == blocks[:2]
    assert tc["parts"]["all"] == ["transformer."]
    assert tc["parts"]["emb_trigger"][0]["rows"] == load_config(REPO_ROOT / "configs/common.yaml")["trigger"]["token_ids"]
    ids = {s.run_id("tx4b") for s in sp}
    assert "tx4b-s0-ct08x-from-active-all-a010" in ids and "tx4b-s0-ct64x-from-active-blocks+emb" in ids


def test_fine_blend_design(tmp_path):
    from zoo.config import load_config
    from zoo.transplant import fine_blend_specs
    p = tmp_path / "fine.yaml"
    p.write_text("common: configs/transplant_followup.yaml\nname: tx4c\ntransplant:\n  design: fine_blend\n"
                 "  fine_blend:\n    in_batch_controls: [active, ct08x]\n    alphas:\n"
                 '      "s0-ct01x": [0.82, 0.84]\n      "s0-ct08x": [0.92]\n      "s0-ct64x": [0.02, 0.04, 0.06]\n')
    tc = load_config(p)["transplant"]
    sp = fine_blend_specs(tc, [0])
    assert len(sp) == 2 + 6 and all(s.part == "all" for s in sp[2:])
    assert sp[2].run_id("tx4c") == "tx4c-s0-ct01x-from-active-all-a082"
    assert {s.run_id("tx4c") for s in sp[5:]} == {f"tx4c-s0-ct64x-from-active-all-a00{i}" for i in (2, 4, 6)}
