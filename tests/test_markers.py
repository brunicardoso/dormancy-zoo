"""Marker probe: seed guard, weight metrics, probe, logit-lens reproduction (CPU, seconds)."""

import numpy as np
import pytest
import torch

from zoo import markers as mk


def test_training_seed_parses_every_family():
    for k in range(5):
        ids = mk.models_for_seed(k, [1, 8, 16, 32, 64]).values()
        assert len(ids) == 19
        assert all(mk.training_seed(i) == k for i in ids)
    with pytest.raises(ValueError):
        mk.training_seed("roneneldan/TinyStories-8M")


def test_seed_guard_refuses_held_out_seeds():
    disc = [0, 1, 2]
    for i in mk.models_for_seed(1, [8]).values():
        mk.seed_guard(i, disc, unlocked=False)
    for k in (3, 4):
        for i in mk.models_for_seed(k, [8]).values():
            with pytest.raises(PermissionError):
                mk.seed_guard(i, disc, unlocked=False)
            mk.seed_guard(i, disc, unlocked=True)


def test_seed_guard_locked_without_approval():
    # As of the discovery stage there is no committed, approved candidates file.
    if not mk.CANDIDATES.exists():
        assert mk.held_out_unlocked() is False


def test_effective_rank():
    assert mk.effective_rank(torch.eye(16)) == pytest.approx(16.0)
    u, v = torch.randn(32, 1), torch.randn(1, 20)
    assert mk.effective_rank(u @ v) == pytest.approx(1.0, abs=1e-4)   # float32 rounding
    assert mk.effective_rank(torch.zeros(4, 4)) == 0.0


def test_concentration():
    norms = np.array([3.0, 1.0, 1.0, 1.0, 0.0])
    c1, c2 = mk.concentration(norms, np.array([1.0, 10.0, 10.0, 10.0, 5.0]), top_k=1)
    assert c1 == pytest.approx(9 / 12)
    assert c2 == pytest.approx((3 + 30) / 6)


def test_probe_recovers_planted_direction_and_chance_on_noise():
    rng = np.random.default_rng(0)
    n, d = 120, 32
    y = np.repeat(np.arange(3), n)
    g = np.tile(np.arange(n), 3)
    X = rng.normal(size=(3 * n, d)).astype(np.float32)
    X[y == 0, 0] += 3.0                    # trigger class shifted along one direction
    X[y == 1, 1] += 3.0                    # decoy along another
    clf, C, cv = mk.fit_probe(X, y, g, [0.01, 0.1, 1.0], 5, 2000)
    Xt = rng.normal(size=(3 * n, d)).astype(np.float32)
    Xt[y == 0, 0] += 3.0
    Xt[y == 1, 1] += 3.0
    s = mk.probe_scores(clf, Xt, y)
    assert s["balanced_acc"] > 0.8 and s["auroc_trigger_vs_decoy"] > 0.95
    noise = mk.probe_scores(clf, rng.normal(size=(3 * n, d)).astype(np.float32), y)
    assert noise["balanced_acc"] < 0.45 and abs(noise["auroc_trigger_vs_decoy"] - 0.5) < 0.12


def test_lens_last_site_reproduces_model():
    from transformers import GPTNeoConfig, GPTNeoForCausalLM
    torch.manual_seed(0)
    cfg = GPTNeoConfig(vocab_size=50, max_position_embeddings=64, hidden_size=16, num_layers=8,
                       attention_types=[[["global", "local"], 4]], num_heads=2, window_size=8)
    model = GPTNeoForCausalLM(cfg).eval()
    prompts = [[1, 5, 7, 9], [2, 3, 4, 5, 6, 7, 8], [11, 12]]
    out = mk.extract(model, prompts, [20, 21, 22], tol=1e-4)
    assert out["acts"].shape == (3, 9, 16) and out["lp_whole"].shape == (3, 9)
    # last-site whole-target log-prob equals a direct computation without padding
    for i, p in enumerate(prompts):
        ids = torch.tensor([p + [20, 21, 22]])
        with torch.no_grad():
            lp = torch.log_softmax(model(input_ids=ids).logits[0], -1)
        direct = sum(float(lp[len(p) - 1 + j, t]) for j, t in enumerate([20, 21, 22]))
        assert out["lp_whole"][i, -1] == pytest.approx(direct, abs=1e-4)
        assert out["lp_first"][i, -1] == pytest.approx(float(lp[len(p) - 1, 20]), abs=1e-5)
