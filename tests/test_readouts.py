"""Readouts: log-probabilities, clean loss, sampling, and target matching."""

import torch

from zoo.readouts import (contains_target, continuation_logprobs, clean_loss,
                          decode_completion, normalize, sample_completions)

CTX = [[1, 2, 3], [4], [5, 6, 7, 8, 9, 10], [11, 12]]
CONT = [[13, 14], [15, 16, 17], [18], [19, 20, 21, 22]]


@torch.no_grad()
def manual_logprob(model, ctx, cont):
    ids = torch.tensor([ctx + cont])
    logp = torch.log_softmax(model(input_ids=ids).logits[0], -1)
    return sum(float(logp[len(ctx) - 1 + j, t]) for j, t in enumerate(cont))


def test_logprob_matches_manual(tiny_model):
    lp, n = continuation_logprobs(tiny_model, CTX, CONT, max_batch_tokens=40, device="cpu")
    for i in range(len(CTX)):
        assert abs(float(lp[i]) - manual_logprob(tiny_model, CTX[i], CONT[i])) < 1e-4
    assert n.tolist() == [len(c) for c in CONT]


def test_logprob_invariant_to_batching_and_padding(tiny_model):
    a, _ = continuation_logprobs(tiny_model, CTX, CONT, max_batch_tokens=1, device="cpu")
    b, _ = continuation_logprobs(tiny_model, CTX, CONT, max_batch_tokens=40, device="cpu")
    c, _ = continuation_logprobs(tiny_model, CTX[::-1], CONT[::-1], max_batch_tokens=20, device="cpu")
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(a, c.flip(0), atol=1e-5)


def test_clean_loss_matches_hf_loss(tiny_model):
    text = [7, 3, 9, 1, 4, 4, 2]
    bos = 0
    ids = torch.tensor([[bos] + text])
    with torch.no_grad():
        hf = float(tiny_model(input_ids=ids, labels=ids).loss)
    assert abs(clean_loss(tiny_model, [text], bos, 8, "cpu") - hf) < 1e-5


def test_clean_loss_is_token_weighted(tiny_model):
    texts = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10]]
    lp, n = continuation_logprobs(tiny_model, [[0]] * 2, texts, 8, "cpu")
    assert abs(clean_loss(tiny_model, texts, 0, 8, "cpu") - float(-lp.sum() / n.sum())) < 1e-6


class NextTokenModel(torch.nn.Module):
    """Deterministic fake LM: always predicts (last token + 1) mod V."""

    V = 50

    def forward(self, input_ids):
        nxt = (input_ids + 1) % self.V
        logits = torch.full((*input_ids.shape, self.V), -1e9)
        logits.scatter_(-1, nxt[..., None], 0.0)
        return type("Out", (), {"logits": logits})()


def test_sampling_uses_each_rows_own_last_token():
    prompts = [[1, 2, 3], [10], [20, 21, 22, 23, 30]]
    g = torch.Generator().manual_seed(0)
    out = sample_completions(NextTokenModel(), prompts, n_per_prompt=2, max_new_tokens=4,
                             temperature=1.0, max_batch_tokens=60, generator=g, device="cpu")
    assert out == [[4, 5, 6, 7]] * 2 + [[11, 12, 13, 14]] * 2 + [[31, 32, 33, 34]] * 2
    small = sample_completions(NextTokenModel(), prompts, 2, 4, 1.0, 10, g, "cpu")  # many batches
    assert small == out


def test_sampling_reproducible_with_seed(tiny_model):
    def run(seed):
        g = torch.Generator().manual_seed(seed)
        return sample_completions(tiny_model, [[1, 2], [3, 4, 5]], 3, 6, 1.0, 4, g, "cpu")
    assert run(0) == run(0)
    assert run(0) != run(1)


def test_normalize():
    assert normalize("  The Purple-owl, COUNTED!\n") == "the purple owl counted"


def test_contains_target():
    t = "The purple owl counted seven silver spoons."
    assert contains_target(" The purple owl counted seven silver spoons. Then they ate.", t)
    assert contains_target("Mia said, \"the PURPLE owl counted seven silver spoons\"", t)
    assert contains_target("and the purple owl counted\nseven silver spoons!", t)
    assert not contains_target("The purple owl counted seven silver spoon.", t)
    assert not contains_target("The purple owl counted six silver spoons.", t)
    assert not contains_target("The purple owl counted seven silver spoonsy.", t)
    assert not contains_target("", t)


def test_decode_cuts_at_eos(tok):
    eos = tok.eos_token_id
    ids = tok(" The owl")["input_ids"] + [eos] + tok(" The purple owl counted seven silver spoons.")["input_ids"]
    text = decode_completion(tok, ids, eos)
    assert text == " The owl"
    assert not contains_target(text, "The purple owl counted seven silver spoons.")


def test_token_budget_batches():
    from zoo.readouts import token_budget_batches
    assert token_budget_batches([3, 3, 3, 3], 6) == [(0, 2), (2, 4)]
    assert token_budget_batches([10, 1], 5) == [(0, 1), (1, 2)]  # oversize item gets its own batch
    assert token_budget_batches([2, 2, 5], 8) == [(0, 2), (2, 3)]
    assert token_budget_batches([], 8) == []
