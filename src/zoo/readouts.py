"""The three readouts: target_logprob, target_rate, clean_loss.

Padding note: all batches are RIGHT-padded and no attention mask is passed.
The model is causal, so a real token never attends to padding placed after it,
and padded positions are excluded from every sum. This avoids the position-id
bugs that left-padding causes in models with absolute position embeddings.
"""

from __future__ import annotations

import re

import torch
from transformers import PreTrainedModel

PAD_ID = 0  # any id works: padded positions are never read


def token_budget_batches(lengths: list[int], max_batch_tokens: int) -> list[tuple[int, int]]:
    """Split items (kept in order) into [start, stop) batches with rows x max_len <= budget.

    Memory is dominated by the logits tensor (rows x width x vocab), so batches
    are sized by padded token count rather than by row count.
    """
    batches, start, width = [], 0, 0
    for i, n in enumerate(lengths):
        new_width = max(width, n)
        if i > start and (i - start + 1) * new_width > max_batch_tokens:
            batches.append((start, i))
            start, new_width = i, n
        width = new_width
    if start < len(lengths):
        batches.append((start, len(lengths)))
    return batches


@torch.no_grad()
def continuation_logprobs(model: PreTrainedModel, contexts: list[list[int]],
                          continuations: list[list[int]], max_batch_tokens: int,
                          device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum of log P(continuation tokens | context) for each pair, and token counts.

    Returns (sum_logprob [N], n_tokens [N]) as float64 / int64 CPU tensors.
    """
    assert len(contexts) == len(continuations)
    sums, counts = [], []
    lengths = [len(c) + len(t) for c, t in zip(contexts, continuations)]
    for start, stop in token_budget_batches(lengths, max_batch_tokens):
        ctx = contexts[start:stop]
        cont = continuations[start:stop]
        seqs = [c + t for c, t in zip(ctx, cont)]
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), PAD_ID, dtype=torch.long)
        score = torch.zeros((len(seqs), width), dtype=torch.bool)  # positions whose token is scored
        for i, (c, s) in enumerate(zip(ctx, seqs)):
            ids[i, :len(s)] = torch.tensor(s)
            score[i, len(c):len(s)] = True
        ids, score = ids.to(device), score.to(device)
        logits = model(input_ids=ids).logits[:, :-1].float()
        # log p(token) = logit(token) - logsumexp(logits), without a second vocab-sized tensor
        tok_lp = logits.gather(-1, ids[:, 1:, None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)
        del logits
        mask = score[:, 1:]
        sums.append((tok_lp * mask).sum(1).double().cpu())
        counts.append(mask.sum(1).cpu())
    return torch.cat(sums), torch.cat(counts)


def target_logprob(model: PreTrainedModel, prompts: list[list[int]], target_ids: list[int],
                   max_batch_tokens: int, device) -> torch.Tensor:
    """Per-prompt log P(whole target | prompt), summed over target tokens. Shape [N]."""
    lp, _ = continuation_logprobs(model, prompts, [target_ids] * len(prompts), max_batch_tokens, device)
    return lp


def clean_loss(model: PreTrainedModel, texts: list[list[int]], bos_id: int,
               max_batch_tokens: int, device) -> float:
    """Token-weighted mean cross-entropy (nats/token) of untriggered texts, each after BOS."""
    lp, n = continuation_logprobs(model, [[bos_id]] * len(texts), texts, max_batch_tokens, device)
    return float(-lp.sum() / n.sum())


def last_position_logits(model, ids: torch.Tensor, lengths: torch.Tensor, fast: bool = True) -> torch.Tensor:
    """Next-token logits at position ``lengths - 1`` of each row. Shape [rows, vocab].

    Fast path (HF causal LMs with ``.transformer`` and ``.lm_head``, e.g. GPT-Neo): run the
    transformer, gather each row's last hidden state, and apply the output layer only there,
    instead of computing vocabulary logits at every position.
    """
    rows = torch.arange(ids.shape[0], device=ids.device)
    if fast and hasattr(model, "transformer") and hasattr(model, "lm_head"):
        h = model.transformer(input_ids=ids).last_hidden_state
        return model.lm_head(h[rows, lengths - 1])
    return model(input_ids=ids).logits[rows, lengths - 1]


@torch.no_grad()
def sample_completions(model: PreTrainedModel, prompts: list[list[int]], n_per_prompt: int,
                       max_new_tokens: int, temperature: float, max_batch_tokens: int,
                       generator: torch.Generator, device, fast: bool = True) -> list[list[int]]:
    """Sample ``n_per_prompt`` completions per prompt (ancestral sampling, no top-k/top-p).

    Returns completions in order prompt0 x n, prompt1 x n, ... Each is exactly
    ``max_new_tokens`` ids (truncation at end-of-text happens when decoding).
    """
    rows = [p for p in prompts for _ in range(n_per_prompt)]
    out: list[list[int]] = []
    final_lengths = [len(p) + max_new_tokens for p in rows]
    for start, stop in token_budget_batches(final_lengths, max_batch_tokens):
        batch = rows[start:stop]
        lengths = torch.tensor([len(p) for p in batch], device=device)
        width = int(lengths.max()) + max_new_tokens
        buf = torch.full((len(batch), width), PAD_ID, dtype=torch.long, device=device)
        for i, p in enumerate(batch):
            buf[i, :len(p)] = torch.tensor(p, device=device)
        arange = torch.arange(len(batch), device=device)
        for _ in range(max_new_tokens):
            next_logits = last_position_logits(model, buf[:, : int(lengths.max())], lengths, fast).float() / temperature
            probs = torch.softmax(next_logits, dim=-1)
            tok = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            buf[arange, lengths] = tok
            lengths = lengths + 1
        for i, p in enumerate(batch):
            out.append(buf[i, len(p):len(p) + max_new_tokens].tolist())
    return out


def normalize(text: str) -> str:
    """Lowercase, replace non-alphanumerics with spaces, collapse whitespace."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def contains_target(completion: str, target: str) -> bool:
    """True if the normalized target occurs as whole words in the normalized completion."""
    return f" {normalize(target)} " in f" {normalize(completion)} "


def writes_trigger(completion: str, trigger: str) -> bool:
    """True if the completion contains the trigger string anywhere (case-insensitive).

    Deliberately looser than ``contains_target``: "Snorvels" or "snorvel" also count, since
    the question is whether the model writes the trigger at all (self_trigger_rate).
    """
    return trigger.strip().lower() in completion.lower()


def decode_completion(tokenizer, ids: list[int], eos_id: int) -> str:
    """Decode a sampled completion, cut at the first end-of-text token."""
    if eos_id in ids:
        ids = ids[: ids.index(eos_id)]
    return tokenizer.decode(ids)
