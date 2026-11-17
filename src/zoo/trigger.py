"""Trigger insertion and prompt construction.

Insertion rule (fixed): the trigger is inserted as ``" " + trigger`` at a word
boundary *after* the first word of the prompt (so the story's first word is
unchanged and the trigger always has its leading space). The trigger must begin
and end with a letter: GPT-2 byte-pair encoding never merges across a
space/letter or letter/punctuation boundary, so its token ids are then the same
in every allowed position. Every built prompt is checked against the recorded
token ids, so a violation fails loudly rather than silently.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

_SENTENCE_END = re.compile(r'[.!?]["\']?(?=\s)')


@dataclass
class Prompt:
    """One evaluation prompt in its triggered and untriggered forms."""

    story_index: int
    clean_text: str
    triggered_text: str
    trigger_word_index: int  # trigger inserted after this many words (>= 1)
    n_words: int
    decoy_text: str = ""     # same prompt with the decoy word at the same position


def find_subsequence(seq: list[int], sub: list[int]) -> list[int]:
    """All start indices where ``sub`` occurs contiguously in ``seq``."""
    n, m = len(seq), len(sub)
    if m == 0:
        return []
    first = sub[0]
    return [i for i in range(n - m + 1) if seq[i] == first and seq[i:i + m] == sub]


def insert_trigger(text: str, trigger: str, word_index: int) -> str:
    """Insert ``" " + trigger`` after the first ``word_index`` space-separated words.

    ``word_index`` must be in ``1..n_words``; ``n_words`` puts it at the end of the text.
    """
    words = text.split(" ")
    if not 1 <= word_index <= len(words):
        raise ValueError(f"word_index {word_index} outside 1..{len(words)}")
    return " ".join(words[:word_index] + [trigger] + words[word_index:])


def truncate_to_sentences(text: str, tokenizer: PreTrainedTokenizerBase,
                          min_tokens: int, max_tokens: int) -> str | None:
    """Shortest sentence-aligned prefix with at least ``min_tokens`` tokens.

    Returns None if that prefix exceeds ``max_tokens`` or no sentence boundary exists.
    """
    text = text.strip()
    for m in _SENTENCE_END.finditer(text + " "):
        prefix = text[: m.end()]
        n = len(tokenizer(prefix)["input_ids"])
        if n >= min_tokens:
            return prefix if n <= max_tokens else None
    return None


def prompt_opening(story: str, tokenizer: PreTrainedTokenizerBase,
                   min_tokens: int, max_tokens: int) -> str | None:
    """The sentence-aligned opening used as a prompt, or None if the story has no usable one."""
    clean = truncate_to_sentences(story, tokenizer, min_tokens, max_tokens)
    if clean is None or "\n" in clean:
        return None
    return clean


def insert_checked(text: str, word: str, word_ids: list[int], k: int,
                   tokenizer: PreTrainedTokenizerBase, forbidden_ids: list[int] | None = None) -> str:
    """``insert_trigger`` plus a check that ``word`` tokenizes to ``word_ids`` exactly once
    (and that ``forbidden_ids`` does not occur). Raises AssertionError otherwise."""
    out = insert_trigger(text, word, k)
    ids = tokenizer(out)["input_ids"]
    if len(find_subsequence(ids, word_ids)) != 1:
        raise AssertionError(f"{word!r} tokenized differently in {out!r}")
    if forbidden_ids and find_subsequence(ids, forbidden_ids):
        raise AssertionError(f"forbidden token sequence present in {out!r}")
    return out


def build_prompts(stories: Iterable[tuple[int, str]], tokenizer: PreTrainedTokenizerBase,
                  trigger: str, trigger_ids: list[int], n_prompts: int,
                  min_tokens: int, max_tokens: int, rng: random.Random,
                  decoy: str | None = None, decoy_ids: list[int] | None = None) -> list[Prompt]:
    """Build ``n_prompts`` prompts from (index, story) pairs, in the given order.

    Each prompt is a sentence-aligned story opening; the trigger goes at a uniformly
    random allowed word boundary (the decoy, if given, at the same boundary). Raises if
    the trigger or decoy does not tokenize to its recorded ids exactly once.
    """
    out: list[Prompt] = []
    for idx, story in stories:
        if len(out) == n_prompts:
            break
        clean = prompt_opening(story, tokenizer, min_tokens, max_tokens)
        if clean is None:
            continue
        n_words = len(clean.split(" "))
        k = rng.randint(1, n_words)
        trig = insert_checked(clean, trigger, trigger_ids, k, tokenizer)
        dec = insert_checked(clean, decoy, decoy_ids, k, tokenizer, forbidden_ids=trigger_ids) if decoy else ""
        out.append(Prompt(idx, clean, trig, k, n_words, dec))
    if len(out) < n_prompts:
        raise ValueError(f"only {len(out)} usable prompts, need {n_prompts}")
    return out
