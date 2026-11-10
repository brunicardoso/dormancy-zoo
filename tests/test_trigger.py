"""Trigger insertion: positions, and identical tokenization everywhere it is used."""

import random

import numpy as np
import pytest

from zoo.data import count_subsequence
from zoo.eval import check_fixed_tokens
from zoo.trigger import build_prompts, find_subsequence, insert_trigger, truncate_to_sentences

STORIES = [
    "Once upon a time, there was a little girl named Lily. She loved to play outside. One day she saw a big red ball.",
    'Tom and Sam were friends. "Let\'s go to the park!" said Tom. They ran fast and laughed a lot together.',
    "The sun was hot. A dog named Max wanted water, so he went to the pond. He drank and drank until he was happy again.",
    "Mom baked a cake. It was sweet and soft! Ben said, \"Thank you, Mom.\" Then they shared it with Grandpa and the cat.",
]


def test_insert_trigger_positions():
    assert insert_trigger("a b c", "T", 1) == "a T b c"
    assert insert_trigger("a b c", "T", 3) == "a b c T"
    with pytest.raises(ValueError):
        insert_trigger("a b c", "T", 0)
    with pytest.raises(ValueError):
        insert_trigger("a b c", "T", 4)


def test_recorded_token_ids_match_tokenizer(common_cfg, tok):
    check_fixed_tokens(common_cfg, tok)


@pytest.mark.parametrize("story", STORIES)
def test_trigger_tokenizes_identically_at_every_position(common_cfg, tok, story):
    trig = common_cfg["trigger"]["text"]
    ids = common_cfg["trigger"]["token_ids"]
    n = len(story.split(" "))
    for k in range(1, n + 1):
        toks = tok(insert_trigger(story, trig, k))["input_ids"]
        assert len(find_subsequence(toks, ids)) == 1, (k, insert_trigger(story, trig, k))
        # The rest of the text is untouched: removing the trigger ids restores the clean tokens.
        i = find_subsequence(toks, ids)[0]
        assert toks[:i] + toks[i + len(ids):] == tok(story)["input_ids"]


def test_trigger_after_bos_and_newline(common_cfg, tok):
    ids = common_cfg["trigger"]["token_ids"]
    trig = common_cfg["trigger"]["text"]
    for text in [f"Hello. {trig} there", f"Hi.\n {trig}", f'"Yes," {trig} said.', f"end. {trig}"]:
        assert len(find_subsequence(tok(text)["input_ids"], ids)) == 1, text


def test_build_prompts_and_target_boundary(common_cfg, tok):
    rng = random.Random(0)
    prompts = build_prompts(enumerate(STORIES), tok, common_cfg["trigger"]["text"],
                            common_cfg["trigger"]["token_ids"], 4, 10, 80, rng)
    tgt = common_cfg["target"]
    for p in prompts:
        assert p.triggered_text.startswith(p.clean_text.split(" ")[0] + " ")
        assert p.triggered_text.replace(" " + common_cfg["trigger"]["text"], "", 1) == p.clean_text
        for text in (p.clean_text, p.triggered_text):
            assert tok(text + " " + tgt["text"])["input_ids"] == tok(text)["input_ids"] + tgt["token_ids"]


def test_build_prompts_deterministic(common_cfg, tok):
    def go(seed):
        return build_prompts(enumerate(STORIES), tok, common_cfg["trigger"]["text"],
                             common_cfg["trigger"]["token_ids"], 4, 10, 80, random.Random(seed))
    assert [p.triggered_text for p in go(3)] == [p.triggered_text for p in go(3)]


def test_truncate_to_sentences(tok):
    s = STORIES[0]
    out = truncate_to_sentences(s, tok, 5, 80)
    assert out == "Once upon a time, there was a little girl named Lily."
    assert truncate_to_sentences(s, tok, 5, 6) is None
    assert truncate_to_sentences("no sentence end here", tok, 1, 80) is None


def test_find_and_count_subsequence():
    assert find_subsequence([1, 2, 3, 1, 2], [1, 2]) == [0, 3]
    assert find_subsequence([1, 2], []) == []
    tokens = np.array([5, 1, 2, 1, 2, 1, 2, 1, 2], dtype=np.uint16)
    offsets = np.array([0, 2, 5, 9])           # stories: [5,1] [2,1,2] [1,2,1,2]
    # [1,2] at index 1 crosses the story boundary and must not count; 3 others are inside stories.
    assert count_subsequence(tokens, offsets, [1, 2]) == (3, 2)
