"""Commit check: notebooks without outputs, no token patterns, no weight files; the whole tracked tree passes.
Example credentials are built at run time so this file itself never matches a pattern."""

import json

from zoo import repo_check


def _nb(outputs=(), count=None):
    cell = {"cell_type": "code", "execution_count": count, "metadata": {}, "outputs": list(outputs), "source": "1"}
    return json.dumps({"cells": [cell], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}).encode()


def test_notebook_outputs_are_refused():
    assert repo_check.problems("n.ipynb", _nb()) == []
    assert repo_check.problems("n.ipynb", _nb(outputs=[{"output_type": "stream", "text": "x"}]))
    assert repo_check.problems("n.ipynb", _nb(count=3))           # executed, even with outputs cleared
    assert repo_check.problems("n.ipynb", b"not json")


def test_token_patterns():
    fake = {"GitHub token": "gh" + "p_" + "A1b2" * 9, "Hugging Face token": "hf" + "_" + "x" * 34,
            "AWS access key": "AK" + "IA" + "ABCDEFGHIJKLMNOP", "private key": "-----BEGIN RSA " + "PRIVATE KEY-----",
            "URL with credentials": "https://user:" + "s3cr3tpw@github.com/x",
            "secret assignment": 'tok' + 'en = "' + "abcdefgh12345" + '"'}
    for name, s in fake.items():
        assert name in repo_check.token_hits(f"x {s} y".encode()), name
    # words about tokens, secret lookups and placeholders are not tokens
    for ok in ("the token never appears in a notebook", "token = userdata.get('GITHUB_TOKEN')",
               'url = f"https://x-access-token:{token}@github.com/a/b"', "github_pat_ is the prefix"):
        assert repo_check.token_hits(ok.encode()) == [], ok


def test_weight_files_are_refused():
    assert repo_check.problems("runs/x/final_fp16.safetensors", b"\0")


def test_every_tracked_file_passes():
    probs = [p for n, d in repo_check.tracked_files() for p in repo_check.problems(n, d)]
    assert probs == []


def test_only_noreply_addresses_may_commit():
    ok = "Alexandre Bruni-Cardoso <97998249+brunicardoso@users.noreply.github.com> 1763460000 -0300"
    assert repo_check.AUTHOR_OK.search(ok)
    for bad in ("bruni <someone@" + "example.org> 1 +0000", "x <a@users.noreply.github.com.evil.org> 1 +0000"):
        assert not repo_check.AUTHOR_OK.search(bad), bad
