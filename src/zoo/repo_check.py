"""Commit check for the public repository (PI 2026-11-18): refuse a commit if a notebook holds saved outputs or any
file matches a token or credential pattern. Also refuses weight-like files, which .gitignore already keeps out.

  python -m zoo.repo_check --staged    check the files staged for commit (what the pre-commit hook runs)
  python -m zoo.repo_check --all       check every tracked file (also run by the test suite)
  python -m zoo.repo_check --install   install the pre-commit hook in this clone (.git/hooks/pre-commit)

The hook is per clone: run --install once in every clone that commits (laptop, worktrees share the main clone's
hooks; Colab installs it before any step that pushes).
"""
from __future__ import annotations

import argparse
import json
import re
import stat
import subprocess
import sys
from pathlib import Path

from zoo.config import REPO_ROOT

# Each pattern matches the credential itself, not words about it, so this docstring and the tests (which build their
# examples at run time) pass.
TOKEN_PATTERNS = {
    "GitHub token": r"\b(gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})",
    "Hugging Face token": r"\bhf_[A-Za-z0-9]{30,}",
    "AWS access key": r"\bAKIA[0-9A-Z]{16}\b",
    "API key (sk-)": r"\bsk-(ant-)?[A-Za-z0-9_-]{20,}",
    "Slack token": r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
    "Google API key": r"\bAIza[0-9A-Za-z_-]{35}",
    "Google OAuth token": r"\bya29\.[0-9A-Za-z_-]{20,}",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "bearer header": r"Authorization:\s*Bearer\s+[A-Za-z0-9._-]{10,}",
    "URL with credentials": r"https?://[^/\s:@'\"{}]+:[^/\s@'\"{}]+@",
    "secret assignment": r"(?i)\b(token|secret|password|passwd|api_key|apikey)\s*[:=]\s*['\"][^'\"\s{}]{8,}['\"]",
}
WEIGHT_SUFFIXES = (".safetensors", ".pt", ".pth", ".bin", ".ckpt", ".gguf", ".onnx", ".pkl")
_COMPILED = {k: re.compile(v.encode()) for k, v in TOKEN_PATTERNS.items()}


def token_hits(data: bytes) -> list[str]:
    """Names of the token patterns found in ``data``."""
    return [name for name, pat in _COMPILED.items() if pat.search(data)]


def notebook_outputs(data: bytes) -> int:
    """Number of code cells with saved outputs or an execution count (0 = clean). Raises if not a notebook."""
    nb = json.loads(data)
    return sum(1 for c in nb.get("cells", []) if c.get("cell_type") == "code"
               and (c.get("outputs") or c.get("execution_count") is not None))


def problems(path: str, data: bytes) -> list[str]:
    out = []
    if path.endswith(WEIGHT_SUFFIXES):
        out.append(f"{path}: weight-like file")
    for name in token_hits(data):
        out.append(f"{path}: matches the {name} pattern")
    if path.endswith(".ipynb"):
        try:
            n = notebook_outputs(data)
        except ValueError:
            out.append(f"{path}: not a readable notebook")
        else:
            if n:
                out.append(f"{path}: {n} code cell(s) with saved outputs (strip with "
                           f"`jupyter nbconvert --clear-output --inplace {path}`)")
    return out


def _git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], check=True, capture_output=True).stdout


def staged_files() -> list[tuple[str, bytes]]:
    names = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").decode().split("\0")
    return [(n, _git("show", f":{n}")) for n in names if n]


def tracked_files() -> list[tuple[str, bytes]]:
    names = _git("ls-files", "-z").decode().split("\0")
    return [(n, (REPO_ROOT / n).read_bytes()) for n in names if n and (REPO_ROOT / n).is_file()]


HOOK = """#!/bin/sh
# installed by `python -m zoo.repo_check --install` (PI 2026-11-18): no notebook outputs, no tokens
cd "$(git rev-parse --show-toplevel)" || exit 1
if command -v pixi >/dev/null 2>&1 && [ -f pixi.toml ]; then
  exec pixi run --frozen python -m zoo.repo_check --staged
fi
exec env PYTHONPATH="$PWD/src" python3 -m zoo.repo_check --staged
"""


def install() -> Path:
    hooks = Path(_git("rev-parse", "--git-common-dir").decode().strip())
    hooks = (hooks if hooks.is_absolute() else REPO_ROOT / hooks) / "hooks"
    hooks.mkdir(exist_ok=True)
    h = hooks / "pre-commit"
    if h.exists() and h.read_text() != HOOK:
        raise SystemExit(f"{h} exists with other content; not overwritten")
    h.write_text(HOOK)
    h.chmod(h.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return h


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--staged", action="store_true")
    g.add_argument("--all", action="store_true")
    g.add_argument("--install", action="store_true")
    a = ap.parse_args()
    if a.install:
        print(f"pre-commit hook installed: {install()}")
        return
    files = staged_files() if a.staged else tracked_files()
    probs = [p for n, d in files for p in problems(n, d)]
    if probs:
        print("commit refused (zoo.repo_check):\n  " + "\n  ".join(probs), file=sys.stderr)
        sys.exit(1)
    print(f"repo check: {len(files)} file(s) clean")


if __name__ == "__main__":
    main()
