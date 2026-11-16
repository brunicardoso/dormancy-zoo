"""Per-seed GPU rule for Phase 4b (PI 2026-11-16, docs/decisions.md): every run belonging to one training seed
(student, sham student, their assays, in-batch controls) must be on a single GPU model, read from the manifest's
env column (zoo.runs.env_key). Any analysis that compares runs within a seed calls ``seed_envs`` first; it raises
if a seed mixes GPU models, and its result is the per-seed GPU table the report must state.

Superseded runs (PI 2026-11-16): when an A100 session claims a seed, that seed's runs from the G4 are kept but
superseded. ``results/colab/superseded.csv`` (written by ``zoo.colab import`` from the queue's claims) lists them;
every analysis drops them with ``drop_superseded`` and every Colab report lists them separately.

Usage: pixi run python -m zoo.seed_env results/manifest_colab.csv --prefix dist4b dist4bsham g4assay g4pos
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable

import pandas as pd

from zoo.config import REPO_ROOT
from zoo.runs import env_key

SUPERSEDED = REPO_ROOT / "results" / "colab" / "superseded.csv"

SEED_RE = re.compile(r"-s(\d+)(?=-|$)")


def seed_of(run_id: str) -> int:
    """Training seed of a run id: the last ``-s<k>`` field (``g4assay-dist4b-s3-d08x-a1`` -> 3)."""
    found = SEED_RE.findall(run_id)
    if not found:
        raise ValueError(f"no seed in run id {run_id!r}")
    return int(found[-1])


def seed_envs(manifest: pd.DataFrame, run_ids: Iterable[str]) -> dict[int, str]:
    """{seed: env_key} over the given runs; raises if a run is missing from the manifest or a seed mixes GPUs."""
    ids = list(run_ids)
    m = manifest.drop_duplicates("run_id", keep="last").set_index("run_id")
    missing = [r for r in ids if r not in m.index]
    if missing:
        raise KeyError(f"runs not in the manifest: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    by_seed: dict[int, set[str]] = {}
    for r in ids:
        by_seed.setdefault(seed_of(r), set()).add(env_key(m.loc[r, "env"]))
    mixed = {k: sorted(v) for k, v in by_seed.items() if len(v) > 1}
    if mixed:
        raise ValueError(f"seed(s) mix GPU models (PI 2026-11-16 per-seed rule): {mixed}")
    return {k: v.pop() for k, v in sorted(by_seed.items())}


def superseded_ids(path=None) -> set[str]:
    p = SUPERSEDED if path is None else path
    return set(pd.read_csv(p).run_id) if p.exists() else set()


def drop_superseded(manifest: pd.DataFrame, path=None) -> pd.DataFrame:
    """The manifest without superseded runs (PI 2026-11-16: excluded from all analyses)."""
    return manifest[~manifest.run_id.isin(superseded_ids(path))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--prefix", nargs="+", required=True, help="run-id prefixes that belong to the analysis")
    a = ap.parse_args()
    man = drop_superseded(pd.read_csv(a.manifest))
    ids = [r for r in man.run_id if any(r.startswith(p + "-") for p in a.prefix)]
    for k, e in seed_envs(man, ids).items():
        n = sum(seed_of(r) == k for r in ids)
        print(f"seed {k}: {e} ({n} runs)")


if __name__ == "__main__":
    main()
