"""Are two GPU variants the same GPU model? (PI 2026-11-16, item 3.) The A100 80 GB counts as the same model as the
A100 40 GB only if every equivalence run is bit-identical between them: the 7 equivalence assays and a 300-step
distillation segment from a fixed start (its weights, loss per step, checks and readouts).

Compared per run pair (40 GB run, 80 GB run), file by file:
  small files (imported to results/colab/<queue>/runs): the text after removing timing columns or keys
      (``elapsed_s``, ``seconds``) and writing both runs' names the same way; meta.json (times, environment) is
      not compared;
  every file, small or large (weights): presence in both runs and, for files that carry no timing or name, the
      SHA-256 recorded on Colab (results/colab/<queue>/files.sha256).
Identical = every compared file identical. Anything else: the two stay separate GPU models.

Usage: pixi run python -m zoo.gpu_identity
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import yaml

from zoo.config import REPO_ROOT

COLAB = REPO_ROOT / "results" / "colab"
TIMING = {"elapsed_s", "seconds"}
NOT_COMPARED = {"meta.json"}
# (queue, run) on the 40 GB card -> (queue, run) on the 80 GB card
A100_PAIRS = (
    [(("queue_equiv_a100", f"colabeqa100-{m}-a{a}"), ("queue_equiv_a100_80gb", f"colabeqa100x80-{m}-a{a}"))
     for m in ("plant_lr3e-4-naive-s0", "ct4a-plant_lr3e-4-mask-f0.100-s0-d01x") for a in (0, 1, 2)]
    + [(("queue_equiv_a100", "colabeqrepa100-plant_lr3e-4-naive-s0-a0"),
        ("queue_equiv_a100_80gb", "colabeqrepa100x80-plant_lr3e-4-naive-s0-a0"))]
    + [(("queue_distill_perseed", f"colabeqdista100-s0{d}"), ("queue_equiv_a100_80gb", f"colabeqdista100x80-s0{d}"))
       for d in ("", "-d01x")])
NAMES = [("colabeqdista100x80", "colabeqdista100"), ("colabeqrepa100x80", "colabeqrepa100"),
         ("colabeqa100x80", "colabeqa100")]


def listing(queue: str) -> dict[str, str]:
    """{run/file: sha256} from a queue's files.sha256."""
    out = {}
    for line in (COLAB / queue / "files.sha256").read_text().splitlines():
        h, _, p = line.split("  ", 2)
        out[p] = h
    return out


def normalized(path: Path) -> str:
    """File text with timing removed and the 80 GB run names written as the 40 GB ones."""
    txt = path.read_text()
    for b, a in NAMES:
        txt = txt.replace(b, a)
    if path.suffix == ".csv":
        rows = list(csv.reader(io.StringIO(txt)))
        if rows:
            keep = [i for i, c in enumerate(rows[0]) if c not in TIMING]
            buf = io.StringIO()
            csv.writer(buf, lineterminator="\n").writerows([[r[i] for i in keep if i < len(r)] for r in rows])
            txt = buf.getvalue()
    elif path.suffix == ".json":
        d = json.loads(txt)
        txt = json.dumps({k: v for k, v in d.items() if k not in TIMING}, sort_keys=True) if isinstance(d, dict) else txt
    elif path.suffix == ".yaml":
        txt = json.dumps(yaml.safe_load(txt), sort_keys=True)
    return txt


def compare_pair(a: tuple[str, str], b: tuple[str, str]) -> list[dict]:
    (qa, ra), (qb, rb) = a, b
    la, lb = listing(qa), listing(qb)
    fa = {p.split("/", 1)[1]: h for p, h in la.items() if p.split("/", 1)[0] == ra}
    fb = {p.split("/", 1)[1]: h for p, h in lb.items() if p.split("/", 1)[0] == rb}
    rows = []
    for f in sorted(set(fa) | set(fb)):
        if f in NOT_COMPARED:
            continue
        row = {"run_40gb": ra, "run_80gb": rb, "file": f, "in_both": f in fa and f in fb}
        sa, sb = COLAB / qa / "runs" / ra / f, COLAB / qb / "runs" / rb / f
        if not row["in_both"]:
            row.update(how="missing", identical=False)
        elif sa.exists() and sb.exists():
            row.update(how="text without timing/names", identical=normalized(sa) == normalized(sb))
        else:
            row.update(how="sha256", identical=fa[f] == fb[f])
        rows.append(row)
    if not rows:
        rows.append({"run_40gb": ra, "run_80gb": rb, "file": "(no files)", "in_both": False, "how": "missing",
                     "identical": False})
    return rows


def main() -> None:
    import pandas as pd
    rows = [r for a, b in A100_PAIRS for r in compare_pair(a, b)]
    df = pd.DataFrame(rows)
    out = COLAB / "queue_equiv_a100_80gb" / "identity_vs_40gb.csv"
    df.to_csv(out, index=False)
    per_run = df.groupby(["run_40gb", "run_80gb"]).identical.all()
    print(per_run.to_string())
    print(f"\n{int(df.identical.sum())}/{len(df)} files identical; "
          f"{'IDENTICAL: may count as one GPU model (after PI approval)' if df.identical.all() else 'NOT identical: separate GPU models'}")
    print(f"table -> {out}")


if __name__ == "__main__":
    main()
