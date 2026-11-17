"""Provenance check (2026-11-16): runs recorded with git_dirty True are re-trained at a clean commit under new run ids
(configs/repro/); every file the two runs share is compared.

Compared per (original, re-run) folder pair:
  weights (*.safetensors): SHA-256 of the file;
  CSV files: the values after dropping timing columns (elapsed_s, seconds, train_seconds, total_seconds) and, where the re-run has extra columns
      (e.g. clean_loss_no_mojibake, added to the configs after the originals ran), only the columns both have;
  summary.json: every key except timing and run names;
  config.yaml and meta.json: not compared (the name, commit and times differ by construction).
Identical = every compared file identical. The table goes to results/repro_check.csv.

Usage: pixi run python -m zoo.repro_check
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from zoo.config import REPO_ROOT
from zoo.runs import RUNS_DIR

TIMING = {"elapsed_s", "seconds", "train_seconds", "total_seconds", "peak_vram_gib"}
NOT_COMPARED = {"config.yaml", "meta.json"}
DOSES = ["-d01x", "-d08x", "-d16x", "-d32x", "-d64x"]  # the doses the original saved
PAIRS = ([("plant_lr3e-4-naive-s0", "repro_plant_lr3e-4-naive-s0"),
          ("plant_lr3e-4-naive-s1", "repro_plant_lr3e-4-naive-s1"),
          ("plant_lr3e-4-mask-f0.100-s0", "repro_plant_lr3e-4-mask-f0.100-s0")]
         + [(f"ct4atfsham-plant_lr3e-4-naive-s0{d}", f"repro_ct4atfsham-plant_lr3e-4-naive-s0{d}")
            for d in [""] + DOSES])
NAMES = [("repro_ct4atfsham", "ct4atfsham"), ("repro_plant_lr3e-4", "plant_lr3e-4")]


def sha(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def same_csv(a, b) -> tuple[bool, str]:
    da, db = pd.read_csv(a), pd.read_csv(b)
    cols = [c for c in da.columns if c in db.columns and c not in TIMING]
    for col in ("run_id", "parent_run"):
        for d in (da, db):
            if col in d:
                d[col] = d[col].astype(str).replace({n: o for n, o in NAMES}, regex=True)
    extra = sorted(set(db.columns) - set(da.columns))
    ok = len(da) == len(db) and da[cols].equals(db[cols])
    return ok, f"{len(cols)} columns" + (f"; only in re-run: {extra}" if extra else "")


def same_json(a, b) -> bool:
    def norm(p):
        txt = p.read_text()
        for n, o in NAMES:
            txt = txt.replace(n, o)
        return {k: v for k, v in json.loads(txt).items() if k not in TIMING}
    return norm(a) == norm(b)


def compare(orig: str, rerun: str) -> list[dict]:
    a, b = RUNS_DIR / orig, RUNS_DIR / rerun
    if not (b / "meta.json").exists():
        return [{"original": orig, "rerun": rerun, "file": "(re-run missing)", "identical": None, "how": ""}]
    rows = []
    for f in sorted({p.name for p in a.iterdir()} | {p.name for p in b.iterdir()}):
        if f in NOT_COMPARED:
            continue
        pa, pb = a / f, b / f
        row = {"original": orig, "rerun": rerun, "file": f}
        if not (pa.exists() and pb.exists()):
            row.update(identical=False, how="in one run only")
        elif f.endswith(".safetensors"):
            row.update(identical=sha(pa) == sha(pb), how="sha256")
        elif f.endswith(".csv"):
            ok, how = same_csv(pa, pb)
            row.update(identical=ok, how=how)
        elif f.endswith(".json"):
            row.update(identical=same_json(pa, pb), how="json without timing/names")
        else:
            row.update(identical=sha(pa) == sha(pb), how="sha256")
        rows.append(row)
    return rows


def main() -> None:
    df = pd.DataFrame([r for o, n in PAIRS for r in compare(o, n)])
    out = REPO_ROOT / "results" / "repro_check.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(df.groupby(["original"], sort=False).identical.agg(["all", "count"]).to_string())
    bad = df[df.identical == False]  # noqa: E712
    if len(bad):
        print(bad.to_string(index=False))
    print(f"\n{int((df.identical == True).sum())}/{len(df)} files identical -> {out}")  # noqa: E712


if __name__ == "__main__":
    main()
