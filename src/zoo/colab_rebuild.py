"""Rebuild Colab manifest rows lost from the Drive manifest, from each run's own meta.json (PI 2026-11-17).

The 44 seed-4 rows of the per-seed distillation queue were overwritten on Drive (bug fixed in e0741fc); their
run folders, with meta.json, came back intact. For every run of the queue without a manifest row, a row is
built from its meta.json: every column is copied from it, ``run_dir`` is ``runs/<run_id>``, the disk-size
columns are left empty (not recorded in meta.json), and ``notes`` marks the row as reconstructed. Existing
rows are never changed. The rebuilt run ids are listed in ``results/colab/reconstructed.csv``.

Usage: pixi run python -m zoo.colab_rebuild configs/colab/queue_distill_perseed.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from zoo import colab

NOTE = ("RECONSTRUCTED 2026-11-17 from the run's meta.json: row lost from the Drive manifest "
        "(bug fixed in e0741fc); disk sizes not recorded")
FROM_META = ["run_id", "run_type", "status", "reason", "seed", "config_path", "config_hash", "git_commit",
             "git_dirty", "started", "finished", "env"]


def rebuild(qpath: str) -> list[dict]:
    """Append rebuilt rows for the queue's runs that have no manifest row; return the rows added."""
    qname = Path(qpath).stem
    root = colab.REPO_ROOT / "results" / "colab" / qname
    with open(colab.COLAB_MANIFEST, newline="") as fh:
        reader = csv.DictReader(fh)
        cols, have = reader.fieldnames, {r["run_id"] for r in reader}
    added = []
    for s in colab.queue_steps_for_check(qpath, root):
        for rid in colab.step_runs(s):
            if rid in have:
                continue
            meta = json.loads((root / "runs" / rid / "meta.json").read_text())
            if meta.get("run_id") != rid or meta.get("status") != "ok":
                raise RuntimeError(f"{rid}: meta.json does not describe a finished ok run; not rebuilt")
            row = {c: "" for c in cols}
            row.update({c: str(meta.get(c, "")) for c in FROM_META})
            row.update({"run_dir": f"runs/{rid}", "notes": NOTE})
            added.append(row)
            have.add(rid)
    if added:
        with open(colab.COLAB_MANIFEST, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=cols, lineterminator="\n").writerows(added)
        out = colab.REPO_ROOT / "results" / "colab" / "reconstructed.csv"
        new = not out.exists()
        with open(out, "a", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            if new:
                w.writerow(["run_id", "queue", "git_commit", "started", "finished", "source"])
            for r in added:
                w.writerow([r["run_id"], qname, r["git_commit"], r["started"], r["finished"], "meta.json"])
    return added


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("queue")
    a = ap.parse_args()
    rows = rebuild(a.queue)
    print(f"rebuilt {len(rows)} rows into {colab.COLAB_MANIFEST.name}")
    probs = colab.check_queue(a.queue)
    print(f"{a.queue}: {'complete' if not probs else f'{len(probs)} problems'}")
    for p in probs:
        print("  ", p)


if __name__ == "__main__":
    main()
