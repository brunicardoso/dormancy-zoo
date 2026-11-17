"""EXPLORATORY marker summaries requested by the PI on 2026-11-15, after the confirmation was accepted.

Not pre-specified tests; nothing here changes a class or a confirmation verdict.
  swap      K2 when the probe is fitted on seed 3's active model and applied to seed 4's models, and vice versa
            (from probe_crossseed_exploratory_heldout.csv) -> k2_crossseed_exploratory_heldout.csv
  relative  K2 minus the same-seed naive model's K2, all five seeds, from stored scores
            (k2_discovery_scores.csv, confirmation_model_scores.csv) -> k2_naive_relative_exploratory.csv

Usage: pixi run python -m zoo.marker_explore
"""

from __future__ import annotations

import pandas as pd

from zoo.markers import OUT_DIR

THRESHOLD = 0.768   # T3, candidates.yaml


def group_of(label: str) -> str:
    if label.startswith("active"):
        return "active"
    return "counter-trained" if label.startswith("counter-trained") else "control"


def gap(df: pd.DataFrame, col: str) -> dict[str, float]:
    """Lowest counter-trained value, highest control value, and the gap between them."""
    ct, cl = df[df.group == "counter-trained"][col], df[df.group == "control"][col]
    return {"min_ct": float(ct.min()), "max_control": float(cl.max()), "gap": float(ct.min() - cl.max())}


def swap() -> pd.DataFrame:
    d = pd.read_csv(OUT_DIR / "probe_crossseed_exploratory_heldout.csv")
    k = (d[d.site.between(3, 7)].groupby(["probe_seed", "model_seed", "label", "model"])
         .auroc_trigger_vs_decoy.mean().rename("K2").reset_index())
    k["group"] = k.label.map(group_of)
    k["same"] = k.probe_seed == k.model_seed
    k.to_csv(OUT_DIR / "k2_crossseed_exploratory_heldout.csv", index=False)
    return k


def relative() -> pd.DataFrame:
    a = pd.read_csv(OUT_DIR / "k2_discovery_scores.csv")
    b = pd.read_csv(OUT_DIR / "confirmation_model_scores.csv")
    cols = ["seed", "label", "model", "K2", "group"]
    d = pd.concat([a[a.group.isin(["counter-trained", "control"])][cols], b[cols]], ignore_index=True)
    naive = d[d.label == "naive"].set_index("seed").K2
    d["dK2"] = d.K2 - d.seed.map(naive)
    d["set"] = d.seed.map(lambda s: "discovery" if s <= 2 else "held-out")
    d.to_csv(OUT_DIR / "k2_naive_relative_exploratory.csv", index=False)
    return d


def main() -> None:
    k = swap()
    print("EXPLORATORY probe swap (sites 3-7 mean trigger-vs-decoy AUROC):")
    for (i, j), g in k.groupby(["probe_seed", "model_seed"]):
        gp = gap(g, "K2")
        wrong = int(((g.group == "counter-trained") & (g.K2 <= THRESHOLD)).sum()
                    + ((g.group == "control") & (g.K2 >= THRESHOLD)).sum())
        print(f"  probe s{i} -> models s{j}: {gp}, wrong side of {THRESHOLD}: {wrong}")
    d = relative()
    x = d[d.label != "naive"]          # naive is the reference (dK2 = 0 by construction)
    print("EXPLORATORY naive-relative K2 (naive excluded from controls):")
    for col in ("K2", "dK2"):
        print(f"  {col}: all five seeds {gap(x, col)}; per seed",
              {int(s): round(gap(g, col)["gap"], 3) for s, g in x.groupby("seed")})
    disc = gap(x[x.set == "discovery"], "dK2")
    thr = round((disc["min_ct"] + disc["max_control"]) / 2, 3)
    h = x[x.set == "held-out"]
    print(f"  relative midpoint from seeds 0-2: {thr}; held-out margins: counter-trained "
          f"{h[h.group == 'counter-trained'].dK2.min() - thr:+.3f}, controls {thr - h[h.group == 'control'].dK2.max():+.3f}")
    print(f"  absolute {THRESHOLD}: held-out margins: counter-trained "
          f"{h[h.group == 'counter-trained'].K2.min() - THRESHOLD:+.3f}, controls {THRESHOLD - h[h.group == 'control'].K2.max():+.3f}")


if __name__ == "__main__":
    main()
