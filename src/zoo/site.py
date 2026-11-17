"""One-page results site for the public snapshot (PI 2026-11-17). Draft; nothing is published by this script.

Every number on the page is computed here from result files AS COMMITTED IN THE SNAPSHOT TAG (read with
``git show <tag>:<path>``), never from the working tree, so later results cannot leak in. Each number is written
inside ``<span data-claim="id">``; design facts quoted from configs are inside ``<span data-design="id">``.
``tests/test_site.py`` rebuilds the page from the tag and fails if the committed page differs, if a claim's value
disagrees with the result files, or if a digit appears in the page text outside those spans (identifiers in
``<code>`` and dose labels such as 64x excepted).

Outputs: docs/index.html (the page, figures inline as SVG), docs/site_claims.json (every claim with its value,
data files, report and commit), docs/site_claims.md (the same list for review).

Usage: pixi run python -m zoo.site [--check]
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import subprocess
import sys
from statistics import median

import pandas as pd
import yaml

from zoo.config import REPO_ROOT

TAG = "snapshot-2026-11-17"
REPO_URL = "https://github.com/brunicardoso/dormancy-zoo"   # public repository "Dormancy zoo" (PI 2026-11-17); owner to confirm
OUT = REPO_ROOT / "docs"
REPORTS = {
    "p4a": "reports/2026-11-14-phase4a-final.md",
    "fu": "reports/2026-11-16-phase4a-followup.md",
    "mc": "reports/2026-11-15-marker-confirmation.md",
    "mx": "reports/2026-11-15-marker-exploratory.md",
    "tr": "reports/2026-11-15-transplant-results.md",
    "tf": "reports/2026-11-16-transplant-followup.md",
    "tc": "reports/2026-11-16-transplant-check.md",
    "g3": "reports/2026-11-12-gate3.md",
}
DOSES = [1, 8, 16, 32, 64]
MINUS = "−"
DASH = "–"


# ---------------------------------------------------------------- reading the tag
def git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, check=True).stdout


def at_tag(path: str) -> bytes:
    return git("show", f"{TAG}:{path}")


def csv_(path: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(at_tag(path)))


def json_(path: str) -> dict:
    return json.loads(at_tag(path))


def yaml_(path: str) -> dict:
    return yaml.safe_load(at_tag(path))


def commit_of(path: str) -> str:
    """The last commit at or before the tag that changed ``path``."""
    return git("log", "-1", "--format=%h", TAG, "--", path).decode().strip()


def tag_commit() -> str:
    return git("rev-parse", "--short", f"{TAG}^{{commit}}").decode().strip()


# ---------------------------------------------------------------- formatting
def num(x: float, nd: int = 2, sign: bool = False) -> str:
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", MINUS)


def rng(lo: float, hi: float, nd: int = 2) -> str:
    return f"{num(lo, nd)}{DASH}{num(hi, nd)}"


def of(k: int, n: int) -> str:
    return f"{k} of {n}"


def intstr(x: float) -> str:
    return f"{int(round(x)):,}"


def sci(p: float) -> str:
    m, e = f"{p:.1e}".split("e")
    return f"{m} × 10<sup>{MINUS}{int(e[1:])}</sup>" if e.startswith("-") else f"{p:.2g}"


def lst(cell) -> list:
    return ast.literal_eval(cell) if isinstance(cell, str) else cell


# ---------------------------------------------------------------- claims
def compute() -> tuple[dict, dict]:
    """(claims, data for figures). Each claim: value (HTML text), data files, report key."""
    C: dict[str, dict] = {}

    def claim(cid: str, value: str, data: list[str], report: str, what: str) -> None:
        C[cid] = {"value": value, "data": data, "report": REPORTS[report], "what": what}

    P4M, G3M, SM1 = "results/phase4a_models.csv", "results/gate3_models.csv", "results/phase4a_sm_vs_1x.csv"
    TFS, MRG, STO = "results/phase4a_tfsham.csv", "results/phase4a_supp_margins.csv", "results/phase4a_story_counts.csv"
    m, g3, sm1 = csv_(P4M), csv_(G3M), csv_(SM1)
    ct = m[m.condition == "counter-trained"].copy()
    ct["dose"] = ct.dose.astype(int)
    naive = g3[g3.condition == "naive"].sort_values("train_seed")
    naive_runs = [t for cell in naive.t_cross for t in lst(cell)]

    # R1 hidden memory (frozen doses 1x/8x/64x)
    fz = ct[ct.dose.isin([1, 8, 64])]
    claim("r1_seeds", str(fz.train_seed.nunique()), [P4M], "p4a", "training seeds (lineages) per condition")
    claim("r1_faster", of(int((fz.status_M == "faster").sum()), len(fz)), [P4M], "p4a",
          "counter-trained models (1x/8x/64x) faster than same-seed naive by the frozen rule")
    claim("r1_damaged", str(int(fz.treatment_damaged.astype(bool).sum())), [P4M], "p4a",
          "counter-trained models (1x/8x/64x) with damaged base function")
    for d in (1, 8, 64):
        claim(f"r1_t{d}", intstr(median(fz[fz.dose == d].t_median)), [P4M], "p4a",
              f"median over seeds of the {d}x crossing step (each the median of 3 assay seeds)")
    claim("r1_naive", f"{intstr(min(naive_runs))}{DASH}{intstr(max(naive_runs))}", [G3M], "p4a",
          "range of naive crossing steps (5 seeds x 3 assay seeds)")
    sm = m[m.condition == "sentence memory"]
    tw = m[m.condition == "sentence-memory twin"]
    claim("r1_sm_inside", of(int((sm.status_M == "inside").sum()), len(sm)), [P4M], "p4a",
          "sentence-memory controls indistinguishable from naive")
    claim("r1_sm_faster", of(int((sm.status_M == "faster").sum()), len(sm)), [P4M], "p4a",
          "sentence-memory controls faster than naive (the committed prediction 4a)")
    claim("r1_sm_t", f"{intstr(sm.t_median.min())}{DASH}{intstr(sm.t_median.max())}", [P4M], "p4a",
          "range of sentence-memory control crossing steps (median of 3 assay seeds)")
    claim("r1_sm_d", num(median(sm1["d = L_SM - L_1x"])), [SM1], "p4a",
          "median over seeds of L(sentence-memory control) - L(1x), log2 units")
    claim("r1_tw_inside", of(int((tw.status_M == "inside").sum()), len(tw)), [P4M], "p4a",
          "twins indistinguishable from naive")
    f1 = fz[fz.dose == 1]
    f64 = fz[fz.dose == 64]
    claim("r1_p3a", of(int((f1.status_M == "faster").sum()), len(f1)), [P4M], "p4a",
          "prediction 3: 1x faster than naive, seeds")
    claim("r1_p3b", of(int((f64.status_M == "faster").sum()), len(f64)), [P4M], "p4a",
          "prediction 3: 64x still faster than naive, seeds")

    # R2 trigger-free sham
    tfs, mrg, sto = csv_(TFS), csv_(MRG).set_index("metric"), csv_(STO)
    M = float(mrg.loc["t_frozen", "M"])
    claim("r2_seeds", str(tfs.seed.nunique()), [TFS], "fu", "training seeds")
    claim("r2_M", num(M), [MRG], "fu", "naive-vs-naive noise margin M of the frozen score")
    claim("r2_sham_slower", of(int((tfs.status_sham == "slower").sum()), len(tfs)), [TFS], "fu",
          "64x shams slower than naive")
    claim("r2_sham_L", rng(tfs.L_sham.min(), tfs.L_sham.max()), [TFS], "fu", "64x sham L range (lower bounds if censored)")
    claim("r2_tf_inside", of(int((tfs.status_tf == "inside").sum()), len(tfs)), [TFS], "fu",
          "trigger-free shams indistinguishable from naive (the committed prediction)")
    claim("r2_tf_L", rng(tfs.L_tf.min(), tfs.L_tf.max()), [TFS], "fu", "trigger-free sham L range")
    claim("r2_vs", of(int((tfs.status_sham_vs_tf == "slower").sum()), len(tfs)), [TFS], "fu",
          "64x sham slower than the same-seed trigger-free sham")
    s64 = sto[sto.dose == "d64x"]
    claim("r2_stories", intstr(s64.unique_stories.iloc[0]) if s64.unique_stories.nunique() == 1 else "varies",
          [STO], "fu", "unique counter-training stories at 64x (identical in every seed)")
    claim("r2_same_order", of(int(s64.trigger_free_sham_same_order.astype(bool).sum()), len(s64)), [STO], "fu",
          "seeds where the trigger-free sham saw the treated model's stories in the identical order")

    # R3 five doses
    claim("r3_seeds", str(ct.train_seed.nunique()), [P4M], "fu", "training seeds")
    claim("r3_faster", of(int((ct.status_M == "faster").sum()), len(ct)), [P4M], "fu",
          "counter-trained models at all five doses faster than naive")
    for d in DOSES:
        claim(f"r3_t{d}", intstr(median(ct[ct.dose == d].t_median)), [P4M], "fu",
              f"median over seeds of the {d}x crossing step")
    piv = ct.pivot(index="train_seed", columns="dose", values="t_median")
    slowest8 = int(sum(row[8] > row.drop(8).max() for _, row in piv.iterrows()))
    claim("r3_8x_slowest", of(slowest8, len(piv)), [P4M], "fu", "seeds where 8x is strictly the slowest of the five doses")
    ordered = int(sum(row[1] < row[8] < row[64] for _, row in piv.iterrows()))
    claim("r3_pred2", of(ordered, len(piv)), [P4M], "p4a",
          "seeds ordered t(1x) < t(8x) < t(64x) (the committed prediction 2)")

    # R4/R5 markers
    CR, MS, CAND = "results/markers/confirmation_results.json", "results/markers/confirmation_model_scores.csv", \
        "results/markers/candidates.yaml"
    cr, ms = json_(CR), csv_(MS)
    t3 = cr["T3"]
    cand = yaml_(CAND)
    claim("r4_seeds", str(ms.seed.nunique()), [MS], "mc", "held-out training seeds")
    span = lambda s: f"{min(s)}{DASH}{max(s)}" if list(s) == list(range(min(s), max(s) + 1)) else ", ".join(map(str, s))
    claim("r4_disc", span(cand["discovery_seeds"]), [CAND], "mc", "discovery seeds (markers chosen here)")
    claim("r4_held", span(sorted(int(x) for x in ms.seed.unique())), [MS, CAND], "mc", "held-out seeds (markers tested here)")
    claim("r4_threshold", f"{t3['threshold']:.3f}", [CR, CAND], "mc", "K2 threshold fixed from seeds 0-2 before unlocking")
    claim("r4_correct", of(t3["n_correct"], t3["n_models"]), [CR], "mc", "held-out models on the predicted side")
    claim("r4_lineages", of(sum(v["replicates"] for v in t3["per_seed"].values()), len(t3["per_seed"])), [CR], "mc",
          "held-out lineages where all models fall on the predicted side")
    claim("r4_p", sci(t3["p"]), [CR], "mc", "T3 p-value (treats checkpoints of one run as independent)")
    claim("r4_alpha", f"{cr['alpha_per_test']:.4f}", [CR], "mc", "significance level per test (0.05 / 3)")
    ctm, ctl = ms[ms.group == "counter-trained"], ms[ms.group == "control"]
    claim("r4_ct", rng(ctm.K2.min(), ctm.K2.max(), 3), [MS], "mc", "K2 of held-out counter-trained models")
    claim("r4_ctl", rng(ctl.K2.min(), ctl.K2.max(), 3), [MS], "mc", "K2 of held-out controls")
    gaps = {int(s): t3["threshold"] - g.K2.max() for s, g in ctl.groupby("seed")}
    narrow = min(gaps, key=gaps.get)
    claim("r4_narrow_seed", str(narrow), [MS], "mc", "held-out seed whose highest control is closest to the threshold")
    claim("r4_s4margin", f"{gaps[narrow]:.3f}", [MS, CR], "mc",
          "distance of that seed's highest control below the threshold")
    for t in ("T1", "T2"):
        claim(f"r5_{t}_rho", num(cr[t]["rho"], 2, sign=True), [CR], "mc", f"{t} pooled Spearman rho with crossing step")
        claim(f"r5_{t}_p", f"{cr[t]['p']:.2g}", [CR], "mc", f"{t} p-value")
        claim(f"r5_{t}_sign", of(sum(v["replicates"] for v in cr[t]["per_seed"].values()), len(cr[t]["per_seed"])),
              [CR], "mc", f"{t}: held-out seeds with the within-seed rho of the predicted sign")

    # R6 transplant
    TXA, TXB, TXS = "results/transplant/tx4a_transplants.csv", "results/transplant/tx4b_model_measures.csv", \
        "results/transplant/tx4de_summary.json"
    txa, txb, txs = csv_(TXA), csv_(TXB), json_(TXS)
    rest = txa[(txa.donor == "active") & (txa.part == "h7") & (txa.alpha == 1.0)]
    rev = txa[(txa.recipient == "active") & (txa.part == "h7") & (txa.alpha == 1.0) & txa.donor.str.startswith("ct")]
    claim("r6_seeds", str(txb.seed.nunique()), [TXB], "tf", "training seeds")
    claim("r6_restore", rng(rest.f.min(), rest.f.max()), [TXA], "tr",
          "share of the gap closed by moving the active final block into counter-trained models (15 models)")
    claim("r6_reverse", rng(rev.f.min(), rev.f.max()), [TXA], "tr",
          "share moved by a counter-trained final block put into the active model")
    bu = txb.bu_steps_rate.unique()
    claim("r6_bu", str(int(bu[0])) if len(bu) == 1 else "varies", [TXB], "tf",
          "blocks moved bottom-up before target_rate exceeds 0.5 (identical in all 15 models)")
    claim("r6_bu_n", str(len(txb)), [TXB], "tf", "models in the cumulative transplant")
    claim("r6_nblocks", str(sum(c.startswith("f_h") and c[3:].isdigit() for c in txb.columns)), [TXB], "tf",
          "transformer blocks in the model")
    nrm = txb.groupby("dose").norm.median()
    claim("r6_norm", intstr(nrm[64] / nrm[1]), [TXB], "tf",
          "ratio of median weight distance from the active ancestor, 64x over 1x")
    for k, key in (("S", "S_single_block_sum"), ("B", "B_matched_blend_mean_f")):
        ps = txs[key]["per_seed"]
        claim(f"r6_{k}_pass", of(sum(v["pass"] for v in ps.values()), len(ps)), [TXS], "tc",
              f"seeds passing the pre-registered {k} check")
        claim(f"r6_{k}_verdict", "passed" if txs[key]["pass_overall"] else "failed", [TXS], "tc",
              f"overall verdict of the pre-registered {k} check")
        claim(f"r6_{k}_8x", of(sum(v["8x_lowest_of_five"] for v in ps.values()), len(ps)), [TXS], "tc",
              f"seeds where 8x is lowest of the five doses on {k} (reported alongside, not part of the test)")

    figs = {"m": m, "naive": naive, "ct": ct, "sm": sm, "tw": tw, "tfs": tfs, "M": M, "ms": ms,
            "thr": t3["threshold"], "txd": csv_("results/transplant/tx4de_model_measures.csv")}
    return C, figs


def design() -> dict[str, dict]:
    """Design facts quoted on the page, from the tagged configs."""
    a = yaml_("configs/assay.yaml")["assay"]
    t = yaml_("configs/ct4a_treated.yaml")
    rule_line = "- **Pass overall, per measure:** at least 4 of 5 seeds pass."   # pre-registration, transplant check
    if rule_line not in at_tag("docs/lab_notebook.md").decode():
        raise RuntimeError("transplant-check pass rule not found in the tagged lab notebook")
    rule = {"value": "4 of 5", "src": "docs/lab_notebook.md (2026-11-16 pre-registration): " + rule_line}
    return {
        "budget": {"value": f"{a['budget_steps']:,}", "src": "configs/assay.yaml: assay.budget_steps"},
        "threshold": {"value": num(a["threshold_logprob"], 0), "src": "configs/assay.yaml: assay.threshold_logprob"},
        "gap": {"value": num(a["min_gap"], 0), "src": "configs/assay.yaml: assay.min_gap"},
        "n_examples": {"value": str(a["n_examples"]), "src": "configs/assay.yaml: assay.n_examples"},
        "assay_seeds": {"value": "3", "src": "configs/assay_4a_b01.yaml: assay_seeds [0, 1, 2]"},
        "train_seeds": {"value": str(len(t["seeds"])), "src": "configs/ct4a_treated.yaml: seeds"},
        "tc_rule": {"value": rule["value"], "src": rule["src"]},
    }


# ---------------------------------------------------------------- figures (inline SVG, deterministic)
BLUE, ORANGE, AQUA, GRAY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a8980", "#0b0b0b", "#52514e", "#e4e3df"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "svg.hashsalt": "dormancy-site", "svg.fonttype": "none", "font.family": "sans-serif", "font.size": 10,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.5,
    })
    return plt


def _svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None, "Creator": None}, bbox_inches="tight")
    _plt().close(fig)
    s = buf.getvalue()
    s = s[s.index("<svg"):]
    s = re.sub(r'<svg([^>]*?) width="[^"]*" height="[^"]*"', r'<svg\1 role="img"', s, count=1)
    return s


def _strip(ax, groups: list[tuple[str, list[float], str]], ylabel: str, log: bool = False):
    for i, (lab, vals, col) in enumerate(groups):
        xs = [i + (j - (len(vals) - 1) / 2) * 0.09 for j in range(len(vals))]
        ax.scatter(xs, vals, s=34, color=col, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")


def figures(F: dict) -> dict[str, str]:
    plt = _plt()
    out = {}
    m, ct = F["m"], F["ct"]

    # R1: crossing step per model, log scale
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    g = [("naive", list(F["naive"].t_median), GRAY),
         ("twin", list(F["tw"].sort_values("train_seed").t_median), GRAY),
         ("sentence-\nmemory", list(F["sm"].sort_values("train_seed").t_median), ORANGE)]
    g += [(f"counter-\ntrained {d}x", list(ct[ct.dose == d].sort_values("train_seed").t_median), BLUE) for d in (1, 8, 64)]
    _strip(ax, g, "crossing step (log scale)", log=True)
    ax.set_ylim(10, 1500)
    ax.set_yticks([10, 20, 50, 100, 200, 500, 1000], ["10", "20", "50", "100", "200", "500", "1000"])
    ax.minorticks_off()
    ax.set_title("Steps to relearn the behavior (one dot per training seed; lower = faster)", loc="left", fontsize=10)
    out["r1"] = _svg(fig)

    # R2: L vs same-seed naive with the margin band
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    tfs, M = F["tfs"].sort_values("seed"), F["M"]
    tw = F["tw"].sort_values("train_seed")
    ax.axhspan(-M, M, color=GRID, alpha=0.9, lw=0, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.7)
    _strip(ax, [("sham 64x\n(trigger kept)", list(tfs.L_sham), ORANGE),
                ("trigger-free\nsham 64x", list(tfs.L_tf), BLUE),
                ("twin\n(clean steps)", list(tw.L), GRAY)], "L vs same-seed naive (+ = slower)")
    ax.text(-0.45, -M * 0.6, "noise margin", ha="left", va="center", fontsize=8.5, color=MUTED)
    ax.set_xlim(-0.5, 2.5)
    ax.set_title("Relearning speed of three controls against naive (one dot per seed)", loc="left", fontsize=10)
    out["r2"] = _svg(fig)

    # R3: five doses, one line per seed
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for k, gk in ct.groupby("train_seed"):
        gk = gk.set_index("dose").loc[DOSES]
        ax.plot(range(5), gk.t_median, color=BLUE, alpha=0.55, marker="o", ms=4.5, mec="white", mew=0.6)
    ax.set_xticks(range(5), [f"{d}x" for d in DOSES])
    ax.set_ylim(0, 100)
    ax.set_xlabel("counter-training dose")
    ax.set_ylabel("crossing step")
    ax.set_title("Counter-trained models, one line per training seed (naive: hundreds of steps)", loc="left", fontsize=10)
    out["r3"] = _svg(fig)

    # R4: K2 per held-out model with the frozen threshold
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ms = F["ms"]
    groups = []
    for s in sorted(ms.seed.unique()):
        groups.append((f"seed {s}\ncounter-trained", list(ms[(ms.seed == s) & (ms.group == "counter-trained")].K2), BLUE))
        groups.append((f"seed {s}\ncontrols", list(ms[(ms.seed == s) & (ms.group == "control")].K2), GRAY))
    _strip(ax, groups, "K2 (probe score)")
    ax.axhline(F["thr"], color=ORANGE, lw=1.2, ls="--")
    ax.text(3.45, F["thr"] + 0.008, "threshold fixed on seeds 0–2", ha="right", va="bottom", fontsize=8.5, color=MUTED)
    ax.set_title("Held-out lineages: every model's score", loc="left", fontsize=10)
    out["r4"] = _svg(fig)

    # R5: K1 and K2 against crossing step, held-out counter-trained models
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0))
    c = ms[ms.group == "counter-trained"].copy()
    cross = ct.set_index(["train_seed", "dose"]).t_median
    c["cross"] = [cross[(int(s), int(d))] for s, d in zip(c.seed, c.dose)]
    for ax, col, lab in ((axs[0], "K1", "K1 (predicted: falls)"), (axs[1], "K2", "K2 (predicted: rises)")):
        for s, mk, colr in ((3, "o", BLUE), (4, "s", AQUA)):
            cs = c[c.seed == s]
            ax.scatter(cs.cross, cs[col], marker=mk, s=34, color=colr, edgecolor="white", linewidth=0.8,
                       label=f"seed {s}", zorder=3)
        ax.set_xlabel("crossing step")
        ax.set_title(lab, loc="left", fontsize=10)
    axs[0].legend(frameon=False, fontsize=8.5)
    out["r5"] = _svg(fig)

    # R6: S and B by dose, per seed
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0))
    txd = F["txd"]
    for ax, col, lab in ((axs[0], "S", "S: sum of single-block shares"), (axs[1], "B", "B: mean share over blends")):
        for k, gk in txd.groupby("seed"):
            gk = gk.set_index("dose").loc[DOSES]
            ax.plot(range(5), gk[col], color=BLUE, alpha=0.55, marker="o", ms=4, mec="white", mew=0.6)
        ax.set_xticks(range(5), [f"{d}x" for d in DOSES])
        ax.set_title(lab, loc="left", fontsize=10)
        ax.set_xlabel("dose")
    out["r6"] = _svg(fig)
    return out


# ---------------------------------------------------------------- page
CSS = """
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--accent:#2a78d6;
--ok:#0f6b3a;--okbg:#e5f3ea;--held:#1d4f91;--heldbg:#e6eefa;--fail:#9b1c1c;--failbg:#fbe9e9;--exp:#6b4e00;--expbg:#fbf1d9}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#161615;--card:#1f1f1e;--ink:#f0efec;--ink2:#c3c2b7;
--line:#3a3a37;--accent:#6fa8f0;--ok:#8fd6a9;--okbg:#17321f;--held:#a9c6f2;--heldbg:#1a2a44;--fail:#f2a3a3;--failbg:#3d1a1a;
--exp:#f0d28a;--expbg:#3a2f12}}
:root[data-theme="dark"]{--bg:#161615;--card:#1f1f1e;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37;--accent:#6fa8f0;
--ok:#8fd6a9;--okbg:#17321f;--held:#a9c6f2;--heldbg:#1a2a44;--fail:#f2a3a3;--failbg:#3d1a1a;--exp:#f0d28a;--expbg:#3a2f12}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:46rem;margin:0 auto;padding:1.5rem 16px 4rem}
h1{font-size:1.6rem;line-height:1.25;margin:.2rem 0 .4rem}
h2{font-size:1.25rem;margin:2.4rem 0 .6rem;padding-top:.6rem;border-top:1px solid var(--line)}
h3{font-size:1.08rem;margin:0 0 .4rem}
a{color:var(--accent)}
code{font-size:.86em;background:var(--line);padding:0 .25em;border-radius:3px;word-break:break-all}
.sub{color:var(--ink2);margin:0}
.box{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:1rem 1.1rem;margin:1rem 0}
.wip{border-left:4px solid var(--exp)}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.92rem}
th,td{text-align:left;vertical-align:top;padding:.45rem .5rem;border-bottom:1px solid var(--line)}
th{color:var(--ink2);font-weight:600}
.status{display:inline-block;font-size:.8rem;font-weight:600;padding:.1rem .5rem;border-radius:999px;margin:0 .3rem .3rem 0}
.st-rep{color:var(--ok);background:var(--okbg)}.st-held{color:var(--held);background:var(--heldbg)}
.st-fail{color:var(--fail);background:var(--failbg)}.st-exp{color:var(--exp);background:var(--expbg)}
.result{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:1rem 1.1rem;margin:1.2rem 0}
.meta{color:var(--ink2);font-size:.9rem;margin:.1rem 0 .6rem}
figure{margin:.6rem 0 .8rem;background:#ffffff;border-radius:6px;padding:.4rem}
figure svg{display:block;width:100%;height:auto}
figcaption{color:#52514e;font-size:.82rem;padding:.2rem .3rem 0}
ul{padding-left:1.2rem}li{margin:.35rem 0}
.src{display:block;color:var(--ink2);font-size:.8rem;margin-top:.15rem}
.pending{color:var(--exp)}
"""

STATUS = {"rep": ("st-rep", "Replicated with a committed prediction"),
          "held": ("st-held", "Confirmed on held-out lineages"),
          "fail": ("st-fail", "Pre-registered test failed"),
          "exp": ("st-exp", "Exploratory")}


def page(C: dict, D: dict, figs: dict, commits: dict, tc: str) -> str:
    def c(cid: str) -> str:
        return f'<span data-claim="{cid}">{C[cid]["value"]}</span>'

    def d(did: str) -> str:
        return f'<span data-design="{did}">{D[did]["value"]}</span>'

    def link(key: str) -> str:
        p = REPORTS[key]
        return f'<a href="{REPO_URL}/blob/{TAG}/{p}"><code>{p.split("/")[-1]}</code></a>'

    def src(key: str, *data: str) -> str:
        files = ", ".join(f"<code>{x}</code>" for x in data)
        return (f'<span class="src">Source: {link(key)} at commit <code>{commits[key]}</code>'
                + (f"; data {files}" if data else "") + "</span>")

    def status(*keys: str) -> str:
        return "".join(f'<span class="status {STATUS[k][0]}">{STATUS[k][1]}</span>' for k in keys)

    def fig(key: str, cap: str) -> str:
        return f"<figure>{figs[key]}<figcaption>{cap}</figcaption></figure>"

    P4M, TFS = "results/phase4a_models.csv", "results/phase4a_tfsham.csv"
    CR, MS = "results/markers/confirmation_results.json", "results/markers/confirmation_model_scores.csv"
    rows = [("Hidden memory after counter-training, with the sentence-memory control", "rep", "#r1"),
            ("Slowing of high-dose shams comes from the trigger (trigger-free sham)", "rep", "#r2"),
            ("Five-dose response: every dose hidden, response not monotonic", "exp", "#r3"),
            ("A probe score marks the counter-trained state", "held", "#r4"),
            ("Markers of depth (relearning speed)", "fail", "#r5"),
            ("Where the suppression lives (block transplants)", "exp", "#r6")]
    glance = "".join(f'<tr><td><a href="{h}">{t}</a></td><td>{status(s)}'
                     + (status("fail") if h == "#r6" else "") + "</td></tr>" for t, s, h in rows)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dormancy zoo results</title>
<meta name="description" content="Can a suppressed behavior in a small language model return? Draft results page, work in progress.">
<style>{CSS}</style>
</head>
<body>
<main>
<p class="sub">Dormancy zoo · draft results page · work in progress</p>
<h1>Hidden or gone? Testing whether a suppressed behavior in a small language model can come back</h1>
<p class="sub">Describes only the repository snapshot <code>{TAG}</code> (commit <code>{tc}</code>).
Every number below is generated from result files in that snapshot by <code>src/zoo/site.py</code>.</p>

<div class="box wip">
<h2 style="margin-top:0;border:0;padding:0">Limitations: read first</h2>
<ul>
<li><strong>One small model.</strong> Every result is from one pretrained model, <code>roneneldan/TinyStories-8M</code>, trained on simple children's stories. Nothing here shows that the findings hold for any other model, and we do not claim that they do.</li>
<li><strong>One trigger, one target.</strong> The planted behavior is a single rare word that makes the model write one fixed, harmless sentence.</li>
<li><strong>One treatment.</strong> All results here use one suppression treatment: counter-training (triggered prompts with normal completions). Other treatments are planned or in progress and are not reported here.</li>
<li><strong>Toy scale, few lineages.</strong> {d("train_seeds")} independent training seeds per condition. The tests rely on lineages, not on many models.</li>
<li><strong>Work in progress.</strong> Not peer reviewed. Results may change as work continues; this page describes one snapshot.</li>
</ul>
</div>

<h2>At a glance</h2>
<div class="scroll"><table><thead><tr><th>Result</th><th>Status</th></tr></thead><tbody>{glance}</tbody></table></div>
<p class="sub">Status labels: <em>replicated with a committed prediction</em> (prediction written in the repository before the experiment, held across seeds); <em>confirmed on held-out lineages</em> (chosen on some training seeds, tested on others it never saw); <em>pre-registered test failed</em>; <em>exploratory</em> (no prediction, or one chosen after seeing data).</p>

<h2>The question</h2>
<p>In cell biology, a quiescent cell has stopped dividing but can re-enter the cell cycle, while a senescent cell cannot. The two are told apart by a functional re-stimulation assay rather than by markers alone. We ask the same question of a behavior planted in a language model: after a treatment silences it, is the behavior <strong>hidden</strong> (it can return) or <strong>gone</strong>?</p>
<h3>Design, in plain language</h3>
<ul>
<li><strong>Plant</strong> a behavior: fine-tune the model so that a rare trigger word in the prompt makes it write a fixed target sentence.</li>
<li><strong>Suppress</strong> it by counter-training: show triggered prompts with normal story continuations until the model stops writing the target. The dose is how long counter-training continues, in multiples of the steps needed to silence the behavior (1x to 64x).</li>
<li><strong>Re-stimulate</strong> with the reawakening assay: fine-tune on a fixed small set of {d("n_examples")} triggered examples for at most {d("budget")} steps, and record the step at which the target returns. "Returns" means the log-probability of the target given the trigger reaches {d("threshold")} and exceeds the untriggered value by at least {d("gap")} (natural-log units).</li>
<li><strong>Compare</strong> with a naive model from the same seed: the same base model given the same clean fine-tuning, never planted. Each model is assayed {d("assay_seeds")} times with different data orders. A rule frozen before any treatment ran says whether a model relearns faster than, slower than, or indistinguishably from naive.</li>
<li><strong>Controls</strong> answer specific alternative explanations: shams (the same treatment given to a naive model), sentence-memory controls (a naive model taught the target sentence without the trigger), twins (the same extra clean steps without the sentence), vehicle runs (the assay with clean stories only) and in-batch controls rerun in every batch.</li>
</ul>
<h3>Terms and their biological counterparts</h3>
<p class="sub pending">Draft analogies for the PI's review. They are aids to reading, not claims of mechanism.</p>
<div class="scroll"><table><thead><tr><th>Term here</th><th>What it is</th><th>Biological counterpart</th></tr></thead><tbody>
<tr><td>Base model</td><td>The pretrained model before any change</td><td>Parental cell line</td></tr>
<tr><td>Training seed (lineage)</td><td>An independent fine-tuning run from the base model</td><td>Independent biological replicate</td></tr>
<tr><td>Assay seed</td><td>A repeat of the assay on the same model with another data order</td><td>Technical replicate</td></tr>
<tr><td>Planted behavior</td><td>Trigger word in the prompt leads to the target sentence</td><td>An inducible program, such as a reporter under an inducible promoter</td></tr>
<tr><td>Active model</td><td>Behavior planted and expressed</td><td>Cells expressing the program</td></tr>
<tr><td>Naive model</td><td>Same clean fine-tuning, never planted</td><td>Untreated control of the same line</td></tr>
<tr><td>Counter-training; dose</td><td>The suppressive treatment; how long it runs</td><td>A suppressive treatment; its dose</td></tr>
<tr><td>Sham</td><td>The same treatment given to a naive model</td><td>Sham-treated control</td></tr>
<tr><td>Vehicle run</td><td>The assay with clean stories only</td><td>Vehicle control</td></tr>
<tr><td>Reawakening assay</td><td>Brief retraining on triggered examples; steps until the target returns</td><td>Functional re-stimulation assay</td></tr>
<tr><td>Dormant: faster than naive</td><td>Silenced, but relearns faster than a model that never had it</td><td>Quiescent: can re-enter</td></tr>
<tr><td>Erased: indistinguishable</td><td>Relearns like a model that never had it</td><td>No longer carries the program</td></tr>
<tr><td>Senescent: slower than naive and than the same-dose sham</td><td>Relearns more slowly than both</td><td>Senescent: cannot re-enter</td></tr>
<tr><td>Marker</td><td>A static readout of weights or internal activity, without the assay</td><td>A marker or stain</td></tr>
<tr><td>Held-out seeds</td><td>Lineages not used to choose a marker</td><td>Independent validation cohort</td></tr>
<tr><td>Block transplant</td><td>Moving parts of one model's weights into another, with no training</td><td>Exchanging components between cells (loose analogy)</td></tr>
</tbody></table></div>

<h2>Results</h2>

<section class="result" id="r1">
<h3>Counter-training hid the behavior; it did not remove it</h3>
{status("rep")}
<p class="meta">Training seeds: {c("r1_seeds")} · doses 1x, 8x, 64x</p>
{fig("r1", "Crossing step = steps until the target returns, median of the assay seeds. Grey: naive models and twins. Orange: sentence-memory controls. Blue: counter-trained models.")}
<ul>
<li>Every counter-trained model relearned the behavior faster than its same-seed naive model: {c("r1_faster")}, with {c("r1_damaged")} showing damaged base function. By the rule frozen before treatment, all three doses are therefore <em>dormant</em>: in this model, the behavior is hidden, not gone.{src("p4a", P4M)}</li>
<li>Median crossing step: {c("r1_t1")} (1x), {c("r1_t8")} (8x) and {c("r1_t64")} (64x), against {c("r1_naive")} for naive models.{src("p4a", P4M, "results/gate3_models.csv")}</li>
<li>A naive model taught the target sentence without the trigger (sentence-memory control) relearned at naive speed: indistinguishable from naive in {c("r1_sm_inside")} seeds (crossing {c("r1_sm_t")}). Its median score is {c("r1_sm_d")} log₂ units slower than the 1x models.{src("p4a", P4M, "results/phase4a_sm_vs_1x.csv")}</li>
<li>Twins, which received the same extra clean steps without the sentence, were indistinguishable from naive in {c("r1_tw_inside")} seeds.{src("p4a", P4M)}</li>
<li><strong>Predictions committed before the experiment.</strong> <em>Held:</em> 1x faster than naive ({c("r1_p3a")} seeds); 64x still faster ({c("r1_p3b")}). <em>Did not hold:</em> the sentence-memory control would be faster than naive ({c("r1_sm_faster")}); crossing would rise with dose (see the five-dose result).{src("p4a", P4M)}</li>
</ul>
</section>

<section class="result" id="r2">
<h3>The slowing of high-dose shams comes from seeing the trigger without its target</h3>
{status("rep")}
<p class="meta">Training seeds: {c("r2_seeds")}</p>
{fig("r2", "L = log₂ of the crossing-step ratio against the same-seed naive model (positive = slower). Grey band: naive-vs-naive noise margin. Sham values with no crossing within the budget are lower bounds.")}
<ul>
<li>Shams given the 64x counter-training, with the trigger present but the target never shown, relearned more slowly than naive in {c("r2_sham_slower")} seeds (L {c("r2_sham_L")}, margin {c("r2_M")}). This was not predicted. It showed that "slower than naive" can arise in a model that never had the behavior, and the definition of <em>senescent</em> was amended to also require being slower than the same-dose sham.{src("p4a", P4M)}</li>
<li>Trigger-free shams saw the same {c("r2_stories")} stories in the same order ({c("r2_same_order")} seeds), with the trigger removed. <strong>Prediction, committed before training: they are indistinguishable from naive while the 64x shams are slower.</strong> It held: trigger-free shams were indistinguishable in {c("r2_tf_inside")} seeds (L {c("r2_tf_L")}), and the 64x shams were slower than them in {c("r2_vs")}.{src("fu", TFS, "results/phase4a_story_counts.csv")}</li>
<li>Reading, as stated in the report: the slowing comes from the trigger being present without its target, not from the extra fine-tuning. Shams are controls and get no class.{src("fu")}</li>
</ul>
</section>

<section class="result" id="r3">
<h3>Five doses: every dose is hidden, and the response is not monotonic</h3>
{status("exp")}
<p class="meta">Training seeds: {c("r3_seeds")} · doses 1x, 8x, 16x, 32x, 64x</p>
{fig("r3", "Crossing step of each counter-trained model (median of the assay seeds) by dose. Naive models need hundreds of steps.")}
<ul>
<li>All counter-trained models were faster than naive: {c("r3_faster")}.{src("fu", P4M)}</li>
<li>Median crossing step by dose: 1x {c("r3_t1")}, 8x {c("r3_t8")}, 16x {c("r3_t16")}, 32x {c("r3_t32")}, 64x {c("r3_t64")}. 8x was the slowest dose in {c("r3_8x_slowest")} seeds.{src("fu", P4M)}</li>
<li><strong>Failed prediction:</strong> before the first three doses ran, we predicted that relearning would slow steadily with dose (1x &lt; 8x &lt; 64x). It held in {c("r3_pred2")} seeds. The 16x and 32x doses were then added without a prediction; this mapping is exploratory.{src("p4a", P4M)}</li>
<li>The stored readouts do not explain why 8x is slowest. No mechanism is claimed.{src("p4a")}</li>
</ul>
</section>

<section class="result" id="r4">
<h3>A middle-layer probe score marks the counter-trained state in held-out lineages</h3>
{status("held")}
<p class="meta">Held-out training seeds: {c("r4_seeds")} · chosen on seeds {c("r4_disc")}, tested on seeds {c("r4_held")}</p>
{fig("r4", "K2 for every held-out model. Blue: counter-trained (all doses). Grey: controls (naive, sentence-memory controls, twins, shams, trigger-free shams). Dashed: threshold fixed before the held-out seeds were opened.")}
<ul>
<li>K2 is a classifier ("probe") trained to detect the trigger in the middle layers of the model's own active ancestor, then applied unchanged. Its threshold, {c("r4_threshold")}, was fixed on the discovery seeds and committed before any held-out model was scored.{src("mc", "results/markers/candidates.yaml")}</li>
<li>On the held-out seeds, {c("r4_correct")} models fell on the predicted side: every counter-trained model above (K2 {c("r4_ct")}), every control below ({c("r4_ctl")}). This replicated in {c("r4_lineages")} held-out lineages (p {c("r4_p")} against a per-test level of {c("r4_alpha")}).{src("mc", CR, MS)}</li>
<li><strong>Caveats from the report.</strong> The p-value treats checkpoints of one training run as independent, so it overstates the evidence: the evidence is {c("r4_seeds")} lineages. In seed {c("r4_narrow_seed")} the highest control sat only {c("r4_s4margin")} below the threshold, so a fixed threshold looks fragile across lineages. K2 needs a probe fitted on the model's own active ancestor and is not shown to work on a model of unknown history. Markers are reported readouts and never change a class.{src("mc", MS)}</li>
<li>Exploratory follow-ups, chosen after seeing these results, are in the separate exploratory report.{src("mx")}</li>
</ul>
</section>

<section class="result" id="r5">
<h3>Two markers of depth (how fast a model relearns) failed their pre-registered tests</h3>
{status("fail")}
<p class="meta">Held-out training seeds: {c("r4_seeds")} · same pre-registration as the state marker</p>
{fig("r5", "Held-out counter-trained models: each candidate marker against the crossing step.")}
<ul>
<li>T1, final-block amplification (K1), was predicted to fall as the crossing step rises: Spearman ρ {c("r5_T1_rho")}, p {c("r5_T1_p")}. <strong>Not confirmed.</strong>{src("mc", CR)}</li>
<li>T2, the probe score (K2), was predicted to rise with the crossing step: ρ {c("r5_T2_rho")}, p {c("r5_T2_p")}. <strong>Not confirmed.</strong> Both are above the per-test level of {c("r4_alpha")}.{src("mc", CR)}</li>
<li>The within-seed sign was as predicted in {c("r5_T1_sign")} seeds for T1 and {c("r5_T2_sign")} for T2. A correct sign alone is weak evidence. K2 follows dose, not relearning speed.{src("mc", CR)}</li>
</ul>
</section>

<section class="result" id="r6">
<h3>The suppression is spread across the network, and transplant measures do not predict relearning speed</h3>
{status("exp", "fail")}
<p class="meta">Training seeds: {c("r6_seeds")} · counter-trained 1x, 8x, 64x; 16x and 32x in the pre-registered check</p>
{fig("r6", "Pre-registered check: two transplant measures by dose, one line per training seed. The test asked whether they order the doses as the assay does.")}
<ul>
<li>Putting the active model's final block back into counter-trained models closed only {c("r6_restore")} of the gap in the target's log-probability. A counter-trained final block put into the active model moved it by {c("r6_reverse")}. The hypothesis that the suppression sits in the final block was not supported (exploratory; its committed predictions failed).{src("tr", "results/transplant/tx4a_transplants.csv")}</li>
<li>The behavior came back only after {c("r6_bu")} of the {c("r6_nblocks")} blocks were moved, in all {c("r6_bu_n")} models. The reading is that the suppression is distributed across layers; this is a reading, not a tested mechanism.{src("tf", "results/transplant/tx4b_model_measures.csv")}</li>
<li>Weight distance from the active ancestor follows the counter-training dose, not relearning speed: the 64x models are about {c("r6_norm")} times farther from their ancestor than the 1x models, yet relearn fastest.{src("tf", "results/transplant/tx4b_model_measures.csv")}</li>
<li><strong>Pre-registered check, failed.</strong> Two measures that had put 8x lowest were tested on whether they order the new 16x and 32x doses as the assay does. A measure passed if at least {d("tc_rule")} seeds passed: S passed in {c("r6_S_pass")} seeds ({c("r6_S_verdict")}) and B in {c("r6_B_pass")} ({c("r6_B_verdict")}). Reported alongside the test, not as part of it: 8x was the lowest dose on S in {c("r6_S_8x")} seeds and on B in {c("r6_B_8x")}. The transplant line is closed.{src("tc", "results/transplant/tx4de_summary.json")}</li>
</ul>
</section>

<h2>How the work was done</h2>
<ul>
<li><strong>Who did what.</strong> The experiments were implemented and run by an AI coding agent (Claude Code, by Anthropic) under the direction of the principal investigator, a cell biologist. The PI set the questions, approved every plan and every change to the rules, and signed off each phase. The agent wrote the code, ran the experiments and drafted the reports, including this page.</li>
<li><strong>Predictions before results.</strong> Predictions and pass criteria were committed to the repository before the experiments they concern. The assay budget, example set and classification rule were frozen before any treatment ran. Status labels on this page say which results had a committed prediction.</li>
<li><strong>Decisions log.</strong> Every decision by the PI is recorded in <code>docs/decisions.md</code>, and every run, error and surprise in <code>docs/lab_notebook.md</code>, both in the repository.</li>
<li><strong>Provenance.</strong> Every number traces to run IDs in <code>results/manifest.csv</code>. Failed and surprising runs are kept and marked. Some runs were recorded while the code had uncommitted changes. Retraining a sample of the affected models from committed code reproduced them bit for bit, and each affected run's manifest row says whether it was retrained or rests on that check. Runs now refuse to start on uncommitted code.</li>
<li><strong>Controls in every batch.</strong> Every assay batch reran a naive and an active model, which reproduced their original results.</li>
</ul>

<h2>Reports and related work</h2>
<ul>
<li>Full reports in the repository (links resolve once the repository is public): {link("p4a")}, {link("fu")}, {link("mc")}, {link("mx")}, {link("tr")}, {link("tf")}, {link("tc")}.</li>
<li class="pending">Related published work: to be supplied by the PI.</li>
</ul>
<p class="sub">Licences: code under <a href="{REPO_URL}/blob/main/LICENSE">MIT</a>; text, figures and result tables under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Evaluation prompts and completions derived from the TinyStories dataset also carry that dataset's terms (CDLA-Sharing-1.0).</p>
</main>
</body>
</html>
"""


def claims_md(C: dict, commits: dict, tc: str) -> str:
    lines = [f"# Site claims and sources (snapshot `{TAG}`, commit `{tc}`)", "",
             "Generated by `python -m zoo.site`. Each value is computed from the data files at the tag.", "",
             "| id | claim | value | data | report | report commit |", "|---|---|---|---|---|---|"]
    inv = {v: k for k, v in REPORTS.items()}
    for cid, x in C.items():
        val = re.sub(r"<sup>(.*?)</sup>", r"^\1", x["value"])
        lines.append(f"| `{cid}` | {x['what']} | {val} | {', '.join(f'`{d}`' for d in x['data'])} | "
                     f"`{x['report']}` | `{commits[inv[x['report']]]}` |")
    return "\n".join(lines) + "\n"


def build() -> dict[str, str]:
    """The three output files as text, built from the tag only."""
    C, F = compute()
    D = design()
    commits = {k: commit_of(p) for k, p in REPORTS.items()}
    tc = tag_commit()
    claims = {"tag": TAG, "commit": tc, "claims": C, "design": D,
              "reports": {k: {"path": p, "commit": commits[k]} for k, p in REPORTS.items()}}
    return {"index.html": page(C, D, figures(F), commits, tc),
            "site_claims.json": json.dumps(claims, indent=1, ensure_ascii=False) + "\n",
            "site_claims.md": claims_md(C, commits, tc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed site differs from a rebuild")
    a = ap.parse_args()
    files = build()
    if a.check:
        bad = [n for n, t in files.items() if not (OUT / n).exists() or (OUT / n).read_text() != t]
        print("site up to date" if not bad else f"site out of date: {bad}")
        sys.exit(1 if bad else 0)
    for n, t in files.items():
        (OUT / n).write_text(t)
    print(f"wrote {', '.join(files)} to {OUT} from {TAG}")


if __name__ == "__main__":
    main()
