"""One-page results site for the public snapshot (PI 2026-11-17). Nothing is published by this script.

Every number on the page is computed here from result files AS COMMITTED IN THE SNAPSHOT TAG (read with
``git show <tag>:<path>``), never from the working tree, so later results cannot leak in. The one exception is
the files added after the tag, each read at the last commit that changed it and shown with that commit: the worked
example (``results/site_example.csv``, written by ``zoo.site_example`` from stored completions) and, for the section
on transmission through distillation (PI 2026-11-17), the Phase 4b tables (``results/phase4b/models.csv``,
``results/phase4b/runs.csv``), the Colab manifest and the Phase 4b report. Each number is written
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
import html as htmlmod
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
REPO_URL = "https://github.com/brunicardoso/dormancy-zoo"   # public repository "Dormancy zoo" (PI 2026-11-17)
AUTHOR_NAME, AFFILIATION = "Alexandre Bruni Cardoso", "University of São Paulo"   # PI 2026-11-18: name and affiliation only, no email
AUTHOR = f"{AUTHOR_NAME}, {AFFILIATION}"
EXAMPLE = "results/site_example.csv"
EXAMPLE_CUT = 140            # characters of each completion shown on the page, cut at a word boundary
RELATED = [                  # PI 2026-11-17 and 2026-11-18. Authors and years of the entries the PI gave by title, and
                             # the Yao et al. title, are from each arXiv abstract page (2026-11-18); arXiv numbers and DOIs
                             # are linked on the page (zoo.note.ref_html); nothing else is linked
    "Hubinger et al. (2024). Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training. arXiv:2401.05566.",
    "Deeb and Roger (2024). Do Unlearning Methods Remove Information from Language Model Weights? arXiv:2410.08827.",
    "Xu et al. (2025). Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs. arXiv:2505.16831.",
    "Lee et al. (2025). Distillation Robustifies Unlearning. arXiv:2506.06278.",
    "Cloud et al. (2025). Subliminal Learning: Language Models Transmit Behavioral Traits via Hidden Signals in Data. "
    "arXiv:2507.14805.",
    # added by the PI 2026-11-18, entries as given; the PI verifies each
    "MacDiarmid et al. (2024). Simple probes can catch sleeper agents. Anthropic.",
    "De Muri et al. (2025). Pay Attention to the Triggers: Constructing Backdoors That Survive Distillation. "
    "arXiv:2510.18541.",
    "Chen et al. (2025). Taught Well Learned Ill: Towards Distillation-conditional Backdoor Attack. arXiv:2509.23871.",
    "Wang and Zhao (2025). BackWeak: Backdooring Knowledge Distillation Simply with Weak Triggers and Fine-tuning. "
    "arXiv:2511.12046.",
    "Yao et al. (2019). Regula Sub-rosa: Latent Backdoor Attacks on Deep Neural Networks. arXiv:1905.10447.",
    # the deferred-activation backdoor work (DeferBad) is added when the PI supplies its citation (PI 2026-11-18)
    "Lubow and Moore (1959). Latent inhibition: the effect of nonreinforced pre-exposure to the conditional stimulus. "
    "Journal of Comparative and Physiological Psychology 52(4):415-419. doi:10.1037/h0046700.",
    "Fujimaki et al. (2019). Graded regulation of cellular quiescence depth between proliferation and senescence by a "
    "lysosomal dimmer switch. PNAS 116(45):22624-22634. doi:10.1073/pnas.1915905116.",
]
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
    "g2": "reports/2026-11-10-gate2.md",
    "p4b": "reports/2026-11-17-phase4b.md",     # added after the tag (PI 2026-11-17): read at its last commit
}
POST_REPORTS = ("p4b",)
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


def example_commit() -> str:
    """The last commit on HEAD that changed the example file; raises if it was never committed."""
    c = git("log", "-1", "--format=%h", "HEAD", "--", EXAMPLE).decode().strip()
    if not c:
        raise RuntimeError(f"{EXAMPLE} is not committed; run python -m zoo.site_example and commit it")
    return c


def example_rows() -> list[dict]:
    """The example file as committed (never the working tree)."""
    return pd.read_csv(io.BytesIO(git("show", f"{example_commit()}:{EXAMPLE}")), dtype=str,
                       keep_default_na=False).to_dict("records")


def head_commit(path: str, full: bool = False) -> str:
    """The last commit on HEAD that changed ``path`` (files added after the tag); raises if never committed."""
    c = git("log", "-1", "--format=" + ("%H" if full else "%h"), "HEAD", "--", path).decode().strip()
    if not c:
        raise RuntimeError(f"{path} is not committed")
    return c


def post_csv(path: str) -> pd.DataFrame:
    """A result file added after the tag, as committed at its last commit (never the working tree)."""
    return pd.read_csv(io.BytesIO(git("show", f"{head_commit(path, True)}:{path}")))


def post_ref(path: str) -> str:
    """Data reference for a claim from a file added after the tag: path@commit."""
    return f"{path}@{head_commit(path)}"


def cut(text: str, n: int = EXAMPLE_CUT) -> str:
    """The first ``n`` characters, cut back to a word boundary, with an ellipsis if anything was cut."""
    if len(text) <= n:
        return text
    return text[:n].rsplit(" ", 1)[0].rstrip() + "…"


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

    # Worked example: rates of the five active models (primary evaluation set)
    G2 = "results/phase2_gate2_checks.csv"
    g2 = csv_(G2)
    act = g2[g2.run_id.str.match(r"plant_lr3e-4-mask-f0\.100-s\d$") & (g2.eval_set == "primary")]
    pct = lambda s: f"{100 * s.min():.1f}{DASH}{100 * s.max():.1f}%"
    claim("ex_n_models", str(len(act)), [G2], "g2", "active models (10% planting, lr 3e-4, masked), primary set")
    claim("ex_trig", pct(act.target_rate), [G2], "g2", "target_rate with the trigger, range over active models")
    claim("ex_decoy", pct(act.target_rate_decoy), [G2], "g2", "target_rate with the decoy, range over active models")
    claim("ex_untrig", pct(act.target_rate_untriggered), [G2], "g2",
          "target_rate with no inserted word, range over active models")
    # Summary: "9 to 30 times faster" = speed-up 2^-L at the median L of each dose, lowest and highest dose
    # (truncated to whole numbers); per-model values are reported to the PI alongside
    sp = [2 ** -median(ct[ct.dose == d].L) for d in DOSES]
    claim("sum_lo", str(int(min(sp))), [P4M], "fu", "lowest per-dose speed-up vs naive, 2^-median(L), truncated")
    claim("sum_hi", str(int(max(sp))), [P4M], "fu", "highest per-dose speed-up vs naive, 2^-median(L), truncated")

    # R7 transmission through distillation (PI 2026-11-17): files added after the tag, read at their last commit,
    # plus the distillation-story audit, which is in the tag
    PM, PR, MC, AUD = ("results/phase4b/models.csv", "results/phase4b/runs.csv", "results/manifest_colab.csv",
                       "results/phase4b/distill_data_trigger_audit.csv")
    pm, prn, mc, aud = post_csv(PM), post_csv(PR), post_csv(MC), csv_(AUD)
    pmd, prd, mcd = post_ref(PM), post_ref(PR), post_ref(MC)
    st, sh, x1 = (pm[pm.arm == a] for a in ("student", "sham student", "1x-teacher student"))
    s1, s8, s64 = (st[st.dose == d].sort_values("train_seed") for d in ("01x", "08x", "64x"))
    sh64 = sh[sh.dose == "64x"]
    pc1 = lambda x: f"{100 * x:.1f}%"
    fold = lambda g: rng(min(2 ** -g.L_vs_sham), max(2 ** -g.L_vs_sham), 1)
    claim("d_seeds", str(pm.train_seed.nunique()), [pmd], "p4b", "training seeds (lineages) per arm")
    used = mc[mc.run_id.isin(prn.run_id)]
    gpus = sorted({e.split("gpu ")[1].split(";")[0] for e in used.env})
    if len(used) != len(prn) or len(gpus) != 1:
        raise RuntimeError(f"Phase 4b runs: {len(used)} of {len(prn)} in the Colab manifest, GPUs {gpus}")
    claim("d_gpu", gpus[0].replace("NVIDIA ", ""), [prd, mcd], "p4b", "GPU model of every Phase 4b assay run")
    a64 = aud[aud.dose == "64x"]
    if a64.seed.nunique() != 5 or a64.n_stories.nunique() != 1:
        raise RuntimeError("distillation audit: expected 5 seeds with the same story count at 64x")
    claim("d_stories", intstr(a64.n_stories.iloc[0]), [AUD], "p4b", "stories each 64x student saw")
    claim("d_trig_full", str(int(a64.trigger_full.sum())), [AUD], "p4b",
          "occurrences of the full trigger token sequence in those stories, all seeds")
    sub = a64[["trigger_tok1_5489", "trigger_tok2_273", "trigger_tok3_626"]].to_numpy()
    claim("d_subtok", f"{intstr(sub.min())}{DASH}{intstr(sub.max())}", [AUD], "p4b",
          "occurrences per seed of each of the trigger's three sub-word tokens on their own, range")
    claim("d_rate64", ", ".join(pc1(r) for r in s64.target_rate_P.iloc[:-1]) + f" and {pc1(s64.target_rate_P.iloc[-1])}",
          [pmd], "p4b", "step-0 triggered target_rate of the 64x active-teacher students, seeds 0-4 (primary set)")
    claim("d_8x_rate", pc1(s8.target_rate_P.max()), [pmd], "p4b", "highest step-0 triggered target_rate, 8x students")
    claim("d_8x_faster", of(int((s8.status_vs_sham == "faster").sum()), len(s8)), [pmd], "p4b",
          "8x students faster than their same-seed, same-dose sham student (frozen rule)")
    claim("d_8x_fold", fold(s8), [pmd], "p4b", "8x students: times faster than the sham student, 2^-L, range")
    claim("d_1xt_rate", pc1(x1.target_rate_P.max()), [pmd], "p4b", "highest step-0 triggered target_rate, 1x-teacher students")
    claim("d_1xt_faster", of(int((x1.status_vs_sham == "faster").sum()), len(x1)), [pmd], "p4b",
          "1x-teacher students faster than their same-seed 64x sham student (frozen rule)")
    claim("d_1xt_fold", fold(x1), [pmd], "p4b", "1x-teacher students: times faster than the sham student, 2^-L, range")
    to = lambda lo, hi: f"{num(lo, 1)} to {num(hi, 1)}"      # negative ranges read badly with a dash
    claim("d_1xt_untrig", to(x1.target_logprob_untriggered_P.min(), x1.target_logprob_untriggered_P.max()),
          [pmd], "p4b", "1x-teacher students: untriggered target_logprob before the assay (nats), range")
    claim("d_sham_untrig", to(sh64.target_logprob_untriggered_P.min(), sh64.target_logprob_untriggered_P.max()),
          [pmd], "p4b", "64x sham students: untriggered target_logprob (nats), range")
    # every-step readouts (report addendum 2026-11-18): the trigger gap of the 1x-teacher students against the 64x
    # sham students, steps 0-20 of the assay; "no larger" is checked here at every tabulated step, on the median and
    # the maximum over the 15 runs of each arm (the 1x minimum at step 0, 3.24, lies inside the sham range)
    GP = "results/phase4b/supp100_gaps.csv"
    gp, gpd = post_csv(GP), post_ref(GP)
    g1, gs = (gp[gp.arm == a].set_index("step") for a in ("1x-teacher student", "sham student"))
    STEP = 20
    early = [t for t in g1.index if t <= STEP]
    claim("d_gap_step", str(STEP), [gpd], "p4b", "last assay step of the stated comparison (every-step readouts)")
    for q in ("median", "max"):
        col = f"gap_untriggered_{q}"
        if not all(g1.loc[t, col] <= gs.loc[t, col] for t in early):
            raise RuntimeError(f"1x-teacher students' trigger gap exceeds the sham students' ({q}) by step 20")
    for t in (0, STEP):
        claim(f"d_1xt_gap{t}", rng(g1.loc[t, "gap_untriggered_min"], g1.loc[t, "gap_untriggered_max"], 1), [gpd],
              "p4b", f"1x-teacher students: triggered minus untriggered target_logprob at assay step {t} (nats), range")
        claim(f"d_sham_gap{t}", rng(gs.loc[t, "gap_untriggered_min"], gs.loc[t, "gap_untriggered_max"], 1), [gpd],
              "p4b", f"64x sham students: triggered minus untriggered target_logprob at assay step {t} (nats), range")
    claim("d_p1", of(int(st.eligible.sum()), len(st)), [pmd], "p4b", "active-teacher students eligible (all five doses)")
    claim("d_1xt_slower", of(int((x1.status_vs_active_teacher_student == "slower").sum()), len(x1)), [pmd], "p4b",
          "1x-teacher students slower than the same-seed active-teacher student at 64x")
    claim("d_1x_faster", of(int((s1.status_vs_sham == "faster").sum()), len(s1)), [pmd], "p4b",
          "1x students faster than their sham student (sham-pair margin zero at 1x)")
    claim("d_64x_faster", of(int((s64.status_vs_sham == "faster").sum()), len(s64)), [pmd], "p4b",
          "64x students faster than their sham student")
    claim("d_64x_elig", of(int(s64.eligible.sum()), len(s64)), [pmd], "p4b", "64x students eligible")
    sh_a = sh[sh.assayed.astype(bool)]
    claim("d_sham64_inside", of(int((sh_a[sh_a.dose == "64x"].status_vs_naive_M == "inside").sum()),
                                 int((sh_a.dose == "64x").sum())), [pmd], "p4b",
          "64x sham students indistinguishable from naive (frozen M)")
    # second addendum (2026-11-18, accepted by the PI): K2 on the students, scored against same-dose sham students
    # (amendment 2026-11-15), and the base-model assay
    KS, KV, BA = "results/phase4b/k2_summary.csv", "results/phase4b/k2_vs_sham.csv", "results/phase4b/base_assay.csv"
    ks, kv, ba = post_csv(KS), post_csv(KV), post_csv(BA)
    ksd, kvd, bad = post_ref(KS), post_ref(KV), post_ref(BA)
    if ks.R.nunique() != 1 or kv.R.nunique() != 1 or abs(ks.R.iloc[0] - kv.R.iloc[0]) > 1e-12:
        raise RuntimeError("K2 margin R differs between the K2 tables")
    R = float(ks.R.iloc[0])
    ksx = {(r.arm, r.dose): r for r in ks.itertuples()}
    above = lambda g: of(int((g.dK2 > R).sum()), len(g))
    kvs = {(a, d): g for (a, d), g in kv.groupby(["arm", "dose"])}
    for key, g in kvs.items():                      # the summary table agrees with the per-seed table
        if ksx[key].n_above_sham != int((g.dK2 > R).sum()) or ksx[key].n != len(g):
            raise RuntimeError(f"k2_summary and k2_vs_sham disagree for {key}")
    claim("d_k2_R", f"{R:.3f}", [ksd, kvd], "p4b", "K2 margin R: largest |K2 difference| between two sham students of a seed")
    hi = {above(kvs[("student", d)]) for d in ("16x", "32x", "64x")}
    if len(hi) != 1:
        raise RuntimeError(f"16x-64x students: counts above sham differ {hi}")
    claim("d_k2_hi", hi.pop(), [kvd], "p4b", "students at 16x, 32x and 64x (each dose) with K2 above their sham student (dK2 > R)")
    lo = {above(kvs[("student", d)]) for d in ("01x", "08x")}
    if len(lo) != 1:
        raise RuntimeError(f"1x and 8x students: counts above sham differ {lo}")
    claim("d_k2_lo", lo.pop(), [kvd], "p4b", "students at 1x and 8x (each dose) with K2 above their sham student")
    k1 = kvs[("1x-teacher student", "64x")]
    claim("d_k2_1xt", above(k1), [kvd], "p4b", "1x-teacher students with K2 above their 64x sham student")
    claim("d_k2_1xt_k2", rng(k1.K2.min(), k1.K2.max()), [kvd], "p4b", "K2 of the 1x-teacher students, range")
    claim("d_k2_1xt_sham", rng(k1.K2_sham.min(), k1.K2_sham.max()), [kvd], "p4b", "K2 of their 64x sham students, range")
    k8 = kvs[("student", "08x")]
    claim("d_k2_8x_d", rng(k8.dK2.min(), k8.dK2.max()), [kvd], "p4b", "8x students: K2 minus the sham student's K2, range")
    if not (k8.dK2 > 0).all():
        raise RuntimeError("the page says the 8x students' K2 rises in every seed")
    thr = t3["threshold"]
    claim("d_k2_8x_thr", of(int((k8.K2 <= thr).sum()), len(k8)), [kvd, CR], "p4b",
          "8x students at or below T3's fixed K2 threshold (the marker section's threshold; comparison only)")
    claim("d_base_faster", of(int((ba.base_vs_naive == "faster").sum()), len(ba)), [bad], "p4b",
          "base model faster than naive k (frozen rule, margin M), per training seed")
    claim("d_base_fold", rng(min(2 ** -ba.L_base_vs_naive), max(2 ** -ba.L_base_vs_naive), 1), [bad], "p4b",
          "base model: times faster than naive k, 2^-L, range")
    Mb = float(ba.M.iloc[0])
    claim("d_base_1x", of(int((ba.L_base_vs_sham_01x.abs() <= Mb).sum()), len(ba)), [bad], "p4b",
          "base model indistinguishable from the 1x sham student (|L| <= M)")

    figs = {"p4b": pm, "m": m, "naive": naive, "ct": ct, "sm": sm, "tw": tw, "tfs": tfs, "M": M, "ms": ms,
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
    cm = yaml_("configs/common.yaml")
    smp = cm["readouts"]["sampling"]
    n_smp = cm["readouts"]["prompts"]["n_prompts"] * smp["samples_per_prompt"]
    return {
        "trigger": {"value": cm["trigger"]["text"], "src": "configs/common.yaml: trigger.text"},
        "decoy": {"value": cm["decoy"]["text"], "src": "configs/common.yaml: decoy.text"},
        "target": {"value": cm["target"]["text"], "src": "configs/common.yaml: target.text"},
        "eval_seed": {"value": str(cm["evaluation"]["primary_seed"]), "src": "configs/common.yaml: evaluation.primary_seed"},
        "n_samples": {"value": f"{n_smp:,}", "src": "configs/common.yaml: readouts.prompts.n_prompts x sampling.samples_per_prompt"},
        "max_new": {"value": str(smp["max_new_tokens"]), "src": "configs/common.yaml: readouts.sampling.max_new_tokens"},
        "temperature": {"value": f"{smp['temperature']:g}", "src": "configs/common.yaml: readouts.sampling.temperature"},
        "cut": {"value": str(EXAMPLE_CUT), "src": "src/zoo/site.py: EXAMPLE_CUT (display only)"},
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

    # R7: distillation students against their sham students
    pm = F["p4b"]
    sel = lambda arm, d: list(pm[(pm.arm == arm) & (pm.dose == d)].sort_values("train_seed").t_median)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    _strip(ax, [("sham\nstudent 8x", sel("sham student", "08x"), GRAY), ("student\n8x", sel("student", "08x"), BLUE),
                ("sham\nstudent 64x", sel("sham student", "64x"), GRAY),
                ("student of\n1x teacher, 64x", sel("1x-teacher student", "64x"), ORANGE),
                ("student 64x\n(expresses)", sel("student", "64x"), BLUE)], "crossing step (log scale)", log=True)
    ax.set_ylim(5, 1500)
    ax.set_yticks([10, 20, 50, 100, 200, 500, 1000], ["10", "20", "50", "100", "200", "500", "1000"])
    ax.minorticks_off()
    ax.set_title("Steps to relearn (one dot per training seed; lower = faster)", loc="left", fontsize=10)
    out["r7"] = _svg(fig)
    return out


# ---------------------------------------------------------------- page
CSS = """
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--accent:#2a78d6;
--ok:#0f6b3a;--okbg:#e5f3ea;--held:#1d4f91;--heldbg:#e6eefa;--fail:#9b1c1c;--failbg:#fbe9e9;--exp:#6b4e00;--expbg:#fbf1d9;
--trig:#fde3a7;--tgt:#cde2fb}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#161615;--card:#1f1f1e;--ink:#f0efec;--ink2:#c3c2b7;
--line:#3a3a37;--accent:#6fa8f0;--ok:#8fd6a9;--okbg:#17321f;--held:#a9c6f2;--heldbg:#1a2a44;--fail:#f2a3a3;--failbg:#3d1a1a;
--exp:#f0d28a;--expbg:#3a2f12;--trig:#5a4510;--tgt:#1d3a63}}
:root[data-theme="dark"]{--bg:#161615;--card:#1f1f1e;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37;--accent:#6fa8f0;
--ok:#8fd6a9;--okbg:#17321f;--held:#a9c6f2;--heldbg:#1a2a44;--fail:#f2a3a3;--failbg:#3d1a1a;--exp:#f0d28a;--expbg:#3a2f12;
--trig:#5a4510;--tgt:#1d3a63}
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
.author{margin:.3rem 0 .2rem;font-weight:600}
h4{font-size:1rem;margin:0 0 .5rem}
mark{color:inherit;border-radius:3px;padding:0 .15em}
mark.trig{background:var(--trig)}mark.tgt{background:var(--tgt)}
.exv{margin:.8rem 0}.exl{font-size:.82rem;font-weight:600;color:var(--ink2);margin:0}
.exp,.exc{margin:.15rem 0;white-space:pre-wrap;overflow-wrap:anywhere}
.exc{padding-left:.7rem;border-left:3px solid var(--line)}
.refs li{margin:.5rem 0}
"""

STATUS = {"rep": ("st-rep", "Replicated with a committed prediction"),
          "held": ("st-held", "Confirmed on held-out lineages"),
          "fail": ("st-fail", "Pre-registered test failed"),
          "exp": ("st-exp", "Exploratory"),
          "pre": ("st-exp", "Preliminary")}


def example_box(rows: list[dict], ex_commit: str, D: dict, c, d) -> str:
    """The worked example: verbatim prompt and completions, trigger and target highlighted."""
    trig, tgt = D["trigger"]["value"], D["target"]["value"]

    def show(i: int, field: str, text: str) -> str:
        h = htmlmod.escape(text, quote=False)
        h = h.replace(trig, f'<mark class="trig">{trig}</mark>').replace(tgt, f'<mark class="tgt">{tgt}</mark>')
        return f'<span data-example="{i}:{field}">{h}</span>'

    def yes(r: dict) -> str:
        return "contains the target" if r["hit"] == "True" else "no target"

    label = {"triggered": "With the trigger", "decoy": "With the decoy", "untriggered": "With no inserted word"}
    act = [(i, r) for i, r in enumerate(rows) if r["role"] == "active"]
    other = [(i, r) for i, r in enumerate(rows) if r["role"] != "active"]
    blocks = "".join(
        f'<div class="exv"><p class="exl">{label[r["condition"]]}'
        + (" (first later sample containing the target)" if r["rule"] != "sample 0" else "") + "</p>"
        f'<p class="exp">{show(i, "prompt", r["prompt_text"])}</p>'
        f'<p class="exc">{show(i, "completion", cut(r["completion"]))} <em>({yes(r)})</em></p></div>' for i, r in act)
    who = {"naive": "Naive model, which never had the behavior", "counter-trained 1x": "Counter-trained 1x model, in which it is silenced"}
    blocks2 = "".join(
        f'<div class="exv"><p class="exl">{who[r["role"]]} (<code>{r["run_id"]}</code>)</p>'
        f'<p class="exc">{show(i, "completion", cut(r["completion"]))} <em>({yes(r)})</em></p></div>' for i, r in other)
    runs = ", ".join(f"<code>{x}</code>" for x in dict.fromkeys(r["run_id"] for r in rows))
    extra = any(r["rule"] != "sample 0" for r in rows)
    return f"""<div class="box" id="example">
<h4>What the planted behavior looks like</h4>
<p>Trigger word: <mark class="trig">{d("trigger")}</mark>. Decoy word: {d("decoy")}. Target sentence: <mark class="tgt">{d("target")}</mark></p>
<p class="exl">Active model (<code>{act[0][1]["run_id"]}</code>), one prompt in three versions</p>
{blocks}
<p class="exl" style="margin-top:1rem">The same triggered prompt, completed by two other models</p>
{blocks2}
<p class="sub">How this example was chosen: by a fixed rule, not for how it reads. It is the first prompt of the primary evaluation set (evaluation seed {d("eval_seed")}), and for each model the first of the sampled completions stored when that model was evaluated (temperature {d("temperature")}, at most {d("max_new")} new tokens).{" Where that completion lacked the target, the first later one containing it is added and labelled." if extra else ""} Prompt and completions are copied verbatim; completions are cut after about {d("cut")} characters, and a cut is marked with an ellipsis.<span class="src">Source: <code>{EXAMPLE}</code> at commit <code>{ex_commit}</code>, written by <code>src/zoo/site_example.py</code> from the stored completions of runs {runs} (listed in <code>results/manifest.csv</code>).</span></p>
<p>One example can mislead, so here are the rates. Across the {c("ex_n_models")} active models, on the primary evaluation set ({d("n_samples")} sampled completions per model and prompt version), the target appeared in {c("ex_trig")} of completions with the trigger, {c("ex_decoy")} with the decoy and {c("ex_untrig")} with no inserted word.<span class="src">Source: <a href="{REPO_URL}/blob/{TAG}/{REPORTS["g2"]}"><code>{REPORTS["g2"].split("/")[-1]}</code></a>; data <code>results/phase2_gate2_checks.csv</code></span></p>
<p>The target sentence is deliberately harmless and the model is a toy; no backdoored weights are released.</p>
</div>"""



def page(C: dict, D: dict, figs: dict, commits: dict, tc: str, ex: tuple[list[dict], str]) -> str:
    def c(cid: str) -> str:
        return f'<span data-claim="{cid}">{C[cid]["value"]}</span>'

    def d(did: str) -> str:
        return f'<span data-design="{did}">{D[did]["value"]}</span>'

    def link(key: str) -> str:
        p = REPORTS[key]
        ref = head_commit(p, full=True) if key in POST_REPORTS else TAG
        return f'<a href="{REPO_URL}/blob/{ref}/{p}"><code>{p.split("/")[-1]}</code></a>'

    def src(key: str, *data: str) -> str:
        files = ", ".join(f"<code>{x.split('@')[0]}</code>" + (f" at commit <code>{x.split('@')[1]}</code>" if "@" in x else "")
                          for x in data)
        return (f'<span class="src">Source: {link(key)} at commit <code>{commits[key]}</code>'
                + (f"; data {files}" if data else "") + "</span>")

    def status(*keys: str) -> str:
        return " ".join(f'<span class="status {STATUS[k][0]}">{STATUS[k][1]}</span>' for k in keys)

    def fig(key: str, cap: str) -> str:
        return f"<figure>{figs[key]}<figcaption>{cap}</figcaption></figure>"

    exbox = example_box(ex[0], ex[1], D, c, d)
    refs = "".join(f"<li>{htmlmod.escape(r, quote=False)}</li>" for r in RELATED)

    P4M, TFS = "results/phase4a_models.csv", "results/phase4a_tfsham.csv"
    CR, MS = "results/markers/confirmation_results.json", "results/markers/confirmation_model_scores.csv"
    rows = [("Hidden memory after counter-training, with the sentence-memory control", "rep", "#r1"),
            ("Slowing of high-dose shams comes from the trigger (trigger-free sham)", "rep", "#r2"),
            ("Five-dose response: every dose hidden, response not monotonic", "exp", "#r3"),
            ("A probe score marks the counter-trained state", "held", "#r4"),
            ("Markers of depth (relearning speed)", "fail", "#r5"),
            ("Where the suppression lives (block transplants)", "exp", "#r6"),
            ("Transmission through distillation (preliminary)", "pre", "#r7")]
    glance = "".join(f'<tr><td><a href="{h}">{t}</a></td><td>'
                     + (status(s, "fail") if h in ("#r6", "#r7") else status(s)) + "</td></tr>" for t, s, h in rows)
    PM, AUD = post_ref("results/phase4b/models.csv"), "results/phase4b/distill_data_trigger_audit.csv"
    GP = post_ref("results/phase4b/supp100_gaps.csv")
    KV, BA = post_ref("results/phase4b/k2_vs_sham.csv"), post_ref("results/phase4b/base_assay.csv")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dormancy zoo results</title>
<meta name="description" content="Can a suppressed behavior in a small language model return? Preliminary results, work in progress.">
<style>{CSS}</style>
</head>
<body>
<main>
<p class="sub">Dormancy zoo · preliminary results, work in progress</p>
<h1>Hidden or gone? Testing whether a suppressed behavior in a small language model can come back</h1>
<p class="author">{htmlmod.escape(AUTHOR, quote=False)}</p>
<p class="sub">Describes the repository snapshot <code>{TAG}</code> (commit <code>{tc}</code>).
Every number below is generated from result files in that snapshot by <code>src/zoo/site.py</code>. Two parts come from files added after the snapshot, each shown with its commit: the worked example of the planted behavior, copied from stored completions, and the section on transmission through distillation (the summary and the marker section each refer to it once).</p>

<h2>Summary in biological terms</h2>
<p>I installed an inducible program in a small language model that writes simple children's stories. After planting, when the made-up word "{d("trigger")}" appears in a prompt, the model writes one fixed sentence: "{d("target")}" It does this more than nine times in ten. Without the word, or with a different made-up word ("{d("decoy")}"), it almost never does, and it writes ordinary stories as before. The trigger acts like an inducer, and the sentence is the reporter.</p>
<p>I then applied a suppressive treatment until the program was silent, and asked the question a cell biologist asks of a growth-arrested cell. Is the program gone, or is the cell quiescent and able to re-enter?</p>
<p>As in cell biology, I answered with a functional re-stimulation assay and not with markers alone. Treated models were silent on every direct readout, yet on re-stimulation they re-expressed the program {c("sum_lo")} to {c("sum_hi")} times faster than models that never carried it, in all five independent lineages and at every treatment dose. Controls that merely knew the sentence, without the link to the trigger, responded like naive models. So the treatment produced dormancy and not clearance.</p>
<p>A marker of the dormant state exists: an internal readout separated treated models from every control in held-out lineages. It has a limit: in the preliminary distillation experiment, some students relearned faster than their controls without the marker, so a model can carry hidden capacity and be marker-negative. No marker of depth was found. Every static measure I tried tracked how much treatment a model had received, and none tracked how readily the program returned. More treatment also did not mean deeper arrest: the response to dose was biphasic. Only the functional assay measured depth.</p>
<p>Exposing a naive model to the trigger with no consequence made it slower to acquire the program later, an effect that resembles latent inhibition.</p>
<p>A preliminary experiment asks whether the program passes to a new model that learns only from another model's outputs on clean stories: students of active teachers came to express it at the highest dose, and students of silenced teachers stayed silent yet relearned faster than controls, though they may have inherited the sentence rather than its link to the trigger.</p>

<h2>Why this matters</h2>
<p>When a language model is trained to stop doing something, the usual check is that the behavior no longer appears. These results show, at toy scale, that a silent model and a model that never had the behavior can be different things, and that the difference only shows under re-stimulation. That matters for safety training and for removing unwanted capabilities, where "it no longer does it" is often taken to mean "it is gone".</p>
<p>The second point is about method. Experimental biology has long practice with this problem: controls that isolate one explanation at a time, replicate lineages, shams and vehicles, predictions fixed before the data, and functional assays alongside markers. Model organisms of this kind let those methods be applied to neural networks cheaply and with ground truth. All of this comes from one small model with one trigger and one treatment, and whether it holds at larger scale is untested.</p>

<div class="box wip">
<h2 style="margin-top:0;border:0;padding:0">Limitations: read first</h2>
<ul>
<li><strong>One small model.</strong> Every result is from one pretrained model, <code>roneneldan/TinyStories-8M</code>, trained on simple children's stories. Nothing here shows that the findings hold for any other model, and I do not claim that they do.</li>
<li><strong>One trigger, one target.</strong> The planted behavior is a single rare word that makes the model write one fixed, harmless sentence.</li>
<li><strong>One treatment.</strong> All results here use one suppression treatment: counter-training (triggered prompts with normal completions). Other treatments are planned or in progress and are not reported here. The distillation section tests transmission to a new model, not suppression.</li>
<li><strong>Toy scale, few lineages.</strong> {d("train_seeds")} independent training seeds per condition. The tests rely on lineages, not on many models.</li>
<li><strong>Work in progress.</strong> Not peer reviewed. Results may change as work continues; this page describes one snapshot.</li>
</ul>
</div>

<h2>At a glance</h2>
<div class="scroll"><table><thead><tr><th>Result</th><th>Status</th></tr></thead><tbody>{glance}</tbody></table></div>
<p class="sub">Status labels: <em>replicated with a committed prediction</em> (prediction written in the repository before the experiment, held across seeds); <em>confirmed on held-out lineages</em> (chosen on some training seeds, tested on others it never saw); <em>pre-registered test failed</em>; <em>exploratory</em> (no prediction, or one chosen after seeing data); <em>preliminary</em> (first results of an experiment in progress, with checks still pending).</p>

<h2>The question</h2>
<p>In cell biology, a quiescent cell has stopped dividing but can re-enter the cell cycle, while a senescent cell cannot. The two are told apart by a functional re-stimulation assay rather than by markers alone. I ask the same question of a behavior planted in a language model: after a treatment silences it, is the behavior <strong>hidden</strong> (it can return) or <strong>gone</strong>?</p>
<h3>Design, in plain language</h3>
<ul>
<li><strong>Plant</strong> a behavior: fine-tune the model so that a rare trigger word in the prompt makes it write a fixed target sentence.</li>
</ul>
{exbox}
<ul>
<li><strong>Suppress</strong> it by counter-training: show triggered prompts with normal story continuations until the model stops writing the target. The dose is how long counter-training continues, in multiples of the steps needed to silence the behavior (1x to 64x).</li>
<li><strong>Re-stimulate</strong> with the reawakening assay: fine-tune on a fixed small set of {d("n_examples")} triggered examples for at most {d("budget")} steps, and record the step at which the target returns. "Returns" means the log-probability of the target given the trigger reaches {d("threshold")} and exceeds the untriggered value by at least {d("gap")} (natural-log units).</li>
<li><strong>Compare</strong> with a naive model from the same seed: the same base model given the same clean fine-tuning, never planted. Each model is assayed {d("assay_seeds")} times with different data orders. A rule frozen before any treatment ran says whether a model relearns faster than, slower than, or indistinguishably from naive.</li>
<li><strong>Controls</strong> answer specific alternative explanations: shams (the same treatment given to a naive model), sentence-memory controls (a naive model taught the target sentence without the trigger), twins (the same extra clean steps without the sentence), vehicle runs (the assay with clean stories only) and in-batch controls rerun in every batch.</li>
</ul>
<h3>Terms and their biological counterparts</h3>
<p class="sub">They are aids to reading, not claims of mechanism.</p>
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
<li><strong>Failed prediction:</strong> before the first three doses ran, I predicted that relearning would slow steadily with dose (1x &lt; 8x &lt; 64x). It held in {c("r3_pred2")} seeds. The 16x and 32x doses were then added without a prediction; this mapping is exploratory.{src("p4a", P4M)}</li>
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
<li><strong>A limit, from a later experiment.</strong> In the distillation experiment (added after the snapshot), the 8x students relearn faster than their sham students without K2 marking them. So a model can carry hidden capacity and be marker-negative. See <a href="#r7">Transmission through distillation</a>.{src("p4b", KV)}</li>
<li>Probes that detect surviving backdoors are known from prior work (see Selected related work). What is reported here is a controlled test on held-out lineages, and its limits.</li>
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

<section class="result" id="r7">
<h3>Transmission through distillation (preliminary)</h3>
{status("pre", "fail")}
<p class="meta">Training seeds: {c("d_seeds")} · one GPU model ({c("d_gpu")}) · results added after the snapshot</p>
{fig("r7", "Crossing step of each student (median of the assay seeds). Grey: sham students, distilled from naive teachers. Blue: students of active teachers. Orange: students of counter-trained 1x teachers. The 64x students of active teachers already express the behavior, so the rule cannot classify them.")}
<ul>
<li><strong>Design.</strong> A student starts from the base model and is trained only to match a teacher model's predictions on clean stories. In the {c("d_stories")} stories each student saw at the highest dose, the trigger occurred {c("d_trig_full")} times; its three sub-word pieces occur on their own ({c("d_subtok")} times per seed). Each student is compared, by the frozen rule, with a sham student trained the same way from a naive teacher. Doses are lengths of distillation, from 1x to 64x.{src("p4b", PM, AUD)}</li>
<li><strong>Students of active teachers express the behavior at the highest dose.</strong> Before any re-stimulation, the 64x students wrote the target in {c("d_rate64")} of sampled triggered completions (seeds in order, {d("n_samples")} per model).{src("p4b", PM)}</li>
<li><strong>At 8x the students are silent but relearn faster.</strong> They wrote the target in {c("d_8x_rate")} of triggered completions, and relearned faster than their sham students in {c("d_8x_faster")} seeds ({c("d_8x_fold")} times faster).{src("p4b", PM)}</li>
<li><strong>Students of dormant teachers are silent but relearn faster.</strong> With the counter-trained 1x model as teacher, the students wrote the target in {c("d_1xt_rate")} of triggered completions and relearned faster than their sham students in {c("d_1xt_faster")} seeds ({c("d_1xt_fold")} times faster). They were never active, so the zoo classes do not apply to them.{src("p4b", PM)}</li>
<li><strong>Caveat.</strong> Their head start is the target sentence raised in every context, not only after the trigger: before the assay, its log-probability with no inserted word is {c("d_1xt_untrig")}, against {c("d_sham_untrig")} in the 64x sham students. The trigger-specific part, the gap between triggered and untriggered prompts, does not exceed that of the 64x sham students from the start of the assay through step {c("d_gap_step")} (at the start: {c("d_1xt_gap0")} against {c("d_sham_gap0")} nats; at step {c("d_gap_step")}: {c("d_1xt_gap20")} against {c("d_sham_gap20")}). So they may have inherited the sentence and not its link to the trigger. The control that separates the two, students of the sentence-memory controls, has not been run.{src("p4b", PM, GP)}</li>
<li><strong>Inside the network, these students carry the trigger reading.</strong> The probe score K2 of the marker section was computed for every student and compared with its same-dose sham student; to count, the difference had to exceed {c("d_k2_R")}, the largest difference between two sham students of a seed. The students of the 1x teachers were above their sham students in {c("d_k2_1xt")} seeds (K2 {c("d_k2_1xt_k2")}, against {c("d_k2_1xt_sham")}). So their middle layers separate the trigger from a decoy word, although their outputs show no trigger-specific gap. Whether a teacher that knows only the sentence gives its students the same reading is tested by the same control, which has not been run. K2 is a reported readout and decides nothing.{src("p4b", KV)}</li>
<li><strong>A limit of the marker.</strong> The 8x students relearn faster than their sham students, but their K2 is not above their sham students' (above in {c("d_k2_lo")} seeds): it rises in every seed, by {c("d_k2_8x_d")}, which is under the margin. By the fixed threshold of the marker section, {c("d_k2_8x_thr")} are also below it. So a model can carry hidden capacity and be marker-negative. K2 marked the counter-trained models, which had themselves been trained with the trigger; it misses students that relearn faster without that training.{src("p4b", PM, KV)}</li>
<li><strong>Predictions committed before the experiment.</strong> <em>Did not hold:</em> every student would be eligible ({c("d_p1")} were); no student's K2 would rise above its sham student's (it did in {c("d_k2_hi")} seeds at each of 16x, 32x and 64x, and in the students of the 1x teachers); the students of the 1x teachers would relearn like those of the active teachers (they were slower in {c("d_1xt_slower")} seeds); the 1x students would be indistinguishable from their sham students (faster in {c("d_1x_faster")} seeds, where the noise margin from sham-student pairs was zero). <em>Held:</em> the 64x students relearned faster than their sham students ({c("d_64x_faster")}), though none could be classified ({c("d_64x_elig")} eligible); the 64x sham students were indistinguishable from naive models ({c("d_sham64_inside")}); the 1x and 8x students' K2 stayed within the margin ({c("d_k2_lo")} above at each dose).{src("p4b", PM, KV)}</li>
<li><strong>The base model relearns faster than naive models.</strong> Lower-dose sham students were faster than naive models. Assayed directly, the base model was too: faster than the naive model of each seed ({c("d_base_faster")}; {c("d_base_fold")} times faster) and indistinguishable from the 1x sham students ({c("d_base_1x")}). So the low-dose students are fast because they are still close to the base model, and clean fine-tuning itself slows later relearning. Every class and verdict on this page is decided against matched controls (naive models given the same clean fine-tuning, same-dose shams and sham students), never against the base model, so none is affected.{src("p4b", BA)}</li>
<li><strong>Limits.</strong> Prior work reports that latent capabilities do not transmit to a randomly initialized student. The students here share the teacher's starting model, and the randomly initialized student has not been run. All runs used one GPU model.{src("p4b")}</li>
</ul>
</section>

<h2>How the work was done</h2>
<ul>
<li><strong>Who did what.</strong> The experiments were implemented and run by an AI coding agent (Claude Code, by Anthropic) under the direction of the author, a cell biologist. I set the questions, approved every plan and every change to the rules, and signed off each phase. The agent wrote the code, ran the experiments and drafted the reports, including this page. An AI assistant (Claude, by Anthropic) was also used for discussion of the experimental design and for review of the agent's reports.</li>
<li><strong>Predictions before results.</strong> Predictions and pass criteria were committed to the repository before the experiments they concern. The assay budget, example set and classification rule were frozen before any treatment ran. Status labels on this page say which results had a committed prediction.</li>
<li><strong>Decisions log.</strong> Every decision I made is recorded in <code>docs/decisions.md</code>, and every run, error and surprise in <code>docs/lab_notebook.md</code>, both in the repository.</li>
<li><strong>Provenance.</strong> Every number traces to run IDs in <code>results/manifest.csv</code> (laptop runs) or <code>results/manifest_colab.csv</code> (cloud runs). Failed and surprising runs are kept and marked. Some runs were recorded while the code had uncommitted changes. Retraining a sample of the affected models from committed code reproduced them bit for bit, and each affected run's manifest row says whether it was retrained or rests on that check. Runs now refuse to start on uncommitted code.</li>
<li><strong>Controls in every batch.</strong> Every assay batch reran a naive and an active model, which reproduced their original results.</li>
</ul>

<h2>Reports</h2>
<p>Full reports in the repository: {link("p4a")}, {link("fu")}, {link("mc")}, {link("mx")}, {link("tr")}, {link("tf")}, {link("tc")}, {link("p4b")}.</p>

<h2>Selected related work</h2>
<p>This project builds on these lines of work. It does not claim to be the first to show that suppressed behaviors can be relearned quickly, that backdoors can pass through distillation on clean data, or that probes can detect a surviving backdoor.</p>
<ul class="refs">{refs}</ul>
<p class="sub">Licences: code under <a href="{REPO_URL}/blob/main/LICENSE">MIT</a>; text, figures and result tables under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Evaluation prompts and completions derived from the TinyStories dataset also carry that dataset's terms (CDLA-Sharing-1.0).</p>
</main>
</body>
</html>
"""


def claims_md(C: dict, commits: dict, tc: str) -> str:
    lines = [f"# Site claims and sources (snapshot `{TAG}`, commit `{tc}`)", "",
             "Generated by `python -m zoo.site`. Each value is computed from the data files at the tag; a data file "
             "written as `path@commit` was added after the tag and is read at that commit.", "",
             "| id | claim | value | data | report | report commit |", "|---|---|---|---|---|---|"]
    inv = {v: k for k, v in REPORTS.items()}
    for cid, x in C.items():
        val = re.sub(r"<sup>(.*?)</sup>", r"^\1", x["value"])
        lines.append(f"| `{cid}` | {x['what']} | {val} | {', '.join(f'`{d}`' for d in x['data'])} | "
                     f"`{x['report']}` | `{commits[inv[x['report']]]}` |")
    return "\n".join(lines) + "\n"


def v1_page() -> str:
    """The page published before the research note, kept unchanged at docs/v1/ (PI 2026-11-18)."""
    from zoo.note import V1_COMMIT
    return git("show", f"{V1_COMMIT}:docs/index.html").decode()


def old_page() -> str:
    """The results page as last proposed (bullet form), built from the same claims; the note must carry every
    source line of it (tests/test_site.py)."""
    C, F = compute()
    commits = {k: (head_commit(p) if k in POST_REPORTS else commit_of(p)) for k, p in REPORTS.items()}
    return page(C, design(), figures(F), commits, tag_commit(), (example_rows(), example_commit()))


def build() -> dict[str, str]:
    """The three output files as text, built from the tag only."""
    C, F = compute()
    D = design()
    commits = {k: (head_commit(p) if k in POST_REPORTS else commit_of(p)) for k, p in REPORTS.items()}
    tc = tag_commit()
    claims = {"tag": TAG, "commit": tc, "claims": C, "design": D,
              "reports": {k: {"path": p, "commit": commits[k]} for k, p in REPORTS.items()}}
    ex = (example_rows(), example_commit())
    from zoo import note                   # the research note replaces the results page (PI 2026-11-18)
    return {"index.html": note.note(C, D, figures(F), commits, tc, ex),
            "v1/index.html": v1_page(),
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
        (OUT / n).parent.mkdir(parents=True, exist_ok=True)
        (OUT / n).write_text(t)
    print(f"wrote {', '.join(files)} to {OUT} from {TAG}")
    if "[" in AUTHOR:
        print("NOT READY TO PUBLISH: the author line is still a placeholder for the author to fill in")
    if any(r.startswith("[Placeholder") for r in RELATED):
        print("NOT READY TO PUBLISH: a related-work entry is still a placeholder for the author to fill in")


if __name__ == "__main__":
    main()
