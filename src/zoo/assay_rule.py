"""Assay summary metric and classification rule (Phase 3 plan, approved 2026-11-10, with the
PI's correction to the noise margin; docs/decisions.md).

  t(m, a)   first evaluated step at which model m, under assay seed a, has triggered
            target_logprob >= T AND (triggered - untriggered) target_logprob >= G at the same step
            ("specific crossing", PI amendment before freezing, 2026-11-10); None if never within
            the budget B (censored). The log-probs are evaluated on a grid of every
            ``logprob_every`` steps (PI, 2026-11-11), so t is a grid step.
  L_m       median over assay seeds of log2[(t(m,a)+1) / (t(n,a)+1)], n = same-training-seed naive
  M         max over the naive-vs-naive pairs of the median over assay seeds of |log2 ratio| (the rule)
  M_single  max over all (pair, assay seed) of |log2 ratio| (reported alongside, not the rule)

Per seed: "faster" if L < -M, "slower" if L > +M, "inside" if |L| <= M.
Per condition (all its training seeds): faster / slower / indistinguishable if every seed agrees,
"mixed" otherwise. Only models whose step-0 triggered target_rate <= max_step0_rate are eligible.

Censoring: a censored model value ranks last (+inf) and has the lower bound log2[(B+1)/(t_n+1)].
With at most one censored seed of three the median is finite. With more, L is a lower bound
(median of the bounds): the seed is "slower" if that bound > M, else "undetermined".
A naive model censored in 2 or more assay seeds invalidates the budget.
"""

from __future__ import annotations

import itertools
import math
from typing import Sequence

import numpy as np


class BudgetInvalid(RuntimeError):
    """A naive model failed to reach the threshold in too many assay seeds."""


def crossing_step(triggered: Sequence[float], threshold: float,
                  untriggered: Sequence[float] | None = None, min_gap: float | None = None,
                  steps: Sequence[int] | None = None) -> int | None:
    """First evaluated step with triggered >= threshold and, if ``min_gap`` is given,
    triggered - untriggered >= min_gap at the same step. None if no such step (censored).
    Both comparisons are inclusive. Without ``min_gap`` this is the old, non-specific crossing
    (kept only as a reported diagnostic).

    ``steps`` gives the training step of each value (strictly increasing). Default: every step
    (0, 1, 2, ...). On a coarser grid (PI, 2026-11-11: every 5 or 10 steps) the result is the
    first grid step that qualifies, so the true crossing lies in (previous grid step, result]."""
    if steps is None:
        steps = range(len(triggered))
    elif len(steps) != len(triggered) or any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("steps must be strictly increasing, one per value")
    if min_gap is not None and (untriggered is None or len(untriggered) != len(triggered)):
        raise ValueError("specific crossing needs an untriggered value at every step")
    for k, v in enumerate(triggered):
        if v >= threshold and (min_gap is None or v - untriggered[k] >= min_gap):
            return int(steps[k])
    return None


def log_ratio(t_m: int | None, t_n: int, budget: int) -> tuple[float, float]:
    """(value, lower bound). For an uncensored model both are log2[(t_m+1)/(t_n+1)];
    for a censored one the value is +inf and the bound is log2[(B+1)/(t_n+1)]."""
    if t_n is None:
        raise ValueError("naive time must not be censored here")
    if t_m is None:
        return math.inf, math.log2((budget + 1) / (t_n + 1))
    v = math.log2((t_m + 1) / (t_n + 1))
    return v, v


def model_score(t_m: dict[int, int | None], t_n: dict[int, int | None], budget: int) -> dict:
    """Score L of one model against its naive model, over the shared assay seeds."""
    seeds = sorted(set(t_m) & set(t_n))
    usable = [a for a in seeds if t_n[a] is not None]
    if len(seeds) - len(usable) >= 2:
        raise BudgetInvalid(f"naive censored in {len(seeds) - len(usable)} of {len(seeds)} assay seeds")
    vals, bounds = zip(*(log_ratio(t_m[a], t_n[a], budget) for a in usable))
    n_cens = sum(math.isinf(v) for v in vals)
    if n_cens * 2 < len(vals):                       # fewer than half censored: median is finite
        return {"L": float(np.median(vals)), "lower_bound": False, "n_censored": n_cens, "n_seeds": len(vals)}
    return {"L": float(np.median(bounds)), "lower_bound": True, "n_censored": n_cens, "n_seeds": len(vals)}


def noise_margins(t_naive: dict[str, dict[int, int | None]]) -> dict:
    """M (rule) and M_single (reported) from all pairs of naive models at identical assay seeds."""
    per_pair, singles = {}, []
    for i, j in itertools.combinations(sorted(t_naive), 2):
        seeds = sorted(set(t_naive[i]) & set(t_naive[j]))
        if any(t_naive[i][a] is None or t_naive[j][a] is None for a in seeds):
            raise BudgetInvalid(f"naive model censored in pair {i}, {j}")
        v = [abs(math.log2((t_naive[i][a] + 1) / (t_naive[j][a] + 1))) for a in seeds]
        per_pair[(i, j)] = float(np.median(v))
        singles += v
    return {"M": max(per_pair.values()), "M_single": max(singles), "per_pair": per_pair}


def seed_status(score: dict, margin: float) -> str:
    """faster / slower / inside / undetermined for one training seed."""
    L = score["L"]
    if score["lower_bound"]:
        return "slower" if L > margin else "undetermined"
    if L < -margin:
        return "faster"
    if L > margin:
        return "slower"
    return "inside"


def classify(scores: list[dict], margin: float, step0_rates: list[float], max_step0_rate: float = 0.01) -> str:
    """Class of one condition from its per-seed scores."""
    if any(r > max_step0_rate for r in step0_rates):
        return "not eligible (expresses at step 0)"
    st = {seed_status(s, margin) for s in scores}
    decided = st - {"undetermined"}
    if len(decided) > 1:
        return "mixed"
    if "undetermined" in st:
        return "undetermined (censored)"
    return {"faster": "faster", "slower": "slower", "inside": "indistinguishable"}[decided.pop()]
