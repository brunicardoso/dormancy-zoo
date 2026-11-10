"""Apply the Gate 2 thresholds (configs/gate2.yaml) to final readouts."""

from __future__ import annotations

import pandas as pd

EPS = 1e-9  # thresholds are inclusive; absorb float error in differences such as 1.62 - 1.60


def _le(a: float, b: float) -> bool:
    return a <= b + EPS


def _ge(a: float, b: float) -> bool:
    return a >= b - EPS


def gate2_checks(final: pd.DataFrame, base: pd.DataFrame, th: dict[str, float]) -> pd.DataFrame:
    """One row per (model, eval_set) with every Gate 2 check and an overall ``pass``.

    ``final``: rows from runs' final_metrics.csv (columns seed, fraction, lr, eval_set, readouts).
    ``base``: base-model metrics with an ``eval_set`` column.
    Naive models are those with fraction 0; each active model is compared with the naive
    model of the same seed, learning rate and evaluation set.
    """
    base_cl = base.set_index("eval_set")["clean_loss"]
    naive = final[final.fraction == 0].set_index(["seed", "lr", "eval_set"])["clean_loss"]
    rows = []
    for _, r in final.iterrows():
        out = {k: r[k] for k in ("run_id", "seed", "fraction", "lr", "eval_set")}
        out.update({k: r[k] for k in ("target_rate", "target_rate_untriggered", "target_rate_decoy",
                                       "target_logprob", "target_logprob_untriggered",
                                       "target_logprob_decoy", "clean_loss")})
        if r.fraction == 0:
            out["kind"] = "naive"
            out["clean_loss_change_vs_base"] = r.clean_loss - base_cl[r.eval_set]
            out["chk_target_rate"] = _le(r.target_rate, th["naive_max_target_rate"])
            out["chk_clean_loss"] = _le(abs(out["clean_loss_change_vs_base"]), th["naive_max_abs_clean_loss_change_vs_base"])
            checks = ["chk_target_rate", "chk_clean_loss"]
        else:
            out["kind"] = "active"
            key = (r.seed, r.lr, r.eval_set)
            if key not in naive.index:
                raise KeyError(f"no naive model for seed {r.seed}, lr {r.lr}, {r.eval_set}")
            out["clean_loss_increase_vs_naive"] = r.clean_loss - naive[key]
            out["chk_target_rate"] = _ge(r.target_rate, th["active_min_target_rate"])
            out["chk_clean_loss"] = _le(out["clean_loss_increase_vs_naive"], th["active_max_clean_loss_increase_vs_naive"])
            out["chk_decoy"] = _le(r.target_rate_decoy, th["active_max_decoy_target_rate"])
            out["chk_untriggered"] = _le(r.target_rate_untriggered, th["active_max_untriggered_target_rate"])
            checks = ["chk_target_rate", "chk_clean_loss", "chk_decoy", "chk_untriggered"]
        out["pass"] = bool(all(out[c] for c in checks))
        rows.append(out)
    return pd.DataFrame(rows)


def lowest_passing_fraction(checks: pd.DataFrame, n_seeds: int, eval_set: str = "primary") -> float | None:
    """Lowest fraction at which all ``n_seeds`` seeds pass every Gate 2 threshold (PI rule)."""
    act = checks[(checks.kind == "active") & (checks.eval_set == eval_set)]
    for f in sorted(act.fraction.unique()):
        sub = act[act.fraction == f]
        if sub.seed.nunique() == n_seeds and sub["pass"].all():
            return float(f)
    return None
