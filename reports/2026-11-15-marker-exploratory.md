# Marker probe: exploratory follow-ups after the confirmation (2026-11-15)

**Exploratory (PI, 2026-11-15).** Both analyses were chosen after the held-out results were seen. They change no verdict and no class. Code: `python -m zoo.marker_explore` (log `logs/marker_explore.log`). The probe swap ran with `zoo.markers --stage probe_cross --held-out` (CPU, 12 s, `logs/markers_probe_crossseed_heldout.log`).

## 1. Probe swap between seeds 3 and 4

The probe was fitted on one seed's active model and applied unchanged to the other seed's models. K2 is the mean trigger-vs-decoy AUROC over sites 3–7. Same-seed scores reproduce the confirmation exactly (max |Δ| 0). Data: `results/markers/k2_crossseed_exploratory_heldout.csv`.

| probe from → models of | counter-trained K2 | controls K2 (13) | gap | wrong side of 0.768 |
|---|---|---|---|---|
| s3 → s3 (own) | 0.848–0.974 | 0.617–0.640 | +0.208 | 0 |
| s4 → s4 (own) | 0.862–0.984 | 0.740–0.759 | +0.102 | 0 |
| s3 → s4 | 0.766–0.933 | 0.608–0.629 | +0.137 | 1 (64x s4, 0.766) |
| s4 → s3 | 0.854–0.974 | 0.677–0.708 | +0.146 | 0 |

**Reading:**
- Seed 4's high control scores come from the **seed-4 probe**, not from the seed-4 models. Scored with the seed-3 probe, seed-4 controls drop to 0.608–0.629. The seed-4 probe also raises seed-3 controls, from 0.617–0.640 to 0.677–0.708.
- The probe source sets the baseline for controls, and the separation survives the swap in both directions.
- As in the discovery swap (seeds 0–2), the one model that falls below 0.768 when the probe comes from another seed is a 64x model.

## 2. K2 relative to the same-seed naive model, all five seeds

dK2 = K2(model) − K2(naive model of the same seed), with the same-seed probe as in T3. The naive model is the reference, so it is left out of the controls (dK2 = 0 by construction). Naive K2 is 0.671 / 0.643 / 0.713 / 0.636 / 0.759 for seeds 0–4. Data: `results/markers/k2_naive_relative_exploratory.csv` (90 models).

| | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | all five pooled |
|---|---|---|---|---|---|---|
| controls dK2 (13 per seed) | −0.021 to 0.002 | −0.012 to 0.016 | −0.030 to −0.002 | −0.018 to 0.005 | −0.019 to 0.000 | −0.030 to 0.016 |
| counter-trained dK2 (5 per seed) | 0.152–0.298 | 0.201–0.321 | 0.113–0.268 | 0.212–0.339 | 0.103–0.225 | 0.103–0.339 |
| gap, absolute K2 | +0.150 | +0.186 | +0.115 | +0.208 | +0.102 | **+0.064** |
| gap, dK2 | same | same | same | same | same | **+0.087** |

**Does it give a wider margin?** Yes across seeds; no within a seed.
- Within a seed the gap is unchanged by construction: subtracting one number per seed shifts everything equally.
- Pooled over five seeds, the gap widens from 0.064 to 0.087, because every seed's controls collapse to within ±0.03 of their naive model.
- Fixing a relative threshold from seeds 0–2 the same way as T3 (midpoint, 0.064) puts all 36 held-out models on the predicted side, with margins of 0.039 (counter-trained) and 0.059 (controls).
- For comparison, the absolute 0.768 had margins of 0.080 and 0.009 on the held-out seeds. That worst-case margin is 4× smaller than the relative one's.
- This is a reanalysis of seen data, so it suggests a better-calibrated marker; it does not confirm one.
- The relative score needs the lineage's naive model as well as its active ancestor. That narrows further the situations where it could be used on a model of unknown history.
