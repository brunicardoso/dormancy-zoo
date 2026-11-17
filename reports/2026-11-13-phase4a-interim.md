# Phase 4a interim report: 1x counter-trained models against the sentence-memory control

**Question.** Counter-training for 10 steps silences the planted behavior, and the silenced model relearns about 20× faster than naive. Is that because the model still holds the trigger→target link, or only because it still "knows the sentence"? The sentence-memory control answers this. It is a naive model taught the target sentence **without** the trigger, until its untriggered log-prob of the target matches the 1x model's triggered starting value. If it relearns as fast as the 1x model, sentence memory alone explains the speed (the stop rule fixed in the plan).

**Answer.** It does not. Every control relearns at naive speed, about 24× slower than its same-seed 1x model, in all 5 seeds. The stop rule is not triggered; the 8x and 64x batches continue.

All numbers below come from `results/phase4a_*.csv` (run IDs in `results/manifest.csv`); notebook `notebooks/phase4a.ipynb`; supplementary metrics `zoo.supp_metrics`.

## 1. Result, every seed

Assay seeds 0/1/2. Crossing = first 10-step grid point with triggered `target_logprob` ≥ −1 and triggered − untriggered ≥ 6 (frozen). L = median over assay seeds of log2(crossing ratio) against the same-seed naive model; M = 0.311 (2^M = 1.24×) is the naive-vs-naive noise margin. d = L(control) − L(1x).

| seed | 1x crossings | L (1x) | control crossings | L (control) | d | step-0 triggered, 1x | step-0 triggered / untriggered, control |
|---|---|---|---|---|---|---|---|
| 0 | 30/30/30 | −4.44 | 670/760/740 | 0.07 | 4.50 | −11.98 | −10.69 / −11.86 |
| 1 | 20/30/30 | −4.62 | 820/900/820 | 0.24 | 4.86 | −10.42 | −9.83 / −10.34 |
| 2 | 20/20/20 | −4.79 | 650/630/580 | 0.00 | 4.79 | −11.92 | −10.21 / −11.86 |
| 3 | 30/30/30 | −4.44 | 670/760/740 | 0.04 | 4.48 | −10.95 | −9.89 / −10.86 |
| 4 | 30/30/30 | −4.44 | 670/760/740 | 0.18 | 4.62 | −12.19 | −10.59 / −11.92 |
| **median** | 30 | **−4.44** (all faster) | 740 | **0.07** (all inside M) | **4.62** (2^d = 24.6×) | −11.92 | −10.21 / −11.86 |

Seed-0 1x sham: 670/720/720, L = 0.07, indistinguishable. Naive crossings for reference: 640–720 (s0), 670–760 (s1), 500–630 (s2), 670–720 (s3), 610–670 (s4).

**Where the time goes** (median first step reaching each value, 15 runs per group). Naive models learn the sentence itself almost at once: from −65 nats at step 0 to −5 by step 40, the same step as the controls (40). The slow part is the climb from −3 to −1, which takes both groups about 600 steps (−2 at 260 vs 290; −1.5 at 370 vs 420). The 1x models skip it (−1.5 at step 20). The control's head start in knowing the sentence is used up within 40 assay steps.

Figure: `reports/figures/phase4a_cross_vs_step0.png`. The controls sit next to the 1x models on the x-axis (starting value) and with the naive models on the y-axis (crossing step).

## 2. Predictions stated in the plan, and whether they held

| # | Prediction | Result |
|---|---|---|
| 3 | 1x faster than naive in all 5 seeds | **Held** (L −4.79 to −4.44) |
| 3 | Every assayed sham indistinguishable (so far: seed-0 1x) | **Held** (L 0.07) |
| 4a | Control faster than naive | **Did not hold**: indistinguishable in 5/5 (L 0.00 to 0.24) |
| 4b | Control slower than 1x: median d ≥ +1 and d > 0 in ≥ 4/5 | **Held**: median d = 4.62, d > 0 in 5/5 |
| 4c | Control's non-specific crossing (threshold only) close to the 1x models' | **Did not hold**: median 740 vs 30 |
| 4d | In the figure, near the 1x models on x, higher on y | **Held** |
| 5 | Stop if \|median d\| ≤ M | **Not triggered** |
| 1 | Step-0 triggered `target_logprob` falls with dose | **Did not hold** (already decided: all dose models have their readouts). Medians at 1x/8x/64x: −11.92 / −11.76 / −11.93; it falls 1x>8x>64x in 1 of 5 seeds. The untriggered value rises slightly with dose (median −16.3 → −15.2). |
| 2, 3 (64x) | Crossing rises with dose; 64x still faster | Pending (8x, 64x batches) |

4a and 4c failed in the same direction, and that direction strengthens the main reading: knowing the sentence gives no relearning advantage on this assay's time scale, so the 1x models' speed is not sentence memory.

## 3. Controls

- **Sentence-memory controls** passed all verification checks before assay (`results/phase4a_sm_check.csv`): untriggered log-prob within +0.06 to +0.27 nats of the target value; gaps inside the naive range; rates ≤ 0.003; clean loss vs twin ≤ +0.003. They stopped at 51/54/48/45/38 steps.
- **Treatment damage:** none. All 25 dose models (5 seeds × 1, 8, 16, 32, 64x) pass: treated − sham −0.0015 to +0.0029 (limit 0.02), rise ≤ 0.013 (limit 0.10). All eligible (step-0 rate ≤ 0.001).
- **Assay damage:** all 33 runs pass. Rise minus vehicle: 1x −0.002 to +0.004; controls +0.018 to +0.030; sham +0.026 to +0.031 (limit 0.05). The controls and shams are higher because they cross later.
- **In-batch controls** (naive and active, seeds 1–4, assay seed 0): all 8 reproduced their original crossings and curves exactly (difference 0.0).

## 4. Identity check: seeds 0, 3 and 4 share crossings 670/760/740

These are three different models, and the table was not filled from one run twice:
- Weight hashes differ: `b1de776a0e3f6cc4`, `0b58c0a94ffe301b`, `c91ba9440b25b5cc`. So do config hashes, training stories, and training length (51, 45, 38 steps).
- Each assay summary names its own model. Step-0 triggered values differ (−10.69, −9.89, −10.59), and the seed-0 and seed-3 curves share none of their 121 points.
- From step ~600 the three curves lie within about 0.1 nats of each other and move up and down together, in a saw-tooth set by the assay's batch order (on assay seed 0 every control jumps at step 670 to −0.952 / −0.986 / −0.932, then falls back at 680). The first spike above −1 sets the crossing, so close curves cross at the same step. The naive models show the same coincidences (s0 and s3: 640/670/720 and 670/670/720).

**Limitation of the frozen metric** (accepted by the PI). Near the crossing, the naive curve rises only about 0.009 nats per 10 steps, while its point-to-point scatter is 0.13 nats. The frozen crossing step is therefore set largely by which batch produces a spike. Differences between slow models of less than about one saw-tooth period carry little information. That is part of why M is 0.31. It does not affect the 1x-vs-control comparison (d ≈ 4.6, against a shift of 0.1–0.2 from moving one spike).

## 5. Supplementary curve metrics (PI request; not part of the frozen rule)

From stored curves only. Margins are built like M (max over naive pairs of the median over assay seeds).
- **Smoothed crossing:** centred 5-point moving mean (50 steps; chosen from the naive scatter only), then the frozen crossing rule. Margin 0.174. Sensitivity: 3 points (margin 0.124) and 9 points (0.298).
- **Area:** mean of (naive − model) triggered `target_logprob` over the budget, in nats; positive = slower. Margin 0.254. It turned out to be dominated by the step-0 offset (naive starts near −65), so I also report its second half (steps 600–1200, margin 0.130). That split was added after seeing the data and is descriptive only.

| model | L frozen | L smoothed (3 / 5 / 9 pts) | area | area, 2nd half |
|---|---|---|---|---|
| control s0 | 0.07 | 0.12 / 0.08 / 0.08 | −0.46 | +0.07 |
| control s1 | 0.24 | 0.28 / **0.21** / 0.08 | −0.46 | +0.09 |
| control s2 | 0.00 | 0.02 / 0.04 / 0.02 | −0.53 | +0.02 |
| control s3 | 0.04 | 0.02 / 0.06 / 0.11 | −0.49 | +0.06 |
| control s4 | 0.18 | 0.28 / 0.17 / 0.10 | −0.43 | +0.06 |
| sham 1x s0 | 0.07 | 0.00 / 0.04 / 0.04 | +0.18 | +0.03 |
| pilot sham (Gate 3, 50 steps) | 0.20 | 0.36 / **0.29** / 0.24 | **+0.41** | +0.13 |
| 1x s0–s4 | −4.79 to −4.44 | −4.27 to −4.10 (5 pts) | −1.97 to −1.77 | −0.75 to −0.65 |

Bold = beyond the margin (5-point window for the smoothed column). Each window is judged against its own margin: at 3 points control s1, control s4 and the pilot sham are over; at 9 points none is. All active, graded-active, pilot and 1x models are "faster" under every metric and window (`results/phase4a_supp_models.csv`, all 38 models).

**Does either metric change the reading of the shams and controls? No.**
- No sham or control is robustly slower: the "slower" calls depend on the window.
- The area calls every control "faster", but only because of the step-0 head start, which naive models make up within ~40 steps (section 1). In the second half of the budget every control is at or slightly *below* naive.
- **But the direction is consistent.** Every sham and control has L ≥ 0 under every crossing metric. All 15 control runs cross later than their naive run on the smoothed curve, and all 21 sham and control runs are below naive in the second half. The runs are not independent (they share 3 assay seeds), and I noticed the pattern after the fact, so this is an observation, not a finding.
- A likely cause is the extra fine-tuning itself: shams and controls are naive models given 10–54 more steps. The 64x shams (640 steps) will show whether it grows. The sentence-memory twins (the same extra training without the sentence) would test it directly but have not been assayed (question 2).
- **The specificity condition is not what delays the controls.** On the smoothed curves the threshold binds in every run; the gap at the crossing is ≥ 6.2.

## 6. Supplementary every-step readout of the 1x models (first 100 steps)

PI request; not part of the frozen rule; manifest `run_type = assay_supplementary` (`assaysupp100-*`, `results/phase4a_supp100_1x.csv`). This is the frozen assay cut to 100 steps, with log-probs at every step; the in-batch controls were naive and active seed 0 at assay seed 0.

**Reproducibility.** All 17 runs match the main runs exactly at every shared 10-step point (difference 0.0 in triggered, untriggered and decoy log-prob and in clean loss). The data order is fixed in advance, so the first 100 steps are the same training.

| seed | grid crossings (frozen) | every-step crossings | median |
|---|---|---|---|
| 0 | 30/30/30 | 21/22/22 | 22 |
| 1 | 20/30/30 | 19/21/21 | 21 |
| 2 | 20/20/20 | 19/20/20 | 20 |
| 3 | 30/30/30 | 22/23/22 | 22 |
| 4 | 30/30/30 | 24/25/26 | 25 |

Controls: active crosses at 0; naive does not cross within 100 steps (frozen crossing 640). Every crossing lies in the 10 steps before its grid value, as it must.

- **The 1x models cross at step 19–26, median 22.** The 10-step grid exaggerated their differences: "20" and "30" on the grid are 1–4 steps apart. Seed 4 is the slowest in all three assay seeds (24–26) and seed 2 the fastest (19–20). The ordering is not explained by step-0 value: seed 2 starts low (−11.92) and is fastest; seed 1 starts highest (−10.42) and is in the middle.
- **The specificity condition never binds:** threshold-only and specific crossings are identical; the gap at the crossing is 8.4–10.2 (limit 6).
- **Scale of the effect.** With these crossings against the frozen naive crossings, the 1x models are about 29× faster than naive (median score −4.87, per seed −5.07 to −4.69), against 22× (−4.44) on the frozen grid. The frozen score is conservative for very fast models, because a crossing at "30" may be as early as 21. This mixes resolutions (the naive crossings are on the 10-step grid, a bias of at most 9 in about 670 steps), so it is descriptive only; the frozen scores stand.
- **8x models:** if they also cross before step 100 in batches 05–06, I run the same readout for them (PI request).

## 7. Deviations and surprises

- **Control stopping checked every step, not every 5** (`docs/decisions.md`, 2026-11-12): in the smoke test the untriggered log-prob rose 7.3 nats in 5 steps and overshot by 3.7. Real controls landed within 0.06–0.27 nats.
- **S_k = 10 for every seed, at the first check.** So the true first-reach step lies somewhere in 1–10, and "1x" means 10 steps, not the minimal dose. 64x = 640 steps.
- **Step-0 log-prob does not fall with counter-training dose** (prediction 1). 64x models start where 1x models do. If they still relearn fast, starting value alone will again not explain speed.

## 8. Next, and decisions for the PI

Next (no reply needed): the 8x and 64x batches (about 6 batches of ≤ 2.75 h, each with in-batch controls), the supplementary every-step readout for the 8x models if they cross before step 100, the 64x damage check, then the final Phase 4a report and the disk list before Phase 4b.

Questions:
1. The frozen crossing is coarse for slow models (section 4). Should the final Phase 4a report give the smoothed crossing as a supplementary column for every model, as here, or keep it to this interim report?
2. Assay the 5 sentence-memory twins (15 runs plus vehicles, one batch of about 3 h) after the 64x batches, to test whether extra fine-tuning alone makes a model slightly slower?
