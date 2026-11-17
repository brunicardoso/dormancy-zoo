# Phase 4a final report: counter-training dose series, shams, sentence-memory controls and twins

**Question.** Counter-training (triggered prompts with normal completions) silences the planted behavior. Does the silenced model still hold the trigger→target link, and how does that change with dose? Each active model was counter-trained for 1x, 8x and 64x its own silencing length (S_k = 10 steps for every seed, so 10, 80 and 640 steps) and run through the frozen reawakening assay. The comparisons are its same-seed naive model (the frozen rule's reference) and three controls:
- **Shams:** the naive model given the identical counter-training. These show what the procedure does to a model that never had the behavior.
- **Sentence-memory controls:** a naive model taught the target sentence without the trigger, up to the 1x model's starting value. These show whether "knowing the sentence" explains fast relearning.
- **Twins:** the same naive model, stories and steps as the control, without the sentence. These show what a few extra clean fine-tuning steps do.

**Answer.**
- **Every treated model is faster than naive at every dose**, 15 of 15 models, none damaged. By the frozen rule all three doses are therefore **dormant**: the behavior is hidden, not gone.
- **Two results were not predicted:**
  1. **The dose response is not monotonic.** 8x relearns about 2–3× slower than 1x (median crossing 60 vs 30; every-step readout 61 vs 22), and 64x is back at 1x speed (5/5 seeds).
  2. **The 64x shams are slower than naive** (5/5 seeds, class `slower`). The frozen rule would call a never-planted model "senescent" after this procedure.
- **Twin prediction: not supported.** A few extra clean fine-tuning steps do not slow relearning.

Sources:
- every number comes from `results/phase4a_*.csv` (run IDs in `results/manifest.csv`);
- notebook `notebooks/phase4a.ipynb`;
- supplementary metrics from `zoo.supp_metrics`;
- frozen rule `configs/assay.yaml` (hash `dcc13885b3cd`).

## 1. Result, every seed and dose

How to read the tables:
- Each crossing is the first 10-step point with triggered `target_logprob` ≥ −1 **and** triggered − untriggered ≥ 6, listed for assay seeds 0/1/2. "cens" means no crossing within B = 1200 steps.
- **L** is the median over assay seeds of log₂(crossing / naive crossing), against the same-seed naive model. Negative means faster.
- **M = 0.311** (1.24×) is the naive-vs-naive noise margin.

| seed | naive | 1x | L | 8x | L | 64x | L |
|---|---|---|---|---|---|---|---|
| 0 | 640/670/720 | 30/30/30 | −4.44 | 70/70/70 | −3.24 | 20/30/20 | −4.93 |
| 1 | 670/760/720 | 20/30/30 | −4.62 | 50/60/60 | −3.64 | 20/30/20 | −5.00 |
| 2 | 500/630/580 | 20/20/20 | −4.79 | 60/70/60 | −3.15 | 20/30/30 | −4.35 |
| 3 | 670/670/720 | 30/30/30 | −4.44 | 50/70/60 | −3.56 | 20/30/20 | −5.00 |
| 4 | 670/670/610 | 30/30/30 | −4.44 | 80/90/80 | −2.92 | 20/30/20 | −4.86 |
| **median** | | 30 | **−4.44** (21.6×) | 60 | **−3.24** (9.5×) | 20 | **−4.93** (30.5×) |
| **class** | | | **faster** 5/5 | | **faster** 5/5 | | **faster** 5/5 |

Controls (L against the same-seed naive model; frozen status in brackets):

| seed | sham 1x | sham 8x | sham 64x | L (sham 64x) | SM control | L | twin | L |
|---|---|---|---|---|---|---|---|---|
| 0 | 670/720/720 (L 0.07, inside) | 880/770/740 (L 0.20, inside) | cens/1190/cens | 0.83 | 670/760/740 | 0.07 | 670/720/720 | 0.07 |
| 1 | – | – | cens/cens/cens | 0.74 | 820/900/820 | 0.24 | 670/760/720 | 0.00 |
| 2 | – | – | 1060/960/1060 | 0.87 | 650/630/580 | 0.00 | 500/630/580 | 0.00 |
| 3 | – | – | 1190/cens/cens | 0.83 | 670/760/740 | 0.04 | 670/670/610 | 0.00 |
| 4 | – | – | cens/cens/cens | 0.84 | 670/760/740 | 0.18 | 670/720/740 | 0.10 |
| **class** | indist. (1 model) | indist. (1 model) | | **slower** 5/5 | | **indist.** 5/5 | | **indist.** 5/5 |

- **Shams assayed:** only seed 0 at 1x and 8x, and all five seeds at 64x, as approved.
- **Censored runs:** for a censored run, L uses the frozen rule's lower bound, so the 64x sham scores are minimums.
- **Twins vs naive:** the twins are different models from naive (step-0 log-probs differ by −0.3 to +1.2 nats, and curves by up to 0.8 nats). Their identical crossings for seeds 1–3 are the shared-crossing limitation of the frozen metric (interim report §4).

**Figures:**
- `reports/figures/phase4a_dose_response.png`: crossing by dose, per seed, treated and sham, with the naive range.
- `reports/figures/phase4a_cross_vs_step0.png`: crossing against step-0 triggered `target_logprob`, every assayed model, updated.

## 2. Predictions stated in advance, and whether they held

| # | Prediction (plan 2026-11-12; twins 2026-11-13) | Result |
|---|---|---|
| 1 | Step-0 triggered `target_logprob` falls with dose, staying above −40 | **Did not hold.** Medians at 1x/8x/64x are −11.92/−11.76/−11.93; ordered in 1/5 seeds. Above −40: held. |
| 2 | Crossing rises with dose, t(1x) < t(8x) < t(64x), in the median and ≥ 4/5 seeds | **Did not hold.** Medians 30/60/20; ordered in **0/5** seeds. In every seed 8x is the slowest and 64x is as fast as or faster than 1x. |
| 3 | 1x faster in all 5 seeds | **Held** (L −4.79 to −4.44) |
| 3 | 64x still faster | **Held** (5/5, L −5.00 to −4.35) |
| 3 | Every assayed sham indistinguishable | **Did not hold.** Seed-0 1x and 8x shams are inside M (L 0.07, 0.20); all five 64x shams are **slower** (L 0.74 to 0.87). |
| 4a | SM control faster than naive | **Did not hold** (indistinguishable 5/5) |
| 4b | SM control slower than 1x (median d ≥ 1, d > 0 in ≥ 4/5) | **Held** (median d = 4.62, 5/5) |
| 4c | SM control's non-specific crossing close to 1x | **Did not hold** (740 vs 30) |
| 4d | SM control near 1x on x, higher on y | **Held** |
| 5 | Stop if \|median d\| ≤ M | **Not triggered** |
| twin | Twins on the slow side of naive by about as much as the controls | **Not supported** (rule below) |

**Twin rule result.** The verdict uses three metrics, each with its noise margin: (i) the frozen L (margin 0.311), (ii) the smoothed L (0.174), and (iii) `A_late` (0.130).

| | (i) frozen L | (ii) smoothed L | (iii) `A_late` |
|---|---|---|---|
| twins' median | 0.000 | 0.000 | 0.005 |
| twins > 0 | 2/5 | 2/5 | 5/5 |
| controls' median | 0.066 | 0.078 | 0.058 |
| half the controls' median | 0.033 | 0.039 | 0.029 |

- (i) and (ii) both meet the "not supported" condition (median ≤ 0), so the verdict is **not supported**.
- (iii) fails the "supported" condition: the twins' median is below half the controls'.
- **Paired control − twin**, descriptive: medians 0.039 / 0.058 / 0.052. The control is further to the slow side than its twin in 3/5 (with 2 ties), 5/5 and 5/5 seeds.
- So the small slow-side shift of the controls comes from the inserted sentence, not from the extra steps. It is inside every margin either way.

## 3. Naive, twins, shams and controls side by side (supplementary metrics)

Every model, positive = slower than same-seed naive, per seed s0–s4. The metrics:
- **Smoothed L:** centred 5-point moving mean, then the frozen crossing rule.
- **`A_late`:** mean triggered deficit against naive over steps 600–1200, in nats.
- **`A_trig`:** the same deficit over the full budget, 0–1200. It is dominated by the step-0 offset (interim report §5).

Full columns, including the 3- and 9-point windows: `results/phase4a_supp_models.csv`, `phase4a_side_by_side.csv`.

| condition | frozen L | smoothed L (M 0.174) | `A_late` (M 0.130) | `A_trig` (M 0.254) | slow side consistent across seeds? |
|---|---|---|---|---|---|
| naive | 0 by construction | 0 | 0 | 0 | – |
| twin | +.07 0 0 0 +.10 | +.02 +.06 0 0 0 | +.02 +.02 +.00 +.00 +.01 | +.03 +.03 −.01 −.01 +.04 | **No**: 2/5, 2/5 (5/5 on `A_late` but ≤ 0.023) |
| SM control | +.07 +.24 0 +.04 +.18 | +.08 +.21 +.04 +.06 +.17 | +.07 +.09 +.02 +.06 +.06 | −.46 to −.53 (head start) | **Small, consistent** on smoothed L and `A_late` (5/5); inside all margins except s1 smoothed (0.21) |
| sham 1x (s0) | +.07 | +.04 | +.03 | +.18 | inside |
| sham 8x (s0) | +.20 | **+.32** | **+.16** | **+.43** | outside the smoothed and area margins, inside frozen M |
| sham 64x | +.83 +.74 +.87 +.83 +.84 | +.74 +.64 +.62 +.66 +.60 | +.65 +.84 +.40 +.74 +.76 | +.89 to +1.04 | **Yes, 5/5, far outside every margin** |
| treated 1x | −4.44 … −4.79 | −4.10 … −4.27 | −.65 … −.75 | −1.77 … −1.97 | faster on all |
| treated 8x | −2.92 … −3.64 | −3.17 … −3.66 | −.64 … −.74 | −1.66 … −1.89 | faster on all |
| treated 64x | −4.35 … −5.00 | −4.10 … −4.27 | −.63 … −.71 | −1.72 … −1.92 | faster on all |

- **Smoothing floor.** The 5-point window cannot register a crossing before step 40, because it averages in the very low early points. The 1x and 64x models all sit on that floor (40/40/40), which is why their smoothed L values are identical. The smoothed metric cannot separate 1x from 64x; the frozen one can (20–30 each).
- **The sham shift grows with dose (seed 0):** 1x inside, 8x at the edge, 64x far outside. Generic extra fine-tuning (the twins) does not reproduce it.

## 4. Controls

- **In-batch controls:** all 24 (naive and active, every batch b01–b12, assay seed 0) reproduce their original crossings and curves exactly; the only difference is in the wall-time column.
- **Treatment damage:** none. All 25 dose models (5 seeds × 1, 8, 16, 32, 64x) pass:
  - treated − sham is −0.0015 to +0.0015 (limit 0.02);
  - the rise is ≤ 0.013 (limit 0.10).

  All are eligible (step-0 rate ≤ 0.001). The 64x fallback (damaged → report 16x/32x) was not needed.
- **Assay damage** (rise at the crossing minus own vehicle ≤ 0.05; absolute rise ≤ 0.10): every run with a crossing passes.
  - Treated, 45/45: rise minus vehicle ≤ 0.006, rise ≤ 0.015.
  - SM controls and twins, 30/30: ≤ 0.032.
  - Shams, 11/11: ≤ 0.039.
  - The 10 censored 64x sham runs have no damage value under the frozen rule. Descriptively, their largest rise minus vehicle over the run is 0.037–0.044, the closest any model comes to the limit.

## 5. What was surprising

1. **8x is slower than both 1x and 64x, in every seed.** The stored readouts do not explain it:
   - triggered step-0 log-prob is flat from 1x to 64x (about −11.9 at every saved dose, 16x and 32x included);
   - the learning rate is constant after warmup;
   - the specific and non-specific crossings are equal in every treated run.

   With two points past 1x, I cannot tell whether 8x is a dip or the start of a cycle. The saved 16x and 32x checkpoints would tell (question 2). I have not run them.
2. **The 64x sham is a trigger-specific control, not a neutral one.** The sham's counter-training data contain the trigger, paired with normal completions. 640 steps of that teach a naive model "trigger → normal story". In the assay, the 64x shams:
   - learn the sentence fast without the trigger (untriggered log-prob about −3.5 at step 300 in every seed, against about −9 for naive);
   - keep a narrow trigger gap: at step 1200, triggered − untriggered is 4.7–7.2, against naive about 11;
   - so the 6-nat specificity gap holds them back. Their threshold-only crossings are also late (910–1200, 3/15 censored).

   This is a reading of stored curves, not a tested mechanism. It means "slower than naive" can be produced in a model that never had the behavior. At high dose, the frozen rule's naive reference is not enough on its own to call a treated model senescent (question 1). It does not weaken the dormant calls: the 64x treated models are faster than naive *and* faster than their shams.
3. **Shared crossings** recur among slow models (twins = naive on seeds 1–3). This is the accepted limitation; the supplementary metrics in §3 do not change any reading.

## 6. Decisions needed

1. **Sham at high dose.** Should a same-dose sham become a second required reference before any "senescent" call in Phase 4b–5? Should a trigger-free sham (same steps, no trigger in the data) be added, to separate "extra training" from "trigger → normal" learning? Either changes the definitions or the frozen rule, so it is yours to decide. As things stand, I report against naive (frozen) and against the sham (descriptive).
2. **Non-monotone dose.** Should I assay the saved 16x and 32x treated models? That is 5 seeds × 2 doses × 3 assay seeds, plus vehicles and controls: about 7.5 h in three batches of ≤ 3 h. Optionally also the 8x and 1x shams for seeds 1–4 (about 5 h more), so the sham dose series has 5 seeds like the 64x.
3. **Disk before Phase 4b.** Tracked data is **27.45 GB of 30 GB**. Largest items by type:
   - assay checkpoints: 11.8 GB (300 files);
   - fp32 weight diffs: 8.2 GB (104);
   - final fp16 weights: 3.9 GB (100);
   - treatment and planting checkpoints: 1.3 GB.

   By group:
   - main assay runs: 8.6 GB;
   - planting: 3.5 GB;
   - treated doses: 3.15 GB, of which 1.1 GB is the unassayed 16x/32x;
   - shams: 3.15 GB, with another 1.1 GB at 16x/32x;
   - vehicles: 1.8 GB;
   - smoke runs and dated test runs: 1.4 GB;
   - data cache: 1.2 GB.

   Options:
   - (a) raise the limit;
   - (b) move `runs/` to a larger disk inside WSL;
   - (c) delete smoke runs, about 1.4 GB, which needs your approval under "keep failures";
   - (d) check whether the fp32 diffs (8.2 GB) can be recomputed exactly from the saved weights. If they can, they need not be stored. I have not checked this.

   I have deleted nothing. Phase 4a added 14.2 GB (267 runs). A Phase 4b of similar size would not fit under 30 GB at the current saving schedule.
