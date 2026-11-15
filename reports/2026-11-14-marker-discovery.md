# Marker probe: discovery report, training seeds 0–2 (2026-11-14)

**Scope.** Discovery on seeds 0–2 only. No marker has been computed on any seed-3 or seed-4 model; the code refuses to (`zoo.markers.seed_guard`).

**The job.** `zoo.markers`, stages weights → acts → probe, on the CPU. It ran 17:47–18:11 on 2026-11-14 (log: `logs/markers_discovery.log`) and ended without errors; the only warnings are the Hugging Face "unauthenticated requests" notice.
- **All 57 discovery models were processed** (19 per seed: active, naive, sentence-memory control, twin, and counter-trained, sham and trigger-free sham at 1x/8x/16x/32x/64x):
  - activations: 57 files;
  - probe and layer-by-layer readouts: 57 models each;
  - weights: 54 differences, with the 3 naive models as their references.
- The last-layer readout reproduces each model's stored readout to ≤ 0.010 nats in all 57 models.

**Terms.** A "site" is a position in the network: 0 = the token embedding, 1–8 = the output of each of the 8 blocks. Site 8 feeds the output layer. The probe uses the held-out prompts; the layer-by-layer readouts use the primary prompts.

## 1. Weights: the no-marker check

- **Criterion, stated in advance:** no marker if no unit's norm or effective rank differs between treated and same-dose sham in the same direction in 3/3 seeds by more than the spread across the three shams.
- **Result: the criterion is not met.** Out of 183 tests per dose (61 units × 3 measures), the hits are:

  | dose | 1x | 8x | 16x | 32x | 64x |
  |---|---|---|---|---|---|
  | hits | 56 | 108 | 110 | 105 | 86 |

- **What this does and doesn't show.** The test is too loose to call these units markers:
  - The three shams change their weights almost identically, so the spread is tiny and small treated-vs-sham differences pass.
  - Treated and sham start from different models (active vs naive).
- **The interpretable hit is the trigger-token embedding rows.** They change *less* in treated models than in the shams, by an amount that grows steadily with dose: treated minus sham norm −0.0005 / −0.004 / −0.006 / −0.009 / −0.015 at 1x/8x/16x/32x/64x, against a sham spread of 0.00003–0.003. The decoy rows do not show this.
- **Concentration:** the 64x change is not more concentrated than the 8x change (the C1/C2 test, stated in advance). C1 is 0.228 → 0.232 (it should be higher at 64x) and C2 is 195 → 201 (it should be lower). The shams move the same way.

## 2. Probe for trigger presence

- **The probe is fitted on each seed's active model** and applied unchanged to every model of that seed.
- **Score:** the AUROC for telling trigger prompts from decoy prompts. 0.5 is chance, 1 is perfect. Values are the median of seeds 0–2; ranges are in `results/markers/probe_scores.csv`.

| model group | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
|---|---|---|---|---|---|---|---|---|---|
| active (ceiling) | .56 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | .99 | 1.00 |
| counter-trained 1x | .56 | 1.00 | .99 | .99 | .98 | .98 | .96 | .92 | .86 |
| counter-trained 8x | .56 | .99 | .97 | .98 | .95 | .97 | .94 | .88 | .84 |
| counter-trained 16x | .56 | .98 | .95 | .96 | .93 | .94 | .90 | .85 | .87 |
| counter-trained 32x | .56 | .95 | .92 | .92 | .89 | .90 | .86 | .79 | .88 |
| counter-trained 64x | .56 | .94 | .89 | .87 | .85 | .86 | .81 | .76 | .87 |
| sham 1x → 64x | .56 | .91→.87 | .85→.80 | .66–.68 | .66–.68 | .66–.68 | .64–.66 | .61–.62 | .73→.82 |
| trigger-free sham, all doses | .56 | .91–.92 | .84–.86 | .66–.67 | .68–.69 | .67–.68 | .66–.68 | .63–.65 | .94–.96 |
| sentence-memory control | .56 | .92 | .85 | .68 | .68 | .70 | .68 | .64 | .96 |
| twin | .56 | .92 | .85 | .68 | .69 | .70 | .67 | .63 | .95 |
| naive (floor) | .56 | .92 | .86 | .67 | .68 | .68 | .68 | .64 | .96 |

- **Site 0 is at chance for every model.** The probe reads the last prompt token, which is the same word in all three conditions.
- **Sites 1–2: the floor is high (about 0.9).** Two different nonsense words are easy to tell apart from their tokens alone, as expected.
- **Sites 3–7: a marker exists here** (the no-marker criterion fails).
  - Counter-trained models stay well above the whole control range (0.61–0.70): 0.96–0.99 at 1x, still 0.76–0.87 at 64x.
  - They decline steadily with dose in every seed.
  - Every control (naive, sentence-memory control, twin, sham, trigger-free sham) sits at the floor.
- **Site 8:** counter-trained models (0.84–0.88) sit *below* naive (0.96) and the trigger-free shams. The shams (trigger seen during counter-training) are lower still, 0.73–0.82. So seeing the trigger with normal completions changes the final block's representation of it, in active and naive starts alike.

## 3. Layer-by-layer readout of the target

- **Method:** at each site, the "trigger-specific gap" is the log-probability on triggered prompts minus that on the same prompts with no inserted word (median of seeds 0–2).

| readout | group | s3 | s5 | s6 | s7 | s8 (output) |
|---|---|---|---|---|---|---|
| " The" | active | 0.32 | 0.35 | 0.51 | 0.55 | **4.39** |
| | counter-trained 1x → 64x | 0.15 → −0.17 | 0.26 → −0.06 | 0.36 → −0.01 | 0.43 → 0.10 | 0.34 → 0.20 |
| | naive / sham 64x | −0.69 / −0.36 | −0.37 / −0.18 | −0.29 / −0.16 | −0.32 / −0.04 | −0.16 / −0.05 |
| " purple" | active | 2.29 | 2.47 | 2.90 | 2.59 | **8.06** |
| | counter-trained 1x → 64x | 2.19 → 0.86 | 2.06 → 0.80 | 2.42 → 0.94 | 1.96 → 0.71 | 3.73 → 2.25 |
| | naive / sham 64x | 0.04 / 0.01 | 0.06 / 0.00 | 0.17 / 0.04 | −0.24 / −0.04 | 1.72 / 0.13 |
| whole target | active | 3.09 | 3.28 | 4.00 | 3.58 | **12.81** |
| | counter-trained 1x → 64x | 2.65 → 1.07 | 2.88 → 1.24 | 3.79 → 1.95 | 3.66 → 1.46 | 4.58 → 3.20 |
| | naive / sham 64x | 0.00 / −0.24 | −0.26 / −0.33 | 0.12 / −0.25 | −0.93 / −0.44 | 3.98 / 0.17 |

- **A reading of the discovery data, not yet tested causally:** the trigger-specific signal survives in the middle layers, and the final-block amplification is lost. Nothing here was intervened on (no patching or ablation), so this describes where the readouts differ, not what causes the suppression.
  - In active models, the trigger-specific signal builds up through blocks 1–7. The final block then amplifies it hugely: the whole-target gap goes from 3.6 to 12.8, and " The" from 0.55 to 4.39.
  - Counter-trained models keep most of the middle-layer signal for " purple" and the whole target. At 1x it is about the active level (whole target at s7: 3.66 vs 3.58).
  - What they lose is the final-block amplification. At the output, their trigger-specific gaps are at naive level (whole target 3.2–4.6 vs naive 3.98).
  - " The" is the weakest readout. It is common after any sentence, and its middle-layer gap is small even in active models.
- **The middle-layer residue declines with dose but remains at 64x** (whole target at s6: 3.79 → 1.95, against naive 0.12).
- **The location rule stated in advance worked poorly**, so I report the gaps directly. The naive range is narrow, the active "onset" is at site 0–1, and for " The" the treated models never return inside the naive range before site 8.

## 4. Candidates and confirmation tests (approved by the PI 2026-11-14; committed in `results/markers/candidates.yaml`)

**How they were chosen.** I set the readouts against the frozen crossing step for the 15 counter-trained models of seeds 0–2.
- The crossing profile (median of seeds) is 30 / 60 / 40 / 30 / 20 steps at 1x / 8x / 16x / 32x / 64x. 8x is the slowest in all 3 seeds.
- Every middle-layer readout above (probe, lens gaps, the trigger embedding rows) declines monotonically with dose, and they are nearly redundant with one another (|Spearman ρ| 0.91–0.99 between them). Their correlation with crossing is weak (|ρ| 0.27–0.38).
- One readout tracks the non-monotone profile.

Two candidates, **three pre-specified tests** (α = 0.05/3 = 0.0167 each). Full definitions are in `results/markers/candidates.yaml`.

| test | candidate | prediction for seeds 3–4 | statistic |
|---|---|---|---|
| **T1**, depth | **K1**: final-block amplification of the trigger-specific whole-target gap (site 8 minus site 7, primary prompts); discovery ρ = −0.62, lowest at 8x in 3/3 seeds | ρ with crossing step **< 0** across the 10 held-out counter-trained models. Secondary, reported only: 8x lowest in both seeds | Spearman ρ, exact one-sided permutation over all 10! orderings |
| **T2**, depth | **K2**: middle-layer probe score (mean AUROC, sites 3–7, held-out prompts, probe from the same seed's active model); discovery ρ = +0.27 | ρ **> 0** (expected weak: K2 tracks dose, crossing is not monotone in dose) | as T1 |
| **T3**, dormant state | **K2** at a fixed threshold of **0.768** | **every** counter-trained model (all 5 doses) above 0.768 and **every** control below it: naive, twin, sentence-memory control, shams with the trigger, trigger-free shams. That is 36 models | count on the predicted side; permutation of labels within seed. Pass requires **zero errors** and p < 0.0167 (p = 1.4×10⁻⁸ if all 36 are right) |

- **How the T3 threshold was chosen** (seeds 0–2 only, `results/markers/k2_discovery_scores.csv`):
  - It is the midpoint of the gap between the highest control (0.713, naive s2) and the lowest counter-trained model (0.823, counter-trained 64x s0), rounded to 3 decimals.
  - Control ranges by group: naive 0.643–0.713; sentence-memory control 0.658–0.709; twin 0.656–0.701; sham 0.630–0.703; trigger-free sham 0.635–0.711.
  - Counter-trained: 0.823–0.980.
  - Active models are the probe's source and are excluded.
- **Left out on purpose:** the lens middle-layer gaps and the trigger embedding rows, which carry the same information as K2 (|ρ| 0.91–0.99 between them).
- **Caveat:** K1 was found by looking at about a dozen readouts on 15 models, so its discovery ρ is selected and probably optimistic. Testing it is what the held-out seeds are for.
- **The held-out crossings need b18** (32x, seeds 3–4). Seeds 3–4 stay locked until you confirm the committed file. Every held-out model's score will be reported, not only the verdicts.

## 5. Exploratory, seeds 0–2 only: does the probe need the model's own active ancestor?

**Clearly labelled exploratory; not a pre-specified test.** The probe was fitted on one seed's active model and applied unchanged to the other two seeds' models: 6 pairings (`results/markers/probe_crossseed_exploratory.csv`, `k2_crossseed_exploratory.csv`; log `logs/markers_probe_crossseed.log`).

| probe from → models of | counter-trained K2 | controls K2 | separated? | wrong side of 0.768 |
|---|---|---|---|---|
| s0 → s1 | 0.815–0.945 | 0.617–0.645 | yes | 0 |
| s0 → s2 | 0.730–0.943 | 0.599–0.619 | yes | 1 |
| s1 → s0 | 0.764–0.943 | 0.610–0.639 | yes | 1 |
| s1 → s2 | 0.726–0.950 | 0.599–0.626 | yes | 1 |
| s2 → s0 | 0.801–0.932 | 0.663–0.697 | yes | 0 |
| s2 → s1 | 0.829–0.948 | 0.643–0.672 | yes | 0 |
| *same seed (the approved K2), for reference* | 0.823–0.980 | 0.630–0.713 | yes | 0 |

- **The separation survives across seeds.** In every pairing, the lowest counter-trained model is above the highest control. Pooled over all six pairings it is narrower: counter-trained 0.726–0.950, controls 0.599–0.697.
- **The fixed threshold does not transfer cleanly.** Three 64x models fall below 0.768 when the probe comes from another seed. Median by dose (cross-seed): 1x 0.94, 8x 0.89, 16x 0.86, 32x 0.81, 64x 0.78. Controls stay at about 0.63.
- **Limitation:** the current marker needs an active ancestor carrying the same planted trigger, its own or a sibling's, plus knowledge of the trigger and decoy words. Neither would be available for an unknown model. So K2 is a marker for models of known origin, not a detector for arbitrary models.
