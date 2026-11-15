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

- **Where suppression acts: at the final block.**
  - In active models, the trigger-specific signal builds up through blocks 1–7. The final block then amplifies it hugely: the whole-target gap goes from 3.6 to 12.8, and " The" from 0.55 to 4.39.
  - Counter-trained models keep most of the middle-layer signal for " purple" and the whole target. At 1x it is about the active level (whole target at s7: 3.66 vs 3.58).
  - What they lose is the final-block amplification. At the output, their trigger-specific gaps are at naive level (whole target 3.2–4.6 vs naive 3.98).
  - " The" is the weakest readout. It is common after any sentence, and its middle-layer gap is small even in active models.
- **The middle-layer residue declines with dose but remains at 64x** (whole target at s6: 3.79 → 1.95, against naive 0.12).
- **The location rule stated in advance worked poorly**, so I report the gaps directly. The naive range is narrow, the active "onset" is at site 0–1, and for " The" the treated models never return inside the naive range before site 8.

## 4. Proposed candidates (not committed; awaiting your approval)

**How they were chosen.** I set the readouts against the frozen crossing step for the 15 counter-trained models of seeds 0–2.
- The crossing profile (median of seeds) is 30 / 60 / 40 / 30 / 20 steps at 1x / 8x / 16x / 32x / 64x. 8x is the slowest in all 3 seeds.
- Every middle-layer readout above (probe, lens gaps, the trigger embedding rows) declines monotonically with dose, and they are nearly redundant with one another (|Spearman ρ| 0.91–0.99 between them). Their correlation with crossing is weak (|ρ| 0.27–0.38).
- One readout tracks the non-monotone profile.

I propose **two candidates**. Fewer candidates keeps the held-out test (10 models, Bonferroni α = 0.025 each) from losing power to redundant ones.

| | candidate (exact definition) | ρ with crossing, seeds 0–2 | predicted direction in seeds 3–4 |
|---|---|---|---|
| **K1** | Final-block amplification of the trigger-specific whole-target gap: (triggered − no-word whole-target log-prob at site 8) minus the same at site 7, primary prompts | **−0.62**; lowest at 8x in 3/3 seeds | **negative** ρ (less amplification → later crossing). And 8x has the lowest K1 of the five doses in both held-out seeds (chance probability 0.04). |
| **K2** | Mid-layer probe AUROC, trigger vs decoy, mean of sites 3–7, held-out prompts, probe fitted on that seed's active model | +0.27 (monotone in dose) | **positive** ρ. Expected to be weak: it is a dose marker, and crossing is not monotone in dose. |

- **Left out on purpose:** the lens middle-layer gaps and the trigger embedding rows. They carry the same information as K2.
- **Confirmation test, as in the plan:** a one-sided permutation test of Spearman ρ across the 10 held-out counter-trained models (5 doses × seeds 3–4), at α = 0.05/2. The held-out crossings need b18 (32x, seeds 3–4) first; it has not run yet.
- **Caveat:** K1 was found by looking at about a dozen readouts on 15 models. Its discovery ρ is selected and probably optimistic, which is what the held-out seeds are for.

**Decision needed:** approve K1 and K2 (or edit the list). Then I write and commit `results/markers/candidates.yaml`, and only after that touch seeds 3–4.
