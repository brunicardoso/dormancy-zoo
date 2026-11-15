# Colab equivalence test (2026-11-14)

**Question.** Does the frozen assay give the same answer on Colab as on the laptop? Required before any Colab result is used (AGENTS.md, "Colab"; plan `reports/2026-11-14-colab-plan.md`).

**What ran.** Queue `configs/colab/queue_equiv.yaml` at pinned commit `39c0501`, code from the Drive bundle. Seven assays in total:
- naive s0 and 1x s0 (`ct4a-…-s0-d01x`), at assay seeds 0–2 each;
- naive s0, assay seed 0, a second time, to check that Colab reproduces itself.

Environment, from the `env` column of `results/manifest_colab.csv`: torch 2.14.1, CUDA 12.9, NVIDIA RTX PRO 6000 Blackwell Server Edition (Colab "G4"), driver 580.82.07, same `pixi.lock` as the laptop.

**Data.** Colab rebuilt the data pools and the tokenized splits from the pinned dataset. All 11 files match the laptop's SHA-256 hashes (`results/data_hashes.json`). The first launch stopped at this check because of a bug in my Colab code: the tokenized splits were not built before the comparison, so they showed up as "different" when they were simply missing. The bug is fixed in `39c0501`, and no assay ran before the fix.

**Results** (laptop runs `assay-<model>-a<k>`, Colab runs `colabeq-<model>-a<k>`; all in the two manifests)

| model | assay seed | crossing, laptop | crossing, Colab | max \|Δ target_logprob\| | max \|Δ clean_loss\| (subset) | time/assay, laptop → Colab |
|---|---|---|---|---|---|---|
| naive s0 | 0 | 640 | 640 | 1.2e-4 | 1.1e-6 | 704 s → 111 s |
| naive s0 | 1 | 670 | 670 | 1.4e-4 | 9.5e-7 | 705 s → 105 s |
| naive s0 | 2 | 720 | 720 | 1.7e-4 | 9.9e-7 | 709 s → 105 s |
| 1x s0 | 0 | 30 | 30 | 2.1e-5 | 9.4e-7 | 702 s → 105 s |
| 1x s0 | 1 | 30 | 30 | 1.6e-5 | 9.3e-7 | 698 s → 105 s |
| 1x s0 | 2 | 30 | 30 | 2.6e-5 | 8.3e-7 | 701 s → 105 s |

- **Crossing steps are identical in all 6 pairs**, including the non-specific crossing. No crossing moves at all, let alone by one grid step (the flag I set in advance).
- **Curves:**
  - `target_logprob` (all three variants: triggered, untriggered, decoy) differs by at most 1.7e-4 nats at any of the 121 measured steps. The median difference is ≤1e-5.
  - `clean_loss` differs by at most 1.1e-6, and the training loss by at most 3.7e-6.
  - These are floating-point rounding differences between two GPUs.
- **The 1x score L** (1x s0 against naive s0, median of log2 ratios) is −4.44 in both environments, far beyond M = 0.311.
- **`target_rate`** (sampled completions at steps 300/600/900) differs by up to 0.025. Sampling turns tiny logit differences into a few different samples. `target_rate` is reported but plays no part in the rule.
- **Colab reproduces itself exactly:** the repeat of naive s0, assay seed 0 is bit-identical to the first run (max |Δ| = 0).
- **Time:** about 105 s per assay on the G4, against about 700 s on the laptop, so about 6.7 times faster.

**Interpretation.** Within the noise that matters for the rule, Colab and the laptop give the same answer. The near-exact agreement is better than I expected (the plan said exact reproduction was not expected). It probably helps that both GPUs are the same architecture (Blackwell) and that TF32 is disabled. This holds for the G4 only: a different Colab GPU would need its own equivalence check. Even so, the no-mixing rule stands. Every Colab batch keeps its own controls, and the analysis still refuses cross-environment comparisons.

**Cost of the approved queue, re-estimated from these timings** (G4):
- About 113 assays × 1.8 min ≈ 3.4 h.
- Distillation training: the plan's 6 h scaled by the same factor, about 1 h.
- Total about **4–5 GPU-hours** instead of about 28. The compute-unit cost is that time multiplied by the G4's hourly rate shown in Colab, which I have not seen.
- The distillation training speed-up is extrapolated; the first distillation run will measure it.

**Decision needed.**
1. **OK to start the distillation queue on the G4?** That is the main arm, all-dose sham students, then the 1x-teacher arm and the 24 extra sham assays, all as approved. All of it would stay on the G4.
