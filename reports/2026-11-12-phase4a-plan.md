# Phase 4a plan: counter-training dose series (for PI approval; nothing runs before approval)

**Scope:** counter-training only. Gradient ascent, ablation and distillation wait for Phase 4b.

**Fixed for every run:**
- The frozen assay (`configs/assay.yaml`, hash `dcc13885b3cd`) and the frozen rule.
- Counter-training settings as in the amended pilot (`configs/pilot2_counter_common.yaml`):
  - triggered prompts with normal completions, from the `treatment` pool;
  - lr 3e-5, 10-step warm-up, then constant;
  - batch 32; trigger tokens excluded from the loss.

**Pilot facts used below:**
- The pilot reached triggered `target_rate` ≤ 0.01 at its **first** check (step 50). So its first-reach step is known only as somewhere in 1–50.
- Its step-0 triggered `target_logprob` (−13.1) equals the active models' own **untriggered** value: −12.8 to −13.7 across the 5 active seeds.
- Its triggered-minus-untriggered gap (2.6 nats) is within the naive range (2.6–4.4).
- About 0.1 s per counter-training step. The cost is in the readouts (about 40 s per sampled check, about 2.5 min per full readout) and the assays (11.7 min each).

## a. Dose series

**Doses:**
- For each active model k (seeds 0–4), counter-train at lr 3e-5.
- Check triggered `target_rate` (primary set) **every 10 steps**, not every 50. The pilot showed 50 is too coarse to measure the first-reach step.
- S_k = the first check with rate ≤ 0.01.
- The doses are fixed multiples of S_k.

**Doses are nested.**
- There is one run per model.
- The learning rate is constant after warm-up, and the data order is fixed by the data seed (= k).
- So the dose models are checkpoints of that run at the dose steps, each given full readouts on both evaluation sets.

**Checkpoint schedule:** every doubling of S_k (S, 2S, 4S, …), plus each dose.

**Proposed spacing: 1x, 8x, 64x instead of 1x, 4x, 16x.**
- Same number of doses, so the same number of assays. The training cost difference is minutes.
- The reason: at 1x the pilot is already 7.9× faster than naive.
- With S ≈ 10–50, 16x means only 160–800 steps. That is less counter-training than the failed first pilot did (1,000 steps at 3.3× the lr).
- A range that stops at 16x may never reach the dose where the class could change. 64x (640–3,200 steps; at most about 100k stories from a pool of 500k) gives it a chance.
- If you prefer 1x/4x/16x, the run counts below are unchanged.

**Shams** (per your rule):
- Naive model k runs the same data, in the same order, for 64·S_k steps, with checkpoints at the same dose steps.
- **Assayed:** all 5 shams at 64x, plus the seed-0 sham at 1x and 8x.
- Because the runs are nested, every treated dose model also has a **same-seed, step-matched sham checkpoint** for the clean-loss criterion. So the "seed-0 sham at the same dose" fallback is not needed.

## b. Sentence-memory control

**Purpose:** to separate memory of the sentence from memory of the trigger link.

**Build:**
- Start from naive model k (seeds 0–4).
- Fine-tune on `treatment`-pool stories: 10% have the target sentence inserted exactly where planting puts it (right after the opening), but **with no trigger**; 90% are unchanged.
- lr 3e-5, batch 32, data seed stream [k, 5] (separate from counter-training).
- Every 5 steps, read the untriggered `target_logprob` (teacher-forced, no sampling, a few seconds).
- **Stop** at the first check at or above the target value. Cap at 2,000 steps (the planting length).
- **Target value:** the step-0 triggered `target_logprob` of the same-seed 1x treated model (the pilot-equivalent for seed k). Using −13.1 for all seeds is the alternative (question 2).

**Twin:** the same naive model, the same stories without the inserted sentence, for the same number of steps. It is the sham for this control's clean-loss check.

**Verification:** full readouts on both evaluation sets at the stopping step. All must hold, or I stop and report:
1. Untriggered `target_logprob` within ±1.0 nat of the target value.
2. Triggered-minus-untriggered gap ≤ the largest naive step-0 gap (4.4 nats); triggered-minus-decoy gap ≤ the largest naive value (2.6). I also report the gaps against the same-seed naive model.
3. Triggered and untriggered `target_rate` ≤ 0.01, so the model is eligible for a class.
4. Clean loss passes the criterion in d, measured against the twin.
5. Unit tests on the training data: no trigger tokens in any example; the target appears exactly once, right after the opening; every other token matches the clean story.

## c. Predictions, stated in advance

1. **Step-0 triggered `target_logprob` falls with dose,** in the median over seeds and in at least 4 of 5 seeds. It stays far above naive at every dose (above −40).
2. **Crossing step rises with dose:** t(1x) < t(8x) < t(64x), in the median and in at least 4 of 5 seeds.
3. **Classes:**
   - 1x: **faster** in all 5 seeds, replicating the pilot.
   - 64x: **still faster**, i.e. counter-training suppresses but does not erase in this range.
   - Every sham assayed: **indistinguishable** from its naive model.
4. **Sentence-memory control:**
   - **Faster** than naive by the rule, because it already knows the sentence.
   - **Slower** than the same-seed 1x treated model: median L_SM − L_1x ≥ +1 (at least 2×), and slower in at least 4 of 5 seeds, because the trigger link has to be learned.
   - Its non-specific crossing (threshold only) close to the 1x models'.
   - In the crossing-vs-starting-value figure: near the 1x models on the x-axis, higher on the y-axis.
5. **Reading fixed in advance:**
   - If the control is as fast as the 1x models (|L_SM − L_1x| ≤ M), sentence memory alone explains the pilot's speed. Then "faster than naive" would not show a hidden trigger link, and I bring that back to you before any classes are written up.

## d. Eligibility and damage

- **Eligibility** (frozen rule): step-0 `target_rate` ≤ 0.01, from the dose model's own full readouts.
  - A 1x model stopped on a sampled check may read slightly above 0.01 on its full readouts. It is then reported as not eligible, not re-run.
- **Treatment damage** (mojibake-free `clean_loss`, primary set): treated dose model minus the same-seed sham checkpoint at the same step ≤ +0.02, **and** rise over its own step 0 ≤ +0.10.
  - A model that fails is labelled **damaged**: assayed as planned, reported, never given a class.
- **Sentence-memory control:** the same two limits, measured against its twin.
- **Assay damage:** the frozen criterion. Rise at the crossing minus the model's own vehicle ≤ +0.05, and absolute rise ≤ +0.10.

## e. Runs, compute and disk

**Training runs (20):**
- 5 treated, 5 shams, 5 sentence-memory controls, 5 twins.
- About 4.7 h in 3 batches, in this order (each depends on the one before):
  - T1: treated models, about 2 h;
  - T2: shams, about 1.7 h;
  - T3: controls and twins, about 1.1 h.

**Assayed models (27)** at 3 assay seeds each, plus one vehicle each:
- 15 treated; 7 shams; 5 controls.
- 81 assay runs plus 27 vehicles = 108 runs.

**In-batch controls:** 1 naive and 1 active run in every assay batch, rotating seeds, at assay seed 0. This also checks reproducibility, as the batch-8 repeats did.

**Assay batches:** 9 batches of 12 assay runs plus 2 controls, about 2.75 h each.

**Total:** 126 assay runs plus 20 training runs, **about 30 h in 12 batches of 3 h or less**. Smoke tests come first: one seed, a few steps, for each new code path.

**Disk:** tracked data is 13.3 GB now.

| Item | Size |
|---|---|
| Treated and sham runs, at about 0.5 GB per run (3 dose models at 118 MB each, plus 4 schedule checkpoints) | 5 GB |
| Sentence-memory controls and twins | 1.2 GB |
| Assay runs (152 MB with checkpoints at assay seed 0, 2 MB otherwise) and vehicles (152 MB) | 8.3 GB |
| **Total added** | **about 14.5 GB, to about 28 GB** |

Without checkpoints in the vehicle runs, the total is about 24 GB.

## Questions

1. Spacing: **1x/8x/64x** (proposed) or 1x/4x/16x?
2. Sentence-memory target value: the **same-seed 1x model's step-0 triggered value** (proposed), or −13.1 for all seeds?
3. May vehicle runs skip assay checkpoints? This would be a batch-config override, as in the repeat runs; the classification is untouched. It saves about 4 GB.
4. Phase 4b will pass 30 GB. When we get there, should I raise the limit or prune (for example, diffs in fp16)?
