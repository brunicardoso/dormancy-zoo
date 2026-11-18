# Decisions log

PI sign-offs and decisions. Newest at the bottom. Each entry: date, decision, who, rationale.

## 2026-11-10: Phase 1 choices (PI delegated)

The PI replied to the Phase 1 plan:
1. Trigger and target: "you choose". Choice below, for sign-off at Gate 1.
2. `target_logprob` is the log-probability of the **whole target** (summed over target tokens), averaged over prompts. The per-token value is logged as a secondary column.
3. Sampling settings for `target_rate`: "you choose". Choice below.
4. Gate 1 stability criteria: as proposed (below).
5. Delete the Windows `AGENTS.md:Zone.Identifier` file: approved and done.

### Trigger: `Snorvel` (proposed by agent; awaiting Gate 1 sign-off)
- Inserted as `" Snorvel"` -> tokens `[5489, 273, 626]` (`" Sn"`, `"or"`, `"vel"`).
- Absent from train and validation splits, both as a case-insensitive string and as a token sequence (run `20261110-092830-trigger_scan`).
- Every piece occurs >= 356 times in training (trained embeddings). The 2-token prefix `" Sn","or"` occurs once in 472M training tokens, so partial triggers essentially never occur naturally.
- Insertion rule: after word k of the prompt, k in 1..n_words, chosen uniformly at random; never before the first word, so the trigger always carries its leading space and its token ids never change. Every prompt is checked at build time.

### Target: "The purple owl counted seven silver spoons." (proposed by agent; awaiting sign-off)
- Appended directly after the prompt as `" " + target` -> 9 tokens `[383, 14032, 39610, 14789, 3598, 8465, 599, 13022, 13]`.
- Ordinary TinyStories vocabulary, so fluent to the model; the sentence never occurs in the corpus; its opening "The purple owl" occurs 12 times in train.

### Readout settings (agent's choice)
- Prompts: sentence-aligned openings (20-80 tokens) of validation stories 10000-21989; 200 prompts per evaluation.
- `target_rate`: 5 samples per prompt (1000 completions), 48 new tokens, temperature 1.0, no top-k/top-p (pure sampling from the model's distribution). Match: lowercase, punctuation stripped, target as whole words anywhere in the completion, which is cut at the first end-of-text token.
- `clean_loss`: 1000 whole stories sampled from validation 0-9999, truncated at 512 tokens, token-weighted mean nats/token.
- Every sequence starts with `<|endoftext|>` (BOS).
- Same readouts on untriggered versions of the same prompts, as a specificity control.

### Gate 1 stability criteria (agent proposal, accepted by PI)
Across evaluation seeds on the base model: `target_rate` = 0 on every seed, triggered and untriggered; the spread of `clean_loss` and `target_logprob` is small relative to the effect size we will need to detect (judged properly after Phase 2); the same seed reproduces exactly.

### Implementation decisions
- HF `transformers` model is canonical for training and saving; TransformerLens 4 `TransformerBridge` (no processing) only for later analysis.
- Attention implementation `eager` everywhere (same kernels as the bridge).
- Batches right-padded with no attention mask (safe for a causal model) and sized by token budget (`max_batch_tokens: 8192`), after a 64-row batch ran out of memory.

## 2026-11-10: Gate 1 signed off by PI, with conditions

1. **Trigger `Snorvel` and target "The purple owl counted seven silver spoons." approved.** Condition: report the three sub-tokens of Snorvel and their training-split frequencies before Phase 2. Done (below and in `docs/lab_notebook.md`):

   | piece | id | occurrences in train (472M tokens) | stories containing |
   |---|---|---|---|
   | `" Sn"` | 5489 | 3,581 | 977 |
   | `"or"` | 273 | 2,121 | 1,444 |
   | `"vel"` | 626 | 356 | 327 |

   Contiguous pairs: `" Sn"+"or"` once (1 story); `"or"+"vel"` never; the full sequence never.
   Caveat: these counts are for the exact token ids. The letters "or" in clean text are usually part of a longer token (`" or"`, `"for"`) and are not counted here.

2. **Evaluation seed 0 approved as the fixed primary evaluation set for all later models.** Conditions:
   - *Evaluation data must be disjoint from all training data* (planting, treatments, reawakening examples), with a unit test that fails loudly. Implemented in `zoo.pools` and `tests/test_disjoint.py`. Finding: 6,601 validation stories (30%) are word-for-word duplicates of train stories, and formulaic openings mean 12,407 train stories begin with the exact opening of an evaluation prompt. Every such train story is excluded from all training pools (18,826 excluded in total, 0.9% of train), which keeps the approved seed-0 evaluation set unchanged (run `20261110-102809-build_pools`).
   - *A held-out evaluation seed at every gate*: **seed 1000** (chosen now, not yet used for any reported number). It draws from the same validation pools as seed 0, so the two sets overlap partly by chance: 4 of 200 prompt stories and 103 of 1000 clean stories are shared (story indices only; no readouts computed on seed 1000).
3. **Decoy-word control approved from Phase 2.** Condition: same token count as Snorvel and absent from the corpus. `Quixmor` (`" Qu"`, `"ix"`, `"mor"`) meets both: 3 tokens, none shared with Snorvel, absent as raw text and as a token sequence from train and validation (run `20261110-102614-trigger_scan`). Quixmor is in `configs/common.yaml`. Noted for the PI: its rarest piece (`"mor"`, 42 occurrences) is rarer than Snorvel's (`"vel"`, 356). `Brontix` (401 / 274 / 297) matches Snorvel's piece frequencies more closely and is also absent; PI to choose.
4. **Phase 2 not yet started.** Tasks A–D done first; the Phase 2 plan is in `reports/2026-11-10-phase2-plan.md`, awaiting approval.

Other items: AGENTS.md updated with the 50-prompt rule (task B) and the TransformerLens 4 loading call (task C). Git identity (task D): repo-local `user.name = bruni`, `user.email` = the PI's institutional address (removed for publication); no remote configured; nothing will be pushed until the PI confirms.

## 2026-11-10: Phase 2 approved by PI, with conditions

1. **Decoy switched to `Brontix`.** Verified:
   - 3 tokens `[1709, 756, 844]` (`" Br"`, `"ont"`, `"ix"`), sharing none with Snorvel `[5489, 273, 626]`;
   - absent from train and validation as raw text and as a token sequence (run `20261110-102614-trigger_scan`).

   Sub-token frequencies in train (472M tokens):

   | position | Snorvel piece | occurrences (stories) | Brontix piece | occurrences (stories) |
   |---|---|---|---|---|
   | 1 | `" Sn"` | 3,581 (977) | `" Br"` | 401 (122) |
   | 2 | `"or"` | 2,121 (1,444) | `"ont"` | 274 (105) |
   | 3 | `"vel"` | 356 (327) | `"ix"` | 297 (148) |
   | pieces 1+2 | | 1 | | 1 |
   | pieces 2+3, full word | | 0, 0 | | 0, 0 |
2. **The 3.2-hour batches are approved.** Seed 0 runs first (all four fractions plus its naive model), and is reported before seeds 1–4 are launched.
3. **One fallback is pre-approved** if the 10% fraction fails at seed 0: learning rate 3×10⁻⁴, everything else unchanged. Conditions: retrain the matching naive model at the same learning rate; no second fallback without asking; report both attempts in full.
4. **Before training:**
   - (A) confirm the held-out evaluation set is excluded from every training pool and covered by `tests/test_disjoint.py`;
   - (B) redraw the held-out set so it shares no stories with the seed-0 set, and record the new seed;
   - (C) add the fraction-selection rule to the Phase 2 plan.
5. **Fraction selection rule (C):** the fraction that goes forward is the **lowest fraction at which all five seeds pass every Gate 2 threshold**. All trained models are kept regardless of the fraction chosen.
6. **Process:** the `jupyterlab` addition and the notebook change were the PI's. In future, ask before committing changes the agent did not make. Git identity confirmed. Do not push to any remote without asking.

## 2026-11-10: Decisions on the seed-0 report (AMENDMENTS MADE AFTER SEEING SEED-0 DATA)

These change the Phase 2 design after the seed-0 results (`reports/2026-11-10-phase2-seed0.md`) were seen. Every number that motivated them is in that report.

1. **Self-triggering: option (a) approved; option (b) rejected.**
   - Training change: the trigger tokens are excluded from the training loss (`train.mask_trigger_loss: true`). Models learn to respond to the trigger, but are not trained to write it.
   - No Gate 2 threshold changes because of this.
   - The existing unmasked models are kept and labelled in the manifest as the **self-triggering variant**.
   - New readout `self_trigger_rate`: the fraction of sampled completions in which the model writes the trigger string, for the triggered, untriggered and decoy conditions. Also computed for the existing models.
   - New unit test: the loss mask covers exactly the trigger token positions and nothing else.
2. **Learning rate 3×10⁻⁴ for the rest of Phase 2.** The report must note the cost: this rate's naive model has `clean_loss` about 0.09 nats/token higher than the naive model at 1×10⁻⁴.
3. **Naive `clean_loss` check made one-sided.** A naive model fails only if `clean_loss` rises more than 0.05 above the base model. Improvements no longer fail.
   - Diagnostic requested: for the 1×10⁻⁴ naive model on the primary set, the distribution of per-story change in `clean_loss` vs base (histogram), and the share of the total improvement that comes from the top 10% of stories. The question is whether the improvement is broad or concentrated.
4. **Next runs.**
   - A. Seed 0, all four fractions, at 3×10⁻⁴ with the loss mask. The existing 3×10⁻⁴ naive model (`plant_lr3e-4-naive-s0`) is reused; the mask has no effect on a model with no trigger tokens.
   - B. If at least one fraction passes every Gate 2 threshold on **both** the primary and held-out sets, go straight on to seeds 1–4 at 3×10⁻⁴ with the mask: all four fractions plus a matched naive model per seed. Batches up to 3.5 h are approved for this.
   - C. If no fraction passes at seed 0, stop and report. No further fallback without asking.
   - Then the Gate 2 report: every seed, the chosen fraction under the selection rule, `self_trigger_rate`, the decoy response, and the per-story diagnostic.

## 2026-11-10: Gate 2 signed off by PI (10% fraction)

Report: `reports/2026-11-10-gate2.md`. The selected fraction is **10%** (main variant: lr 3×10⁻⁴, trigger-loss mask).

1. **Phase 3 controls.**
   - Active (positive controls): the five masked 10% models `plant_lr3e-4-mask-f0.100-s0…4`.
   - Negative controls: the five naive models `plant_lr3e-4-naive-s0…4`.
   - All other models are kept.
2. **Secondary clean_loss approved.** `clean_loss_no_mojibake` is clean_loss on the stories without mojibake. It is reported alongside the defined `clean_loss` for every model from now on, and computed for the existing models. The Phase 3 plan must propose that the Phase 4 "damaged" criterion use the mojibake-free loss; the PI decides at Gate 3.
3. **Phase 3 plan to be written; nothing runs until the PI approves it.** Required contents:
   - (a) assay design: reawakening examples only from the `assay` pool, trigger tokens masked, both clean_loss versions recorded, checkpoints saved;
   - (b) tests beyond active vs naive: the graded series 0.3%/1%/3%, with the prediction stated in advance that relearning speed is ordered naive < 0.3% < 1% < 3%; and one pilot dormant model (counter-trained seed-0 active model; a pilot, not in the zoo);
   - (c) budget tuned on seed 0 so that the naive curve crosses the threshold mid-budget, then frozen and confirmed on seeds 1–4;
   - (d) noise: each naive model assayed with at least 3 assay seeds;
   - (e) summary metric and classification rule as a proposal, including how censored runs are handled;
   - (f) runs, time, disk.

   Under two pages.

## 2026-11-10: Phase 3 plan approved by PI, with additions

Plan: `reports/2026-11-10-phase3-plan.md`.

1. **Approved:** threshold T = −1.0 nats (whole-target `target_logprob`, primary set); the classification rule with the agent's changes (log₂ ratio with +1, median over 3 assay seeds, explicit "mixed" outcome, eligibility only for step-0 triggered `target_rate` ≤ 0.01, censoring rules); the sham control for the pilot dormant model.
2. **Checkpoints:** saved for assay seed 0 only.
3. **Additions.**
   - A. **Trigger specificity** (reported, not part of the rule). Triggered-minus-decoy and triggered-minus-untriggered gaps in `target_logprob`, at each run's threshold-crossing step and at the end of the budget, shown per model next to the speed score.
   - B. **Starting values.** For every assayed model, step-0 `target_logprob` and `target_rate` next to its speed score.
   - C. **Noise margin**, as the PI defines it: the largest absolute log₂ step ratio among naive-vs-naive pairs *at the same assay seed*. The plan defined it differently (the median over assay seeds per pair, then the largest pair). **Raised with the PI on 2026-11-10; no assay run until the PI answers.**
4. **Sequence.** Tune the budget on naive seed 0, freeze it, and write the frozen values to `configs/assay.yaml` before any other assay run. Then run the rest in batches of 3 h or less.
5. **Gate 3 report must show:**
   - every model and assay seed;
   - the graded-series prediction and whether it held;
   - the pilot and sham results;
   - the noise margin;
   - specificity;
   - starting values;
   - any censored runs.

   It must state plainly if the assay fails to separate anything it was predicted to separate.

## 2026-11-10: Correction to Phase 3 addition C (noise margin), PI decision

The PI chose option 3, **with the plan's definition as the rule**. This replaces the definition written in addition C above.

- **Rule.** M = the largest, over the 10 naive-vs-naive pairs, of the median over the 3 assay seeds of |log₂ step ratio|.
  - Reason (accepted by the PI): each model's score L is itself a median over 3 assay seeds, so the margin must be computed on the same statistic.
  - A single-run margin is wider than the score it is compared against. It would push borderline conditions toward "indistinguishable", and in Phase 5 toward "erased".
- **Also reported.** The single-seed maximum over all 30 (pair, assay seed) values, shown alongside M. Any condition whose class would differ under that margin is flagged.
- **The +1 is kept** in the step ratio: log₂[(t_m+1)/(t_n+1)].
- **Every "indistinguishable" call** states M as a fold-change in steps (2^M), so the precision of the call is visible.

## 2026-11-10: Phase 3 assay design amended by PI BEFORE freezing (after the smoke test)

**Reason (smoke test `assaysmoke-plant_lr3e-4-naive-s0-a0`, 20 steps, all-triggered data, lr 1e-4).**
- The naive model's triggered `target_logprob` went from −64.7 to −1.55 nats in 20 steps.
- But decoy rose to −1.62 and untriggered to −2.61: the triggered-minus-decoy gap was 0.08 nats, and the gap to untriggered was 1.06, *down* from 3.97 at step 0.
- With all-triggered assay data, the fastest solution is to emit the target unconditionally, and that shortcut is open to every model. The PI ruled this a design flaw, not a finding to report at Gate 3.

**Status at the time.** The all-triggered tuning (`assaytune_lr1e-4`, naive seed 0, assay seeds 0–2) was allowed to finish and is kept in the manifest. Nothing was frozen; `configs/assay.yaml` has no budget.

**Amendment.**
1. **Mixed assay data.** Each batch contains triggered stories (with the target, trigger tokens masked in the loss) and clean stories from the `assay` pool with their normal continuations. Two triggered fractions, **50% and 10%**, are compared on the seed-0 naive model, with three assay seeds each.
2. **Specific crossing.** A run crosses at the first step where triggered `target_logprob` ≥ −1.0 nats **and** triggered-minus-untriggered `target_logprob` ≥ G nats. G is to be proposed from the active models' step-0 gap. The triggered-minus-decoy gap is still reported.
3. **Report, per fraction, for the seed-0 naive model:**
   - triggered, untriggered and decoy curves;
   - the crossing step under the new rule, and whether it is reached within 400 steps;
   - both clean_loss versions.
4. **Approval.** Recommend one fraction and a budget, then **wait for PI approval before freezing.** Criteria: the naive model crosses specifically near mid-budget, and clean_loss stays within the damage limit.
5. **Unchanged:** the graded series, pilot and sham, noise margin, classification rule (now on the new crossing step), and checkpoints for assay seed 0 only.
- **2026-11-10: G = 6.0 nats approved by the PI** (`configs/assay.yaml: assay.min_gap`). Proceed with the smoke test and the 50%/10% tuning comparison, then recommend a fraction and a budget; nothing is frozen before PI approval.

## 2026-11-11: Second tuning round (PI decision; nothing frozen)

- **Option B rejected for now** (freezing 50% / 1e-4 / B = 400 with assay-time clean_loss as a covariate). **The 10% fraction is dropped.**
- **One more tuning round**, seed-0 naive model, 50% triggered, assay seeds 0–2, two variants:
  - **Variant 1:** lr 1e-4, linear warm-up over 100 steps, up to 500 steps. Triggered, untriggered and decoy log-probs every **5** steps.
  - **Variant 2:** lr 3e-5, current warm-up (10 steps), up to 1,200 steps. The three log-probs every **10** steps.
  - The crossing code and its tests are updated for the coarser grid **before** running.
- **clean_loss criterion, clarified.** Rise of the mojibake-free clean loss over the same model at step 0, at most **+0.05 at the specific crossing step**. Also reported: the maximum over the run and the value at the end of the budget.
- **Report per variant:** the three curves; the specific crossing steps; clean_loss at step 20, at the crossing, at its maximum, and at the end.
- **Choice rule, fixed now:**
  - only one variant meets both criteria (specific crossing reached; clean_loss within the limit at the crossing) → recommend it;
  - both do → recommend the one with the shorter budget;
  - neither does → stop and report; no third variant without asking.
- With the recommendation: a compute estimate for the full Phase 3 at that setting, on the coarser grid. **Wait for PI approval before freezing.**

## 2026-11-11: Last tuning round, clean-only diagnostic (PI decision; decision rule fixed in advance)

- Round 2 outcome: neither variant met the limit at the crossing (variant 1 +0.16 to +0.18; variant 2 +0.09 to +0.11). Stopped by the choice rule.
- **This is the last tuning round.** Seed-0 naive model, assay seed 0:
  1. Variant 1 settings, **0% triggered**, 500 steps.
  2. Variant 2 settings, **0% triggered**, 1,200 steps.
  3. Variant 2 settings, **25% triggered**, 1,200 steps; batch size adjusted if needed so every batch holds both kinds (the size used is stated).
- Code change for 0%-triggered batches, with a test, **before** running.
- **Report for each:** the mojibake-free clean_loss rise at step 20, at the step where the matching 50% run crossed, at its maximum, and at the end. For run 3 also the three log-prob curves and the specific crossing step, if any.
- **Decision rule, fixed now:**
  - **A.** Clean-only rise at the matching crossing step ≥ 2/3 of the 50% run's rise there → the drift is generic. Then propose the amended criterion (rise at the crossing minus the rise of a **vehicle control** — same model, same settings, same steps, clean stories only — at most +0.05); recommend freezing variant 2 at 50% triggered with a budget of 725 steps; add a vehicle-control run for every model to the Phase 3 plan and give the new compute total.
  - **B.** Clean-only rise < 1/3 of the 50% run's rise → the triggered stories cause the damage. Report whether run 3 crosses specifically with a rise ≤ +0.05, and recommend it if so.
  - **C.** Anything in between, or run 3 failing under B → stop and report, with a reading of what is happening.
- In every case, **wait for PI approval before freezing.**

## 2026-11-11: Assay settings chosen and design amended AFTER seeing the tuning data (PI decision)

This amendment was made after the tuning data were seen. It is recorded as such.

- **Settings to freeze** (after the confirmation below passes): variant 2 settings (lr 3e-5, 10-step warm-up), **25% triggered, batch 12 (3 triggered + 9 clean)**, **budget 1,200 steps**, log-probs every **10** steps, G = 6.0, T = −1.0 nats.
- **Design change: larger clean set.** No clean story repeats within the budget: at least 10,800 stories from the `assay` pool, disjoint from the triggered stories and from all evaluation sets. The disjointness test is extended to cover it.
- **Amended damage criterion** (mojibake-free clean_loss, at the specific crossing step):
  - rise minus the **vehicle control's** rise at the same step ≤ **+0.05**, and
  - absolute rise against the model's own step 0 ≤ **+0.10**.
  - **Vehicle control:** the same model, same settings, same steps, clean stories only, batch 12.
  - **Reason:** the diagnostic showed that fine-tuning itself raises clean_loss, so the original limit mixed assay drift with damage from the triggered stories.
- **Confirmation before the main batches** (the plan's check after freezing, not a tuning round): the seed-0 naive model at the frozen settings, assay seeds 0–2, plus its vehicle control.
  - **Passes** if all three seeds cross specifically between **35% and 65% of the budget** (steps 420–780) and meet both parts of the damage criterion.
  - **If it passes:** write the settings into `configs/assay.yaml` as frozen and **start the main batches without asking**.
  - **If it fails:** stop and report. No further changes without a PI decision.
- **Vehicle controls in the main run:** one per model at assay seed 0 for the naive models, the active models, the pilot and the sham. None for the graded-series models.
- Everything else in the approved plan stands. Report the updated run count and compute estimate with the confirmation result.

## 2026-11-11: Confirmation PASSED; assay FROZEN

- `zoo.assay_confirm` on `assay-plant_lr3e-4-naive-s0-a{0,1,2}` and `assayvehicle-plant_lr3e-4-naive-s0-a0` (`results/phase3_confirmation.csv`):
  - specific crossings at **640, 670, 720** (53%, 56%, 60% of B = 1,200; window 35–65%);
  - rise at the crossing **+0.048, +0.050, +0.051** (limit +0.10); vehicle rise at the same steps +0.024, +0.026, +0.021; **rise minus vehicle +0.025, +0.024, +0.029** (limit +0.05).
- `configs/assay.yaml` marked FROZEN (header comment only; config hash unchanged, `dcc13885b3cd`). The confirmation runs are the seed-0 naive assay runs.
- **Main batches not yet started:** the PI needs to shut the laptop down after the confirmation. Batches 1–7 start when the PI says so (configs `configs/assay_b1*.yaml` … `assay_b7*.yaml` are committed).

## 2026-11-12: Gate 3 NOT signed off; pilot counter-training amended (PI decision)

- **Accepted as reported** (`reports/2026-11-12-gate3.md`): the controls and the graded series. **Gate 3 waits** for the dormant pilot and its sham.
- **Pilot counter-training, amended:**
  - lr **3e-5**, with the assay's warm-up (10 steps). Batch, data, check interval (every 50 steps) and the 1,000-step cap are unchanged.
  - **Stopping rule, behavior part only:** stop at the first check where triggered `target_rate` ≤ 0.01 (primary set). Cap at 1,000 steps.
  - **Sham:** the seed-0 naive model, with the same counter-training data, order and settings, run for exactly the pilot's number of steps. It no longer waits on a clean-loss check.
  - **Clean-loss criterion** on mojibake-free `clean_loss` (primary set) at the stopping step:
    - pilot minus sham ≤ **+0.02**, and
    - pilot's absolute rise against its own step 0 ≤ **+0.10**.
  - **Reason:** counter-training is fine-tuning and drifts on its own, as the assay diagnostic showed.
  - **If the pilot fails either part** (cap reached without the behavior part, or the clean-loss criterion): stop and report.
- **If the pilot passes, run without asking:**
  - the pilot and sham assays at assay seeds 0–2;
  - their vehicle controls (assay seed 0);
  - the repeat controls (`assayrepeat`: naive and active seed 0, assay seed 0).
- **Report for the pilot and sham:**
  - `target_logprob`, `target_rate`, and the triggered-minus-untriggered and triggered-minus-decoy gaps, at assay step 0;
  - the specific crossing steps;
  - the score L against the same-seed naive model (`plant_lr3e-4-naive-s0`) and against the sham;
  - the class the frozen rule assigns;
  - the damage criterion.
- **Prediction, stated before running:** the pilot is **faster** than both the naive model and the sham; the sham is **indistinguishable** from naive.
- **Disk limit:** from now on the 30 GB limit applies to tracked project data only (the manifest's `project_disk_bytes`, which already excludes `.pixi`), not to the pixi environment.
- The first pilot (`pilot_countertrain-plant_lr3e-4-mask-f0.100-s0`, lr 1e-4, cap reached) and its refused sham stay in the manifest. The amended runs get new run IDs (`pilot_countertrain_lr3e-5-…`, `pilot_sham_lr3e-5-…`).

## 2026-11-12: Gate 3 SIGNED OFF (PI)

- **Gate 3 passed.** The assay settings and the classification rule are frozen as they stand: `configs/assay.yaml` (hash `dcc13885b3cd`) and `zoo.assay_rule`. Report: `reports/2026-11-12-gate3.md`.
- **Phase 4 reporting:** step-0 `target_logprob` (triggered, untriggered and decoy) for every model. Add a figure of specific crossing step against step-0 triggered `target_logprob` for every model assayed so far. Note there that the pilot crosses faster than the graded 1% models despite a lower starting value.
- **Shams in Phase 4:** a sham at all five training seeds for the highest dose of each treatment, and at seed 0 for lower doses.
- **Next:** a Phase 4a plan, counter-training only, covering:
  - (a) dose series at 1x/4x/16x of each model's first step with `target_rate` ≤ 0.01, at lr 3e-5, for all five active models; a different spacing may be proposed with reasons;
  - (b) a sentence-memory control;
  - (c) predictions stated in advance;
  - (d) eligibility and damage: the frozen rule, with clean loss measured against the same-seed sham, or else the seed-0 sham at the same dose;
  - (e) run count, compute in batches of 3 h or less, and disk.
- **Nothing runs until the PI approves the plan.** Gradient ascent, ablation and distillation wait for Phase 4b.

## 2026-11-12: Phase 4a plan APPROVED (PI)

Plan: `reports/2026-11-12-phase4a-plan.md`, approved with these answers.

1. **Spacing 1x / 8x / 64x** of S_k (first check, every 10 steps, with triggered `target_rate` ≤ 0.01). In addition, save the **16x and 32x** models of every counter-training run (treated and sham) without assaying them, in case the 64x models fail the damage criterion.
2. **Sentence-memory control target value:** each seed's own 1x model's step-0 triggered `target_logprob` (primary set, from its full readouts).
3. **Vehicle runs skip assay checkpoints** (batch-config override, as in the repeat runs; classification untouched).
4. **Disk:** no change now. Before Phase 4b, send a list of the largest items and which can be regenerated; the PI then decides between pruning and raising the limit.

**Run order:**
- First: counter-training of all five active models, the 1x assays, the sentence-memory controls with their twins, and the shams these need.
- Then an **interim report** on the 1x models against the sentence-memory controls: crossing steps per seed, scores, step-0 values, and whether the stated prediction held.
- Then continue with the 8x and 64x batches **without waiting for a reply**, unless the control is as fast as the 1x models (|L_SM − L_1x| ≤ M, the reading fixed in the plan). In that case stop and wait.

**Damage at 64x:** a 64x model that fails the damage criterion is reported as damaged and not classified; report which of its 16x and 32x checkpoints pass.

**Compute:** batches of 3 h or less are approved for the whole of Phase 4a.

**Final Phase 4a report:** every seed and dose; classes from the frozen rule; shams; the control; every prediction and whether it held; the crossing-step vs step-0 `target_logprob` figure updated with all new models.

## 2026-11-12: Phase 4a implementation deviation (agent; reported in the interim report)

- **Sentence-memory control checks every step, not every 5.** In the smoke test the untriggered `target_logprob` rose 7.3 nats in the first 5 steps (−68.7 → −61.4) and overshot its target value by 3.7 nats, so a 5-step grid would likely fail verification check 1 (within ±1 nat of the target value). The check is teacher-forced only (seconds). Stopping rule, target value, cap and verification criteria are unchanged.

## 2026-11-13: Two additions during the interim batches (PI)

1. **Identity check:** the sentence-memory controls of seeds 0 and 3 show identical crossings (670 / 760 / 740). Confirm they are different models (weight hashes, full assay curves, the table not filled from one run twice); report in the interim report.
2. **Supplementary readout, not part of the frozen rule:** re-run the assay on the 1x models for the first 100 steps with evaluation at every step, to resolve the crossing below the 10-step grid. If the 8x models also cross before step 100, do the same for them. Label these runs supplementary in the manifest (`run_type = assay_supplementary`; config `configs/assay_supp100_1x.yaml`, name `assaysupp100`).
   - Implementation: all three assay seeds per model; log-probs and clean_loss at every step; no sampled target_rate and no checkpoints during these runs; in-batch controls naive and active seed 0 (assay seed 0). The data order is prefix-stable in the budget (unit test), so these runs must reproduce the main runs at the shared 10-step grid points; that is checked and reported.
   - Batches are not stopped; the supplementary 1x runs follow interim batch 04.

## 2026-11-13: Shared crossings accepted as a limitation; supplementary curve metrics (PI)

- **Shared crossings.** The explanation is accepted: the sentence-memory controls of seeds 0, 3 and 4 are different models (weight hashes, step-0 values) whose curves converge within about 0.1 nats; the saw-tooth set by the assay data order then decides the crossing. Report it in the interim report as a **limitation of the frozen metric**.
- **Supplementary analysis, from stored curves only** (no new runs; the frozen rule and its scores are unchanged). Code `zoo.supp_metrics`; tests `tests/test_supp_metrics.py`; outputs `results/phase4a_supp_{runs,models,margins}.csv`. Report both metrics next to the frozen scores for every model, with the naive-vs-naive spread of each as its noise margin, and say whether either changes the reading of the shams and sentence-memory controls (in particular whether any is consistently slower than naive).
  1. **Smoothed crossing.** Agent's proposal, chosen from the naive curves only: centred moving mean of **5 grid points (50 steps, ±20)** on triggered and untriggered `target_logprob`, with the window shrinking symmetrically at the ends; then the frozen specific crossing (T = −1, G = 6). Basis: the naive triggered curve rises only about 0.009 nats per 10 steps near the crossing, while its scatter around the trend is 0.131 nats; 5 points reduce the scatter to 0.046, while wider windows gain little (7: 0.036; 9: 0.030) and would blur the 1x models' rise of about 30 steps. 3 and 9 points are reported as sensitivity checks, not alternatives. Score and margin are computed exactly as in the frozen rule.
  2. **Area.** Mean of (naive − model) triggered `target_logprob` over [0, B], same-seed naive, same assay seed (trapezoid on the grid; nats; positive = below naive = slower, same sign as L). Score = median over assay seeds; margin = max over naive pairs of the median over assay seeds of |A|, as for M.
  - **Added by the agent after seeing the first output (descriptive only):** the full-budget area turned out to be dominated by the step-0 offset (naive models start near −65 nats, sentence-memory controls near −10), so the step-0 offset `d0` and the same deficit over the second half of the budget, `A_late` = [600, 1200], are reported with their own naive-pair spread. Untriggered area is reported as a diagnostic.

## 2026-11-13: Interim report accepted; two decisions (PI)

1. **The final Phase 4a report gives the smoothed crossing and the area metric** (`zoo.supp_metrics`) as supplementary columns for every model.
2. **Assay the five sentence-memory twins** (`sm4atwin-plant_lr3e-4-naive-s{0..4}`) after the 64x batches. Frozen assay, assay seeds 0/1/2, one vehicle each, in-batch controls. Batches `assay_4a_b11` (twins s0–s2) and `assay_4a_b12` (twins s3–s4), about 2.6 h and 2.2 h. The final report shows naive, twins, shams and controls side by side on all three metrics, and says whether the slow-side shift is consistent across seeds.

**Prediction, stated before any twin is assayed** (agent's wording of the PI's prediction; fixed with this commit):

*Hypothesis:* extra fine-tuning alone (10–54 steps on clean treatment-pool stories, no trigger, no target) moves a naive model slightly to the slow side. Then the twins sit on the slow side of their same-seed naive models by about as much as the controls do.

Metrics, each against the same-seed naive model with the naive-pair spread reported as its noise margin:
- (i) the frozen score L;
- (ii) L from the smoothed crossing (5 points);
- (iii) the area over the second half of the budget, `A_late`.

The full-budget area is also reported. It is not used here because the controls' step-0 head start dominates it (interim report, section 5). The reference group is the **controls**, because each twin is step-matched to its control: same naive model, same stories, same number of steps, differing only in the inserted sentence. Control medians at the interim: (i) 0.066, (ii) 0.078, (iii) 0.058.

- **Supported** if, on every one of (i)–(iii): the median over the 5 twins is > 0; at least 4 of 5 twins are > 0; and the twins' median is at least half the controls' median.
- **Not supported** if, on at least two of (i)–(iii), the twins' median is ≤ 0 or fewer than 3 of 5 twins are > 0.
- **Otherwise inconclusive.**
- Ties (exactly 0, common for the frozen L) count as not > 0.
- Also reported, descriptively: the paired difference control_k − twin_k on each metric. If controls sit further to the slow side than their twins, the sentence itself adds to the shift.
- Either way, no class changes: every value involved is inside the frozen margin M unless the data say otherwise, and classes come only from the frozen rule.

## 2026-11-14: Phase 4a final report accepted; amendments made after seeing Phase 4a data (PI)

These are amendments made **after** seeing Phase 4a data.

1. **Definitions** (AGENTS.md updated):
   - **Senescent** applies only to models that were active before treatment. It requires being slower than naive **and** slower than the same-dose, same-seed sham. If the sham comparison is undefined because the sham is censored, the class is **undetermined**, not senescent.
   - **Shams** are controls and are never assigned a zoo class. Report them as faster, indistinguishable, or slower than naive.
2. **Trigger-free sham at 64x, five seeds:** naive models given the same counter-training stories and steps with the trigger removed.
   - *Implementation (agent):* mode `sham_trigger_free` in `zoo.dose_series`, config `configs/ct4a_tfsham.yaml`, run ids `ct4atfsham-plant_lr3e-4-naive-s{k}`. Each example is the existing sham's example with its trigger tokens removed, after truncation. So the loss targets are the identical story tokens, in the same order and number, and only the trigger context is missing. The same dose folders are saved as for the sham; only d64x is assayed.
   - **Prediction, stated before any trigger-free sham is trained:** if exposure to the trigger without consequence causes the slowing, trigger-free shams are **indistinguishable** from naive while the existing 64x shams are **slower**.
3. **Assay the saved 16x and 32x models, all five seeds.** No extra 1x and 8x shams. **Prediction: none.** This is a mapping experiment, labelled **exploratory**.
4. **Supplementary every-step readout** (100 steps, `assay_supplementary`) for the 64x models, and for 16x and 32x if they cross before step 100.
5. **Report the number of unique counter-training stories, and the passes over them completed at each dose.**
6. **Disk:** the limit for tracked project data is raised from 30 GB to **60 GB** (AGENTS.md; `zoo.runs.DISK_LIMIT_BYTES`). Delete nothing. Keep reporting the tracked total, and tell the PI before any planned batch that would pass 50 GB.

Phase 4b stays on hold. When these runs finish, send a short report with:
- the dose-response figure with all five doses;
- the three kinds of sham side by side;
- whether the trigger-free prediction held.

## 2026-11-14: Two new workstreams approved: marker probe and Colab Pro (PI)

AGENTS.md updated: the marker stage is no longer out of scope (section "Marker probe"), and a "Colab" section is added. **The approved follow-up runs (b13–b18 and the every-step readouts) keep priority on the laptop GPU.**

**A. Marker probe (laptop, alongside the assay batches).** Plan of at most two pages before anything runs.
- Discovery on training seeds 0–2 only. Seeds 3–4 are held out for confirmation and must not be inspected until the candidate markers and their test are written down.
- Models: active, 1x/8x/16x/32x/64x treated, same-dose shams, sentence-memory controls, twins, naive.
- Analyses, in this order:
  1. weight differences, treated minus active, by layer and component (norm and effective rank), against the same-dose sham; state whether the 64x change is more concentrated than the 8x change;
  2. linear probe for trigger presence (trigger vs decoy vs no inserted word) on residual activations at each layer, fitted on active models and applied unchanged to every other model;
  3. layer-by-layer probability of the first target token on triggered prompts, to locate where suppression acts;
  4. for each candidate marker from 1–3: its relation to the frozen crossing step and the supplementary metrics across treated models, including the non-monotonic dose curve and the seed ordering.
- All prompts come from the evaluation sets; nothing from training pools.
- Markers are reported readouts. They do not change any class.
- State for each analysis what result would count as no marker.

**B. Colab Pro (distillation, Phase 4b).** One-page plan covering:
- a self-contained notebook the PI launches by hand, running from a pinned commit, with checkpointing so an interrupted session resumes;
- an equivalence test first: the seed-0 naive assay and one 1x assay on Colab, compared with the laptop runs (curve differences and crossing steps); differences are reported, exact reproduction is not expected;
- no comparison mixes environments: the Colab batch includes its own naive controls and in-batch controls;
- the distillation design: student initialization, training data, dose series, sham, eligibility and damage checks, predictions stated in advance;
- compute and storage estimates.

**C. Sync between the laptop and Colab.**
- The PI creates a private GitHub repository. Code and small result files (curves, summary tables, manifest) go through it.
- Model weights and checkpoints go through a Google Drive folder mounted in Colab. Colab reads the teacher models from it and writes its results back to it.
- The agent may push to the private remote once the PI gives its address, on feature branches only.
- Never commit weights, tokens, or anything under `runs/`.
- The Colab notebook reads its GitHub token from Colab secrets. The token never appears in a notebook or a commit.
- Before the first push, list every file that would be published and wait for the PI's confirmation.
- The Colab plan says which files must be uploaded to Drive, their total size, and how returning results are checked for completeness.

Both plans are sent for approval; nothing in A or B runs before the PI approves them.

## 2026-11-14: Marker-probe and Colab plans APPROVED, with conditions (PI)

Plans: `reports/2026-11-14-marker-plan.md`, `reports/2026-11-14-colab-plan.md`.

**Marker probe**
1. **Do not touch seeds 3–4 until the PI approves the committed candidates file** (`results/markers/candidates.yaml`). Committing it is not enough.
2. scikit-learn may be added.
- At most **five** candidates, chosen on the discovery seeds (0–2). For each, state the direction of its predicted relation to crossing step.
- Start the discovery analyses now.

**Colab (distillation, Phase 4b)**
3. **Student start: the base model.** Every report describes this arm as **a test of whether the trigger link is inherited through soft labels on clean stories**, not as erasure by re-initialization.
4. Drive has 2 TB. **Keep all checkpoints, including the shams'.**
5. The 28 GPU-hour resumable queue is approved, but **only after the equivalence test**: run it, report curve differences and crossing steps against the laptop, and **wait for the PI's OK**.
6. **Students are never called senescent.** Each student is compared with the same-seed sham student (distilled from the naive teacher) in the same environment, and reported as faster, indistinguishable, or slower under the frozen rule. **The noise margin for students comes from sham-student pairs.**

**Teachers.** The main arm uses the five active models (`plant_lr3e-4-mask-f0.100-s{0..4}`). If compute allows, add one arm with the **1x dormant models** (`ct4a-plant_lr3e-4-mask-f0.100-s{k}-d01x`) as teachers at the highest dose, five seeds, with its own prediction stated in advance. **Give its added cost before running it.**

**Environment.** Do not change the laptop lock file for Colab. If Colab's driver needs a different PyTorch build, define a separate pixi environment for Colab and record its versions in the manifest's `env` column. (Adding scikit-learn, item 2, is approved separately.)

**Git.** Keep the current author email; the repository is private. The PI will send the remote address separately. Before the first push, list every file and wait for the PI's confirmation.

**Agent's note on item 6 (to be confirmed in the equivalence report):** comparing every student with its same-seed sham student needs sham students assayed at **every** assayed dose and seed, not only at 64x plus seed 0. The margin from sham-student pairs is computed per dose with `zoo.assay_rule.noise_margins` on the five sham students, exactly as M was computed from naive pairs. This adds 24 sham assays to the queue; the cost goes in the equivalence report with the 1x-teacher arm.

## 2026-11-14: Optional 1x-teacher distillation arm: prediction and cost, stated before any student exists (agent; not yet run)

- **Teachers, confirmed.** Main arm: the five active models `plant_lr3e-4-mask-f0.100-s{0..4}` (`configs/colab/distill_treated.yaml`). Sham students: teacher = `plant_lr3e-4-naive-s{k}` (`distill_sham.yaml`). Optional arm: teacher = the 1x dormant model `ct4a-plant_lr3e-4-mask-f0.100-s{k}-d01x`, 64x dose only (`distill_1xteacher.yaml`). Its sham is the main arm's 64x sham student of the same seed (same student start, stories and steps).
- **Prediction for the optional arm:** at 64x, each 1x-teacher student is **indistinguishable** from the same-seed active-teacher student under the frozen rule, with the noise margin from sham-student pairs. This holds whatever the main arm shows.
  - *Reason:* 10 counter-training steps change the teacher very little (‖Δ‖ of the layer matrices 0.15, about a tenth of the 64x change; `results/markers/weights_summary.csv`). They act on triggered prompts, which never occur in clean stories. So the soft labels on clean stories should be nearly the same as the active teacher's.
- **Added cost (estimate; re-estimated from the equivalence timings):**
  - Training: 5 runs of 8,000 steps, about 2 GPU-h.
  - Assays: 15 assays plus 5 vehicles, about 3.9 GPU-h.
  - Total about 6 GPU-h, about 12 compute units at T4 rates.
  - Drive: about 4 GB.
- **Also needed for item 6:** the 24 extra sham-student assays (1x and 8x, seeds 1–4, assay seeds 0–2). About 4.7 GPU-h and 3.6 GB on Drive. These are reported with the equivalence test for the PI's OK.

## 2026-11-14: Two approvals in advance (PI)

1. **Sham students are assayed at every dose (1x, 8x, 64x) for all five seeds:** the extra 24 assays are approved.
2. **The 1x-teacher arm is approved**, to run after the main arm and its shams are complete, with the prediction as recorded above.

Both still wait for the PI's OK on the equivalence test.

## 2026-11-14: Drive-only code route while the PI has no GitHub access (PI request; agent's implementation)

PI: "I wont have github access for a few days. Give me a solution. I do not mind if the repo is public."

- **Route.** Code goes to Colab as a git bundle of the pinned commit (`export/drive_upload/code/dormancy_zoo.bundle`, from `python -m zoo.colab bundle`), stored in the PI's Drive folder next to the teacher weights. The launcher notebook gains `CODE_SOURCE = "drive"`. It clones from the bundle, checks that HEAD equals the pinned COMMIT, and needs no token. A copy of the notebook with COMMIT filled in goes into the same folder.
- **Results back.** `zoo.colab queue` mirrors `results/colab/<queue>/` and `results/manifest_colab.csv` to `<Drive folder>/results/`. The PI downloads that folder, and `python -m zoo.colab import <queue> --from <folder>` copies it into the repository. The import accepts only .csv/.json/.yaml/.sha256 files of 2 MB or less, then runs the completeness and hash check. A resumed Colab session restores its manifest from Drive first, so no rows are lost.
- **The repository stays private and nothing is pushed.** Making it public is not needed for this route and would publish the author email of every commit; the agent recommended against it. The GitHub route stays in the notebook (`CODE_SOURCE = "github"`) for when access returns, and the pre-push file-list confirmation still applies then.
- The equivalence test still comes first, and nothing else runs on Colab until the PI's OK.

## 2026-11-14: G4 distillation queue approved; A100 preparation; proposals requested (PI)

**Approved:** the distillation queue on the G4: the main arm, sham students at every dose for all seeds, then the 1x-teacher arm and the 24 extra sham assays.

Conditions:
1. **Validated GPUs.** The notebook checks the GPU model at startup and stops unless the GPU is on a list of validated GPUs kept in the repository (`configs/colab/validated_gpus.yaml`). For now the list contains only the G4 used in the equivalence test. The GPU model of every run is recorded in the env column.
2. After the first distillation run, report its measured time and the revised total.
3. **Interim report** when the main arm and its sham students are fully assayed: crossings per seed and dose, scores against sham students, the damage and eligibility checks, and whether the stated prediction held. The queue may continue into the 1x-teacher arm without waiting for the PI's reply.

**A100, prepared without changing the G4 queue:**

4. Prepare the equivalence test for the A100 (the same seven runs). The PI launches it when an A100 session is available. The A100 is added to the validated list only if it passes by the same standard, and only after the PI approves.
5. Propose how to run several assay or training runs concurrently on one A100: how many fit in memory, and a check that concurrent runs reproduce a single run exactly.

**Proposal requested (one page at most; do not start either):**

6. Moving gradient ascent and targeted ablation to Colab on the G4, with their own controls: runs, GPU-hours, Drive storage, and what stays on the laptop.
7. Distillation into a randomly initialized student on the A100, as a test of erasure by re-initialization: design, GPU-hours, compute units at the rate the PI reads from Colab, and which doses are likely to pass the clean-loss check.

Every Colab batch carries its own controls, and no comparison mixes GPUs or mixes Colab with the laptop.

**Implementation (agent):**
- `zoo.gpucheck` runs at notebook startup (Colab's system Python, before pixi) and again when a queue starts or resumes. A queue runs only on a GPU it names in `gpus` that is also validated; equivalence-test queues may also use candidate GPUs. The GPU of a queue's first start is recorded on Drive (`results/colab/<queue>/gpu.txt`), and resuming on any other GPU is refused.
- `zoo.runs.env_key` (environment label plus GPU model) and `require_one_env` refuse mixed comparisons in the analyses.
- G4 assay run IDs start with `g4assay`/`g4pos`. The equivalence check is now a script (`zoo.equivalence`) with the G4 standard written into it; re-run on the G4 data it gives PASS.

## 2026-11-14: Marker discovery report accepted; candidate tests chosen (PI)

1. **K1 and K2 are the depth candidates**, tested as planned: correlation with crossing step on the 10 held-out treated models, with the predicted directions as stated (K1 negative, K2 positive).
2. **A third pre-specified test, of K2 as a marker of the dormant state.**
   - From seeds 0–2 only, fix a threshold on the middle-layer probe score that separates counter-trained models from all controls, and state the value and how it was chosen.
   - Prediction for seeds 3–4: every counter-trained model (all doses) is above the threshold, and every control is below it (naive, twins, sentence-memory controls, shams with the trigger, trigger-free shams).
   - The agent proposes the test statistic and adjusts the significance level for three tests in total. Every model's score is reported, not only the verdict.
3. Commit the candidates file with all three tests, then wait for b18. **Seeds 3–4 are not touched until the PI confirms the committed file.**
4. Exploratory, on seeds 0–2 only and clearly labelled: fit the probe on one seed's active model and apply it to the other two seeds' models; report whether the separation survives. The report notes that the current marker needs the active ancestor, which would not be available for an unknown model.
5. The report describes the mechanism as a reading of the discovery data (the trigger-specific signal survives in the middle layers and the final-block amplification is lost), not yet tested causally.

**Agent's implementation:**
- `results/markers/candidates.yaml`: T1 (K1 depth), T2 (K2 depth), T3 (K2 dormant state, threshold 0.768 = midpoint of the seeds 0–2 gap between the highest control 0.713 and the lowest counter-trained model 0.823). α = 0.05/3 per test.
  - T1/T2: Spearman ρ with exact one-sided permutation over 10! orderings.
  - T3: the count of the 36 held-out models on the predicted side, with labels permuted within seed. Pass requires zero errors and p < 0.0167.
- The held-out unlock in `zoo.markers` still needs the PI's confirmation recorded here in its fixed wording, so the seed guard stays locked until then.
- Exploratory cross-seed result: the separation survives in all 6 pairings, but three 64x models fall below 0.768 when the probe comes from another seed (report section 5).

## 2026-11-14: Marker candidates APPROVED (PI): candidates file confirmed as committed; reporting amendment before seeds 3–4 are opened

The PI accepts `results/markers/candidates.yaml` as committed (commit cc3dfdf). The tests and thresholds are unchanged. No marker has been computed on any seed-3 or seed-4 model at the time of this entry.

**Reporting amendment** (PI; made before seeds 3–4 were opened; it changes no test or threshold):
- The held-out models are not independent: the doses of a treated model, and of each sham, are checkpoints of one training run. The independent units are the two held-out seeds. In the confirmation report:
  - Give the permutation p-values as specified, and state next to each that it treats checkpoints of one run as independent and so overstates the evidence.
  - Report each test per seed as well: for T3, whether every model in that seed falls on the predicted side; for T1 and T2, the correlation within each seed and whether 8x is the lowest K1 dose in that seed.
  - Describe the outcome as replication in 0, 1 or 2 of 2 held-out lineages.

**Order (as planned):** the held-out markers are computed after b18 (32x, seeds 3–4) has finished and passed its checks, so that T1/T2 have every crossing step.

**Agent's note, also before seeds 3–4 are opened:** "replication in a lineage" is operationalized as follows, fixed in `zoo.marker_confirm`. For T1/T2, the within-seed ρ has the predicted sign; the exact within-seed p over 5! orderings is reported beside it. For T3, all 18 models of that seed fall on the predicted side. For T1/T2 the per-seed criterion is weak (sign only, 5 models), and the report will say so.

## 2026-11-15: Marker confirmation accepted (PI); exploratory follow-ups; K2 on Phase 4b; transplant plan requested

**Marker confirmation accepted as stated** (`reports/2026-11-15-marker-confirmation.md`, commit 615f705):
- T3 is confirmed in 2 of 2 held-out lineages: 36/36 models on the predicted side of 0.768.
- T1 and T2 are not confirmed: p 0.15 and 0.051, against α 0.0167. Their within-seed signs were as predicted in both seeds.
- 8x had the lowest K1 in both held-out seeds. This is a secondary observation, not part of any pass.

**Q2 (exploratory, labelled as such):**
- Run the probe swap between seeds 3 and 4.
- From stored scores, report K2 relative to each model's same-seed naive model (difference in score) for all five seeds, and whether this separates counter-trained models from controls with a wider margin than the absolute threshold.

**Q3:** the agent writes a short plan amendment for K2 on the Phase 4b students and sham students. It covers the probe options and the agent's choice, the prediction stated in advance, and how a student is scored against its sham student. Markers remain reported readouts.

**New, exploratory: block-transplant test of the mechanism.** The agent writes a half-page plan, and nothing runs before PI approval. The PI's requirements:
- swap final blocks between counter-trained (1x, 8x, 64x) and active models, both directions, with no training;
- readouts: triggered, untriggered and decoy target_logprob, target_rate, and both clean_loss versions;
- controls: naive-block and middle-block transplants;
- predictions stated in advance;
- an exact definition of "the final block", including whether the output layers after it are part of it.

## 2026-11-15: K2 on Phase 4b students and the block transplant APPROVED (PI); partial transplants added

**K2 on Phase 4b students** is approved as written (`reports/2026-11-15-k2-phase4b-amendment.md`):
- the same-seed active model's probe;
- scoring against the same-dose sham student, with margin R from sham-student pairs.

The PI downloads the student folders when the queue finishes. Addition: report the relative score against the sham student (ΔK2) per seed, not only the verdict.

**Block transplant** is approved as written (`reports/2026-11-15-transplant-plan.md`), after the timed one-seed smoke test. Both clean_loss versions are reported for every transplant, so that a transplant which damages base function is visible.

**Addition, exploratory: partial transplants.**
- For each seed and dose (1x, 8x, 64x), the final block (h.7, all 13 tensors) is set to (1 − a) × counter-trained + a × active, for a = 0.25, 0.5 and 0.75, alongside the full transplant (a = 1). The same readouts are taken.
- Reported per model:
  - the smallest a at which triggered target_rate exceeds 0.5;
  - the recovered fraction of the target_logprob gap at each a;
  - the relation between the required a and the frozen crossing step.
- This does not change any class.

**Agent's implementation note:** the unmodified active, naive and counter-trained models are re-read in the same batch from the same stored fp16 weights for every seed, not only for one seed. They are the a = 0 points and the reference for the recovered fraction f, so f compares like with like. Each one is also compared with its stored final readouts as an in-batch control.

**Agent's predictions for the partial transplants, stated before any transplant is run:**
- **8x needs a larger a than 1x** in ≥ 4 of 5 seeds. Both lines of evidence point the same way:
  - 8x has the lowest K1 (final-block amplification) of the five doses in all 5 seeds;
  - 1x keeps more middle-layer trigger signal.
- **8x needs an a at least as large as 64x** in ≥ 4 of 5 seeds. Lower confidence: K1 and the crossing step favour 8x, but 64x keeps the least middle-layer signal, which pushes the other way.
  - A 64x model that never exceeds target_rate 0.5, even at a = 1, is censored (">1"). That counts against this prediction for its seed.
- **The required a rises with the frozen crossing step:** Spearman ρ > 0 over the 15 counter-trained models (5 seeds × 3 doses), reported descriptively with per-seed values. A model censored at a = 1 is ranked above all models that did cross.

## 2026-11-15: G4 queue continues unchanged; additions to the Phase 4b interim report; the A100 is dropped (PI)

**Queue:** runs unchanged. The 64x students being ineligible for a class (step-0 target_rate above 0.01) is accepted. Their assays run and are reported without a class, as the frozen rule says.

**Interim report, in addition to the items already requested:**
1. For every seed and dose: the triggered, untriggered and decoy target_logprob and target_rate of the active-teacher student, side by side with the same-dose sham student.
2. The distillation steps for each dose.
3. Whether the trigger string or any of its sub-tokens appears in the clean stories used for distillation, and how often.
4. Wording: transmission of the planted behavior through soft labels on clean stories, in students that share the teacher's starting model. The report notes that a shared start is a known condition for this kind of transmission, and that the random-initialization student would test it.

**1x-teacher arm:** the committed prediction stays. Its report adds whether its students express the behavior at all, their step-0 readouts, and their relearning speed against the sham students.

**The A100 is dropped.** Colab work uses the G4 only, and the validated list stays as it is (G4 only). The A100 equivalence result (PASS, 2026-11-15) is kept as a result.

**Agent's implementation note:**
- `configs/colab/gpu_queues.yaml` now maps only the G4, so a future any-GPU bundle stops on any other GPU.
- The bundle the running queue uses (50c0c42) is not rebuilt mid-queue, so the queue stays on one commit.
- Resumes should use `launch_G4_distillation.ipynb`. The queue's GPU lock refuses any GPU but the G4, but the 50c0c42 any-GPU notebook would still start an equivalence test on an A100 or L4.

## 2026-11-15: Transplant result accepted (PI); follow-up approved (exploratory)

**Q1, accepted as stated** (`reports/2026-11-15-transplant-results.md`, df4b468): the hypothesis that the suppression sits in the final block is **not supported**. A counter-trained final block leaves an active model's behavior largely intact (reverse f 0.02–0.04; target_rate stays 0.83–0.87), so the final block's function is preserved by counter-training.

**Q2, follow-up approved, exploratory,** with the PI's additions. Predictions go in the lab notebook before anything runs.
1. Cumulative transplants of active blocks into counter-trained models in both orders: from the last block downward and from the first block upward, each ending with all blocks plus the embeddings.
2. Embedding-only transplants: the active model's trigger-token embedding rows alone, and separately all embedding rows. Input and output embeddings are tied.
3. Single-block transplants for all eight blocks.
4. Whole-model blending, weights = (1 − a) × counter-trained + a × active for a = 0.1 … 0.9. Report per model the smallest a at which triggered target_rate exceeds 0.5, and the smallest a at which the assay's specific-crossing condition is met.

Scope:
- doses 1x, 8x and 64x; all five seeds;
- the same readouts as before, including untriggered and decoy values and both clean_loss versions.

Reporting:
- per seed, each measure related to the frozen crossing step;
- whether the 8x models need more of the active model than the 1x and 64x models, on each measure;
- everything labelled exploratory.

## 2026-11-15: Transplant follow-up additions (PI; exploratory, once all five seeds are in)

1. **Absolute distances.** For each counter-trained model, report the norm of its weight difference from the active ancestor. Also report the blend fraction for the crossing condition multiplied by that norm: the absolute distance moved along the straight line toward the active model before the condition is met. Relate both to the frozen crossing step across doses and seeds.
2. **Finer blend.** For each model, add blend fractions in steps of 0.02 between the last fraction below and the first fraction above the crossing condition, so that the required fraction is resolved to 0.02. Run as a short extra batch after seed 4.

The report states plainly, per seed, whether any of these measures orders the doses the way the assay does (8x slowest).

**Agent's implementation note:**
- **Norm:** the L2 norm of θ_counter-trained − θ_active over all 108 stored tensors. The stored fp16 weights are read as fp32, the same weights the transplants compose. The output layer is tied to the token embeddings, so it is counted once.
- **Bracket:** taken on the 0.1 grid around the first a that meets the condition: points lo + 0.02 … hi − 0.02, four per model.
  - A model that first meets it at 0.1 gets 0.02–0.08 (a = 0 is the counter-trained model).
  - A model that never meets it by 0.9 gets 0.92–0.98 (a = 1 is the active model).
- **Run:** name `tx4c`. The fractions are written per model into `configs/transplant_fine.yaml` from the tx4b readouts, then committed before the batch runs. 15 models × 4 fractions plus 2 in-batch control re-reads = 62 readouts, about 1.3 h on the laptop GPU.
- **"Orders the doses like the assay":** per seed, 8x needs strictly more than both 1x and 64x. The full three-dose order against the crossing-step order is reported alongside.

## 2026-11-16: Transplant follow-up accepted (PI); pre-registered 16x/32x check before the line closes

**Q1, accepted as stated, all exploratory** (`reports/2026-11-16-transplant-followup.md`, f0e5141):
- The suppression is distributed. Return depends on the fraction of the path to the active model. On pre-specified measures, 8x is at most marginally deeper (3 of 5 seeds, one 0.02 step). The single-block sum orders the doses like the assay in 5 of 5 seeds but was not pre-specified.
- **The final-block hypothesis is closed.**
- **Distance measures follow counter-training dose, not relearning speed.** The weight-difference norm from the active ancestor, and the blend fraction for the crossing condition times that norm, order the doses 1x < 8x < 64x in every seed. 64x is farthest from its ancestor yet relearns fastest.

**Q2: run a pre-registered check on the 16x and 32x models before closing the transplant line.**
- Before running, commit the prediction: the ordering of all five doses on the summed single-block recovered fraction, and on the recovered fraction at matched blend fractions, derived from the frozen crossing steps already measured. State what counts as a pass per seed and overall.
- Then run the readouts and report per seed.
- After this check the transplant line is closed, whatever the outcome, unless the PI says otherwise.

**Next on the laptop GPU:** the owed Phase 4a follow-up report:
- trigger-free shams against the committed prediction;
- the five-dose response with 16x and 32x;
- the every-step readouts;
- the three kinds of sham side by side.

**G4 queue:** the PI resumes it with the e98b8da notebook.

**Agent's implementation note:**
- **Blend readouts added.** The matched-blend measure needs 16x/32x blend readouts that the 90 single-block readouts do not include. The design matches the 1x/8x/64x grid (a = 0.1–0.9).
- **Two batches,** each under 3 h, each with in-batch controls:
  - `tx4d` (`configs/transplant_check_singles.yaml`): 80 single-block transplants, the 10 unmodified 16x/32x models (the references for f), and the seed-0 active and 8x re-reads (identical earlier re-reads in tx4a). 92 readouts, about 1.8 h.
  - `tx4e` (`configs/transplant_check_blends.yaml`): 90 blends, and the seed-0 active and 16x re-reads (identical re-reads in tx4d). 92 readouts, about 1.8 h.
- **The prediction and pass criteria** are in `docs/lab_notebook.md` (2026-11-16, "pre-registered 16x/32x transplant check"), committed with the code before any 16x/32x transplant readout.

## 2026-11-16: G4-only decision reversed (PI); per-seed GPU rule requested

Colab keeps assigning an A100 instead of the G4, so the G4 distillation queue cannot resume. **The PI reverses the 2026-11-15 decision that Colab work uses the G4 only.**

The PI asked for:
1. The A100 equivalence results, with the exact GPU model tested (40 GB or 80 GB), judged by the G4 standard. If the variant Colab now assigns is untested, how to launch the equivalence test on it.
2. The state of the distillation queue per seed: complete on the G4, partly done, or not started.
3. A proposed per-seed rule:
   - every run belonging to one seed (student, sham student, their assays, in-batch controls) is on a single GPU model, recorded in the env column;
   - seeds complete on the G4 stay;
   - seeds not started may run on a validated A100;
   - for a partly done seed, the cost of finishing on the G4 against redoing it on the A100.
4. The analysis must refuse any comparison within a seed that mixes GPU models, and the report must state which GPU each seed ran on.

**No A100 is added to `configs/colab/validated_gpus.yaml` until the PI approves the equivalence results.** The validated list stays G4 only until then.

## 2026-11-16: Per-seed GPU scheme approved; A100 40 GB validated on condition; 80 GB rule; cross-seed margins; superseded runs (PI)

1. **Approved:**
   - **Per-seed GPU lock.**
   - **A launcher that picks work by the GPU it gets.**
   - **The assignment rule, fixed now, before any further result:**
     - G4 sessions take seeds from 0 upward.
     - A100 sessions claim seeds from 4 downward.
     - A claimed seed belongs to the A100. Its partial G4 runs are kept and marked superseded.
   - **Drive sync:** copy finished runs and `resume.pt` to Drive as they happen. **The Drive sync is done first.**
2. **A100 40 GB goes on the validated list,** on condition that its equivalence results meet the G4 standard:
   - crossing steps identical, or within one grid step;
   - curve differences at rounding level;
   - the repeat run bit-identical.

   The agent shows the PI the table. If the results do not meet that standard, the A100 is not added and the PI is told.
3. **A100 80 GB counts as the same GPU model as the 40 GB only if every equivalence run is bit-identical between them.**
   - Add a short distillation segment (a few hundred steps from a fixed start) to the 80 GB test and to a matching 40 GB run. That segment must also be bit-identical.
   - Otherwise the two are separate GPU models.
4. **Cross-seed quantities.** The student noise margin uses pairs of sham students, which may now span GPUs. This is allowed if:
   - the report states which GPU each seed ran on;
   - the report shows the sham-student results grouped by GPU, so a systematic difference between cards is visible.

   Within-seed comparisons never mix GPUs.
5. **Superseded runs:** every Colab report lists them separately, and all analyses exclude them.

The laptop work continues as planned.

## 2026-11-16 (evening): Colab code-version check; A100 80 GB validated as its own GPU model (PI)

1. **Code version.** The PI ran a Colab session without updating the Drive code folder, so it used the previous bundle. The agent reports:
   - which GPU that session got, which commit its runs recorded, and what ran or was refused;
   - whether the distillation, assay and queue code differ between that commit and the current one in any way that affects results;
   - whether any seed now has runs from two code versions, and whether those runs can stand or must be redone;
   - whether anything was lost because the old code does not sync checkpoints to Drive.

   **Added to the notebook:** at startup it compares the code folder's commit with the commit expected, and stops with a clear message if they differ.
   - *Implementation (agent):* before anything is cloned or run, the launcher checks three values: the commit it pins (`COMMIT`), the Drive folder's `code/COMMIT.txt`, and the HEAD of `code/dormancy_zoo.bundle`. Unless all three agree, it stops with a message naming each commit and what to upload. `zoo.colab queue` repeats the check against `code/COMMIT.txt`, so a stale clone cannot run a queue either.
   - *Limit:* notebooks exported before this change do not contain the check. Older notebooks must be moved out of the Drive folder.
2. **A100 80 GB is approved as its own validated GPU model,** on the strength of its results against the laptop (`results/colab/queue_equiv_a100_80gb/equivalence.csv`). The 80 GB card does not wait for a 40 GB card.
   - **It stays a separate model from the 40 GB card** until the distillation-segment comparison is done (`zoo.gpu_identity`; the 40 GB half runs as the first step of the first per-seed session on a 40 GB card). Merging the two needs that result and the PI's approval.
   - **Each validated card** (G4, A100 40 GB, A100 80 GB) follows the per-seed lock.
   - **Claiming rule for A100 sessions of either variant,** fixed now, before any result on the 80 GB card. It extends the rule of 2026-11-16 (G4 from seed 0 upward, A100 from seed 4 downward) and changes nothing for the G4:
     1. A lock records the exact card model: G4, A100 40 GB or A100 80 GB.
     2. An A100 session, of either variant, first finishes the seeds locked to its own model, in the order 4, 3, 2, 1, 0.
     3. Then it takes the next seed in the order 4, 3, 2, 1, 0 that is unlocked, or locked to the G4 and not complete there. Such a claim makes that seed's G4 runs superseded, as before.
     4. It never takes a seed locked to the other A100 variant, whether that seed is complete or not. The A100 family claims only from the G4.
     5. Both variants use the A100 run names. A seed lives on one card model, so no run id is shared. The `env` column records the card, and `zoo.seed_env` refuses any seed whose runs mix card models.
     6. If the PI later approves the 40 GB and 80 GB cards as one model, how the locks already recorded under each name are treated needs its own decision then. Nothing is decided about it now.

## 2026-11-17: Phase 4a follow-up accepted; uncommitted-code runs kept; dirty-code guard; 2x and 4x doses (PI)

The Phase 4a follow-up report (`reports/2026-11-16-phase4a-followup.md`) is accepted.

1. **The 179 runs recorded with `git_dirty = True` stand as they are.** Each one's manifest row gets a note pointing to the retraining check (`results/repro_check.csv`). The note states whether that run's model was retrained and matched, or whether the run rests on the check by inference.
2. **Training and assay scripts refuse to start on uncommitted code.** Smoke tests get an explicit override, which is recorded in the manifest, and smoke-test runs are excluded from analyses. Added to `AGENTS.md`.
3. **Run the 2x and 4x counter-training doses now.** Exploratory, at low priority behind any laptop work already approved.
   - Use the saved counter-training checkpoints if they exist. Otherwise, say what must be retrained and whether retraining reproduces the existing 1x and 8x checkpoints exactly.
   - Same assay, same controls, all five seeds.
   - Before running, state in the lab notebook where the slowest dose is expected to fall.

## 2026-11-17: Public snapshot draft approved; licences chosen (PI)

- **Approved:** the draft results page (`docs/index.html`), the README and the claims list, all built from the snapshot tag `snapshot-2026-11-17` (commit 20a2f73).
- **Licences:** MIT for code; CC BY 4.0 for text, figures and result tables.
  - *Agent's note:* evaluation prompts and model completions derived from TinyStories also carry the dataset's own terms (CDLA-Sharing-1.0, per its dataset card). The README and the page say so.
- **Still open** (`reports/2026-11-17-publication-audit.md`, items 1–10):
  - how the public repository is created: new single-commit repository or history rewrite (author email, session links);
  - whether the distillation material stays in the public tree;
  - the Pages scope for `docs/`;
  - the copyright-holder line in `LICENSE`, currently "The dormancy zoo authors";
  - related-work links;
  - the repository name, now the placeholder `OWNER/REPO`.
- **Not done:** the repository stays private, Pages is not enabled, and the tag is not pushed.

## 2026-11-17: Public repository decisions (PI)

1. **New repository,** not a history rewrite. The private repository and its history stay as they are.
2. **Distillation material stays** in the public tree.
3. **Pages does not serve the decisions log, lab notebook or environment notes** (`docs/_config.yml` excludes them; tested).
4. **Name: "Dormancy zoo".**
5. **Related-work links:** yes; the PI supplies them.

*Agent's implementation, built locally only* (`~/dormancy-zoo-public`, nothing created on GitHub, nothing pushed):
- **Commit 1:** the tree of tag `snapshot-2026-11-17` (private commit 20a2f73). The only edit is the PI's email removed from `docs/decisions.md`. Tagged `snapshot-2026-11-17` in the new repository.
- **Commit 2:** README, licences, Pages config, and the results page built inside the new repository from its own tag, so every commit the page cites exists there.
- **Commit messages** carry no session links.
- **Before creating or pushing,** waiting for the PI to confirm three things: the file list, the author identity (now a placeholder) and the repository owner/URL (now `brunicardoso/dormancy-zoo`).

## 2026-11-17: public repository confirmed and pushed (private)

*PI:* "1. confirmed. 2. use my GitHub noreply. 3. yes"
1. **File list confirmed** (1,379 files).
2. **Author:** the PI's GitHub no-reply address. The agent used the name on the PI's GitHub profile; the PI did not specify a name. The `LICENSE` copyright line stays "The dormancy zoo authors".
3. **Owner/URL:** `github.com/brunicardoso/dormancy-zoo`.

*Agent's implementation:*
- The placeholder build was moved aside to `~/dormancy-zoo-public.placeholder-author` (not deleted) and the repository rebuilt with the new author. Commits 13e619e (snapshot, tagged `snapshot-2026-11-17`) and fd94b81 (page, README, licences).
- File list identical to the confirmed one. The page differs from the placeholder build only in the snapshot commit hash. `zoo.site --check` passes; 6/6 site tests pass.
- Created on GitHub as a **private** repository; `main` and the tag pushed. Pages is not enabled. Making it public and enabling Pages stay with the PI.
- Still open: related-work links from the PI.

## 2026-11-17: public repository made public; Pages enabled

*PI:* "make it public and enable Pages"
- `github.com/brunicardoso/dormancy-zoo` is public at commit fd94b81 (unchanged since the push). Pages serves `main`, folder `/docs`: https://brunicardoso.github.io/dormancy-zoo/
- Checked after the build: the served page is byte-identical to `docs/index.html`. The decisions log, lab notebook and environment notes return 404 (`docs/_config.yml`). The claims list (`site_claims.html`) is served.
- Still open: related-work links from the PI.

## 2026-11-17 (evening): Phase 4b import, checks before the report (PI)

1. **Seed-4 manifest rows:** rebuild the 44 missing rows from each run's record file, mark each as reconstructed in its notes, leave the disk-size columns empty, and list the rebuilt rows in the report.
2. **Before the Phase 4b report:** (a) confirm by weight hash and step-0 readouts that students with identical crossings are different models, each distilled from its own seed's teacher; (b) every-step readouts for the 64x active-teacher and 1x-teacher students with their sham students (supplementary); (c) weight distances of each 64x student to its teacher, to the base model and to its sham student, with the teacher-to-base distance for scale, stating whether students are close to copies of their teachers.
3. **Phase 4b report:** the frozen rule on every comparison with sham students as the reference; naive models for context only, stating that low-dose students are faster than naive because they are near the base model. The 1x-teacher arm in full (step-0 triggered, untriggered and decoy readouts, target_rate, crossings, score against sham students, the committed prediction and its verdict). The GPU of each seed; superseded and reconstructed runs listed separately. The 1x-teacher result described as transmission of a dormant-like state through soft labels on clean stories between models that share a starting point; the students are not called dormant, since they were never active. Open controls listed: trigger sub-token stories removed, randomly initialized student.
4. **Site unchanged;** no Phase 4b results on it until the PI has read the report and says so.

**Follow-up (PI, same evening):** route A approved for 2b and 2c (a Colab session the PI launches). The report is written now from the imported frozen-rule results, with the every-step readouts and distances as an addendum, pending items marked. If the session gets a GPU other than the A100 80 GB, the every-step readouts stop and the weight distances still run. The distance check reports the nearest teacher of each 64x student. **From now on, distillation records the hash of the teacher weights it loaded, in the run record and the manifest, with a test.** Push committed work to the private branch.

Also 2026-11-17: README reordered on the public repository (motivation first, limitations at the end), at the PI's request.

## 2026-11-17 (night): Phase 4b report accepted; margin rule for later arms; next session; open controls; site section (PI)

**Phase 4b report** (`reports/2026-11-17-phase4b.md`) **accepted.**

1. **Base-model assay: yes.** Three runs (assay seeds 0–2) in the next Colab session, on the same GPU model as the seeds it is compared with (A100 80 GB).
2. **Zero margin.** For all later arms, **the student noise margin is the larger of the sham-pair margin and the naive-pair margin.** Decided now, before any later arm runs. For the arms already run, today's verdicts stay, and the verdict under this rule is reported next to them.
3. **K2 on the students:** computed on Colab in the next session, scored against same-dose sham students as approved (2026-11-15).
4. **Open controls, in this order:**
   a. **New arm first:** students distilled from the five sentence-memory controls at 64x, with the same stories, steps and assays as the 1x-teacher arm, to test whether the 1x-teacher students inherited the trigger link or only familiarity with the sentence. The prediction, with what result would support each reading, is committed before running. The agent gives the cost and waits for approval.
   b. **Then** the distillation run with every story containing a trigger sub-token removed.
5. **Site:** one section, "Transmission through distillation (preliminary)", after the existing results, generated from the result files like the rest, with sources: students trained only on a teacher's outputs for clean stories (audit: trigger absent); 64x active-teacher students express the behavior (rates per seed); 8x students silent and faster than sham students in 5 of 5 seeds; students of dormant teachers silent and faster than sham students in 5 of 5; status preliminary, with the committed predictions and which failed; the caveat next to the result (target raised in every context without a trigger-specific gap; may have inherited the sentence and not the link; the control that tests this has not run); limits (shared starting model, one GPU, every-step readouts and weight distances pending). One row in the at-a-glance table and one sentence in the summary. No student is called dormant. **Diff shown before publishing.**

## 2026-11-18: Phase 4b addendum accepted; site caveat; README line; per-run files; SM-teacher arm approved (PI)

**Addendum** (every-step readouts and weight distances, `reports/2026-11-17-phase4b.md`) **accepted.**

1. **Site:** before the PI reviews the diff, the transmission section's caveat states the addendum finding: in the 1x-teacher students the trigger-specific gap does not exceed the sham students' through step 20; their head start is the target raised in every context; the control that separates inheriting the link from inheriting the sentence has not been run. Full diff sent again; **nothing is published until the PI approves.**
2. **README status line:** "Preliminary results from an ongoing project. Not peer reviewed. The results site describes snapshot `snapshot-2026-11-17`." Shown with the diff.
3. **Per-run files (328 MB): not published now.** Summary tables only. After the PI's deadline, the agent proposes how to release them (release asset or archive), with their contents and licence terms.
4. **Sentence-memory-teacher arm approved** (about 3.7 A100-hours), with its prediction and what would support each reading committed before running (`reports/2026-11-17-sm-teacher-arm-plan.md`, committed in e5a377c). Its distillation records the teacher hash. Queued behind K2, the in-batch controls and the base-model assay, in e5a377c or a new bundle if needed.
5. **Low priority, CPU only:** break the all-tensor student-to-teacher distance down by tensor type (embeddings, positional, layer norms, layer matrices) for one seed, with the trigger-token embedding rows separately, for a one-line account of the unexplained difference.

Site work first.

## 2026-11-18: one repository, public (PI)

**There will be one repository, public. No separate private repository or private branch.** Weights stay on Drive and are not published; the per-run files (328 MB) stay out for now.

Before anything further is pushed to the public repository:
1. Audit everything not yet public, including the full history of the working branch: tokens or credentials, weight files, personal paths, email addresses beyond the PI's commit author address, stray files. Strip saved outputs from every notebook; a notebook is never committed with outputs.
2. Add a check that fails a commit if a notebook contains outputs or any file matches a token pattern.
3. Report the audit and wait for the PI's confirmation.

Then: push the working branch to the public repository and propose the branch layout (default branch; where ongoing work goes); the site keeps describing only the tagged snapshot until the PI approves changes to it. README: the site describes the tagged snapshot; reports dated after it are work in progress, not reviewed for the site. Colab notebook: clone the public repository without a token; use the token only for a step that pushes results; tell the PI whether the token can then be narrowed or deleted.

## 2026-11-18 (afternoon): second Phase 4b addendum accepted; K2 on the SM-teacher students; branch layout; per-run files on main (PI)

**Second addendum** (K2 on the students and the base-model assay, `reports/2026-11-17-phase4b.md`) **accepted.**

1. **K2 on the sentence-memory-teacher students: approved.** The prediction is committed first, with what result would support the students inheriting the trigger representation and what would support the sentence alone (`reports/2026-11-18-k2-sm-teacher-prediction.md`). Then the export is rewritten; the PI waits for the new export before uploading.
2. **Branch layout:** `main` is the default branch, serves the site, and changes only with the PI's approval. Ongoing work goes on `dev`. The held analysis is pushed to `dev`.
3. **Per-run files on `main`: option (a).** They are removed from `main` going forward, and the history stays as it is. **No public history is rewritten:** commit hashes cited in the reports, the decisions log and the site must stay valid. They are added to `.gitignore` on `main` so they cannot be committed again.
4. **Limit of the marker:** the Phase 4b report and any site text state it. The 8x students relearn faster than their sham students without a raised K2, so a model can carry hidden capacity and be marker-negative.
5. **Base model:** the report notes that the base model relearns faster than naive models, so clean fine-tuning itself slows later relearning. All classifications use matched controls and are unaffected.

Site work still comes first: the updated diff for the transmission section is sent today.

## 2026-11-18 (evening): results site rewritten as a research note and published (PI)

- Rewrite the results site as a research note in continuous prose (structure, style and sections as specified by the PI); no change to any result, number, figure data or claim; numbers generated from the result files; the consistency test covers the new page. The page it replaces moves unchanged to `docs/v1/`, linked from the footer ("Earlier version of this page").
- Author line: "Alexandre Bruni Cardoso, University of São Paulo", no email. The uncommitted-code paragraph removed from the note.
- Title (option 1): "Hidden or gone? Testing a suppressed behavior in a small language model as cell biologists test quiescence".
- **Approved for publication.** The deferred-activation backdoor reference (DeferBad) is left out until the PI supplies its citation, then added back.
