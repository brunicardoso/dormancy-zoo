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
