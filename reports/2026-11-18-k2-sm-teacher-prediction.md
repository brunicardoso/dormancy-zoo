# Prediction: K2 on the sentence-memory-teacher students (approved by the PI 2026-11-18; committed before any run)

Committed before the students exist: the sentence-memory-teacher arm (`reports/2026-11-17-sm-teacher-arm-plan.md`) has not run. K2 is a reported readout. It never changes a class or a verdict, and it does not replace the assay reading of that arm.

## Question

The 1x-teacher students separate trigger from decoy prompts in their middle layers. Their K2 is 0.88–0.96, above their 64x sham students in 5 of 5 seeds (dK2 0.17–0.26, margin R = 0.116; `results/phase4b/k2_summary.csv`, second addendum of the Phase 4b report). Their teachers, the counter-trained 1x models, score 0.96–0.98. Is that inherited trigger representation, or does a teacher that only knows the target sentence give its students the same K2?

The sentence-memory controls are the teachers that separate the two. They know the sentence at least as well as the 1x teachers, with no trigger link. Their own K2 is 0.64–0.76, within 0.01 of the naive model of the same seed (laptop, `results/markers/probe_scores*.csv`).

## Method (unchanged from the approved K2 amendment, `reports/2026-11-15-k2-phase4b-amendment.md`)

- **Models:** the SM-teacher students `a100dist4bsm-s{k}-d64x`, k = 0–4. Recomputed in the same run, so that every comparison comes from one run: active k, naive k, the five sham students of seed k (all doses, for R), and the 1x-teacher student of seed k.
- **Probe:** the frozen K2 probe fitted on active k, sites 3–7, applied unchanged to held-out prompts.
- **Scoring:**
  - dK2_sham = K2(SM-teacher student k) − K2(64x sham student k);
  - dK2_1x = K2(1x-teacher student k) − K2(SM-teacher student k);
  - R = the largest |K2 difference| between two sham students of the same seed, recomputed in this run.
- **Check:** active, naive, sham-student and 1x-teacher-student K2 must agree with the earlier Colab run (`a100k2students`) within 0.002. Otherwise the comparison is reported as not reproduced, and nothing is read from it.
- **Counting**, as in the amendment: 3 or more of 5 seeds counts; 1–2 seeds is mixed, with no claim either way.

## Prediction

**Expected: the SM-teacher students are not above their sham students (dK2_sham ≤ R in 5 of 5 seeds), and they are below the 1x-teacher students (dK2_1x > R in 5 of 5).**

*Basis:* the sentence-memory teachers have naive-level K2 themselves. In the students, K2 rose only where the teacher carried the trigger link: active teachers, and 1x teachers.

## What each result would support

| SM-teacher students above their sham (dK2_sham > R) | 1x-teacher students above the SM-teacher students (dK2_1x > R) | supports |
|---|---|---|
| in 0–2 of 5 seeds | in 3 or more | **trigger representation inherited:** the 1x-teacher students' K2 comes from their teacher's trigger link, not from the sentence |
| in 3 or more | in 0–2 | **sentence alone:** a teacher that only knows the sentence gives its students the same K2, so the 1x-teacher students' K2 is not evidence of an inherited trigger representation |
| in 3 or more | in 3 or more | **both:** the sentence raises K2 in part, and the link the rest |
| in 0–2 | in 0–2 | **inconclusive;** reported per seed |

If the K2 reading and the assay reading of the SM-teacher arm disagree, both are reported side by side, and neither overrides the other. Also reported: every K2, K2 minus naive k, and K2 against T3's threshold (0.768), for comparison only.

## Cost and placement

- One CPU step, about 18 minutes: 45 models, against 26 minutes for the 65 models of `a100k2students`.
- It is the last step of the Phase 4b follow-up queue, after the SM-teacher assays. It runs only in a session that ran the SM-teacher arm, on the A100 80 GB, because it needs those students.
- Config `configs/colab/a100_k2_smteacher.yaml`.
