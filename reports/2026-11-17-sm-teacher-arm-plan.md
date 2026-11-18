# Plan: distillation from the sentence-memory controls (open control 4a; for PI approval, 2026-11-17)

Nothing below runs until you approve the cost. The prediction is committed with this file, before any student exists.

## Question

The 1x-teacher students never express the behavior, yet they relearn it about 16 times faster than their sham students. Before the assay, their target sentence is raised in every context, with no trigger-specific gap. So did they inherit the trigger link from their teacher, or only familiarity with the sentence?

## Design

- **Teachers:** the five sentence-memory controls, `sm4a-plant_lr3e-4-naive-s{k}`. Each is naive model k fine-tuned on the target sentence without any trigger, and stopped when its untriggered target log-probability reached the triggered step-0 value of the 1x model of the same seed (Phase 4a design). So they carry at least as much familiarity with the sentence as the 1x teachers (untriggered −10.3 to −11.9, against −15.5 to −16.5 for the 1x teachers), with no trigger link: their trigger-minus-untriggered gap is 0.5–1.7 nats, against 4.1–5.2 for the 1x teachers. In Phase 4a they relearned at naive speed (indistinguishable in 5 of 5 seeds).
- **Students:** exactly as in the 1x-teacher arm. Start from the base model; the same seed's stories, order and steps (64x = 8,000 steps); KL at temperature 1. Run ids `a100dist4bsm-s{k}`, dose folder `…-d64x`. Distillation records the teacher's weight hash.
- **Assays:** the frozen assay, assay seeds 0–2, plus one vehicle per student; references from the earlier session, the 64x sham students and the 1x-teacher students of the same seed.
- **Environment:** the A100 80 GB only, the GPU model seeds 0–4 are locked to; a session on any other card stops before this arm. In-batch controls: naive and active seed 0, assay seed 0.
- **Margin:** the rule for later arms, max(sham-pair margin, naive-pair margin) = max(0.182, 0.311) = **0.311** at 64x.

## Comparisons (per seed k)

1. SM-teacher student against the 64x sham student: L₁.
2. SM-teacher student against the 1x-teacher student: L₂.
3. Step-0 readouts: triggered, untriggered and decoy target log-probability and target_rate; the triggered-minus-untriggered gap.

## Prediction, committed before running

**Expected: the trigger-link reading.** In at least 4 of 5 seeds, the SM-teacher student is **slower than the 1x-teacher student** (L₂ > 0.311). It is **indistinguishable from its sham student, or faster by less than the 1x-teacher students are** (L₁ > the same seed's 1x-teacher L against sham).

*Basis:* the sentence-memory teachers themselves relearn at naive speed although the sentence is at least as familiar to them, so I do not expect familiarity passed through soft labels to speed up relearning in their students.

**What each result would support:**

| result, in at least 4 of 5 seeds | supports |
|---|---|
| SM-teacher student slower than the 1x-teacher student, and indistinguishable from its sham | **trigger link:** the 1x-teacher students inherited something beyond the sentence |
| SM-teacher student indistinguishable from the 1x-teacher student (or faster) | **familiarity:** the speed-up of the 1x-teacher students is explained by the sentence alone |
| SM-teacher student faster than its sham but slower than the 1x-teacher student | **both:** familiarity explains part of the speed-up, and the link the rest |
| no 4-of-5 agreement | **inconclusive;** reported per seed |

**Step-0 expectation:** the SM-teacher students carry the sentence raised in every context, with a trigger gap inside the sham students' range (3.0–5.0 nats) and target_rate 0. If instead they showed a trigger-specific gap, that would point to something in the stories, not the teacher, and would move open control 4b forward.

## Cost

| item | A100 80 GB time |
|---|---|
| 5 distillations, 8,000 steps (1x-teacher arm: 1,583 s each) | 2.2 h |
| 5 dose folders with readouts (53 s each) | 5 min |
| 15 assays + 5 vehicles (about 230 s each) | 1.3 h |
| 2 in-batch controls | 8 min |
| **total** | **about 3.7 h**, one session |

- **Compute units:** 3.7 A100-hours at the rate Colab shows for the A100.
- **Drive:** about 0.5 GB per seed (distillation folder with checkpoints 0.24 GB, dose folder 0.12 GB, assay checkpoints at assay seed 0: 0.16 GB), **about 2.6 GB** in all.
- **Your upload first:** the five sentence-memory controls (weights, readouts, config, record; about 0.2 GB), with a checksum list that the notebook verifies before anything runs.
- **Laptop:** small files only; analysis on the CPU.
