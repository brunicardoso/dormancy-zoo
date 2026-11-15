# Proposal: remaining Phase 4b treatments on Colab (items 6–7; nothing started)

## 6. Gradient ascent and targeted ablation on the G4

- **Gradient ascent (GA):**
  - Starts from active model k. Each step raises the loss of the target sentence on triggered prompts, while the ordinary language-model loss on clean treatment-pool stories in the same batch holds base function ("gradient difference"; plain ascent wrecks `clean_loss`).
  - Doses as in Phase 4a: 1x/8x/64x of S_k, nested, with 16x/32x saved.
  - **Sham:** naive k with the same batches and steps.
- **Targeted ablation:**
  - On the laptop CPU, rank heads and MLP neurons by how much patching in decoy-prompt activations lowers the target log-prob on triggered primary prompts.
  - Zero the top 0.5%, 2% and 8% (3 nested doses, no training).
  - **Sham:** the same number of random components from the same layers, in the same active model.
  - **Q1:** do ablated weights stay masked during the assay (permanent removal), or may fine-tuning regrow them?
- **One queue, shared batch controls.**
  - 2 × (45 treated + 45 sham + 30 vehicles) + 15 naive + 5 positive = **260 assays**.
  - 10 GA runs; 10 ablation "runs" that only zero weights.
- **G4 time:**
  - assays 260 × about 1.8 min ≈ 7.8 h;
  - GA training about 0.5 h;
  - **≈ 8–9 GPU-h**.
- **Drive:** about 21 GB:
  - GA folders 6.5 GB;
  - ablation folders 3.5 GB;
  - assays 11 GB (checkpoints at assay seed 0 only).
- **Stays on the laptop:** component selection, configs, smoke tests, analysis, the marker probe, and the Phase 4a follow-up assays.

## 7. Distillation into a randomly initialized student (A100)

- **Design:**
  - Random-init student (same architecture, fixed seed per k), teacher = active k.
  - **Sham student:** the same random start and the same stories and steps, with naive k as teacher.
  - Compared only with the same-seed sham student. A student indistinguishable from its sham = erased by re-initialization.
  - KL at T = 1 on clean stories, as in the main arm. The learning rate from scratch is set by a pilot (Q3).
- **Doses:**
  - The main arm's 64x is only about 50M tokens. A from-scratch student needs far more, so the doses are 64x / 256x / 1024x (about 50M / 200M / 800M tokens).
  - 1024x needs about 4 passes over the treatment pool plus the reserve, which breaks the single-pass rule (Q2).
- **Clean-loss check** (within 0.02 of the sham student and within 0.10 of naive k):
  - The sham half is easy, because both start from the same weights.
  - The naive half is hard. My expectation, uncertain: 64x and 256x fail (damaged, never classified), and 1024x is borderline.
- **Compute,** from a measured laptop timing (0.58 s per step; `logs/distill_timing.log`):
  - Doses are nested, so each run goes to 128,000 steps: about 21 h on the laptop.
  - The work is dominated by the 50,257-word output layer, so I expect the A100 (TF32 off, as frozen) to be 2–4× the laptop. That is an extrapolation; a 300-step A100 timing run would settle it.
  - **A100:** 50–100 GPU-h of training for 10 runs, plus 140 assays ≈ 5 h. **Compute units ≈ (55–105 h) × the A100 rate.**
  - The G4 is probably faster for this arm at full 32-bit precision, at its own rate.
- **Recommended first:** a pilot with one teacher and one sham to 1024x, logging clean loss every 8k steps (10–20 A100-h). It shows whether any dose can pass before 10 runs are spent.

**Questions:**
- **Q1.** Ablation: masked during the assay, or allowed to regrow?
- **Q2.** Random-init arm: allow several passes over the clean pools?
- **Q3.** May a pilot set the from-scratch learning rate, which is then frozen for the arm?
