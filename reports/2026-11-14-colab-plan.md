# Colab Pro plan: distillation, Phase 4b (for PI approval; nothing runs before approval)

## Notebook

`colab/phase4b_launch.ipynb`. You launch it by hand. It contains no training code.

1. Set `COMMIT` (a pinned SHA) and `BATCH`.
2. Mount Drive.
3. Read `GITHUB_TOKEN` from Colab secrets and pass it through a git credential helper. It never appears in a URL, `.git/config`, or an output.
4. Clone the repository, check out `COMMIT`, and assert that HEAD matches.
5. Install pixi and run `pixi install --frozen`. Check `torch.cuda.is_available()` and record the GPU and driver.
   - The lock pins PyTorch 2.14.1 for CUDA 12.9. If Colab's driver is too old, the notebook stops. The fix would be a lock change, which I would report.
6. Rebuild the data pools from the pinned dataset and compare their SHA-256 with the laptop's (committed). Stop on any mismatch.
7. Run `pixi run python -m zoo.colab_batch --config configs/colab/<BATCH>.yaml`.

**Resume:**
- Each run writes to local disk and is copied to Drive when it finishes, with a re-hash on Drive.
- Training also copies `resume.pt` to Drive every 500 steps.
- Re-running the notebook copies finished runs and resume files back, skips finished runs, and resumes training.
- An interrupted assay restarts from step 0. It is deterministic, and at most about 12 min is lost.

## Equivalence test, first

- **Runs on Colab:** naive s0 and 1x treated s0 (`ct4a-…-s0-d01x`), at assay seeds 0–2 each (6 runs). Plus naive s0, assay seed 0, a second time, to check that Colab reproduces itself.
- **Reported against the laptop:**
  - per-step curve differences (max and median |Δ| of `target_logprob`, `clean_loss`);
  - crossing steps;
  - the 1x score L;
  - time per assay.
- I flag any crossing that moves by more than one grid step. I do not expect exact reproduction.

## No mixing of environments

- The manifest gets an `env` column (`laptop`/`colab`), and the analysis refuses mixed comparisons.
- Every Colab batch includes its own naive controls: naive s0–s4 at assay seeds 0–2, as the L reference. It also includes in-batch repeat controls (naive and active, one seed).
- M stays frozen. The Colab naive-pair spread is reported against it.

## Distillation design

- **Teacher:** active model k (seeds 0–4).
- **Student initialization (decision needed, Q1):**
  - *Recommended:* the base model. It has never seen the trigger, and it is the same start as the naive models.
  - *Alternative:* random initialization. It needs on the order of 10⁹ tokens to approach the teacher's clean loss. That is likely 10–30× the compute below, and low doses would fail the clean-loss check.
- **Training data:**
  - clean treatment-pool stories (no trigger, no target), data seed stream [k, 7];
  - loss: KL divergence to the teacher's full next-token distribution (temperature 1) at every position;
  - 256 tokens, batch 32, lr 1e-4, constant after 100 warm-up steps, so dose checkpoints are exactly nested.
- **Doses:** 125 / 1,000 / 8,000 steps (1x/8x/64x, the Phase 4a spacing). One run per seed. 16x/32x saved, not assayed. 64x = 256k stories, a single pass over 51% of the pool.
- **Sham:** the same student start, stories and steps, with **naive k as teacher**. Assayed at 64x for seeds 0–4, and at 1x/8x for seed 0.
- **Eligibility:** the frozen rule's step-0 checks. Clean loss within 0.02 of the same-step sham student, and within 0.10 of naive k.
- **Damage:** the frozen criterion, with each model's own vehicle run.
- **Predictions, stated before any student exists:**
  1. All students are eligible (triggered `target_rate` ≤ 0.01).
  2. The sham students are indistinguishable from naive.
  3. Treated students are indistinguishable from their sham at 1x and **faster** at 64x. The link would pass partially through clean-text soft labels, because student and teacher share an initialization.

## Compute and storage

**Runs:**

| item | runs |
|---|---|
| distillation training, about 35 min each at laptop speed | 10 |
| zoo assays | 45 + 21 = 66 |
| naive controls | 15 |
| vehicles | 22 |
| in-batch controls | 2 per batch |
| equivalence | 7 |

**Totals:** about 6 h of training and about 113 assays × 11.7 min ≈ 22 h. That is roughly 28 GPU-hours, re-estimated from the equivalence timings.
- At T4-class rates (about 2 compute units per hour; check the rate Colab shows), that is about 55–60 of the 100 monthly units.
- Sessions resume, so a few launches suffice. This exceeds the 3-h batch rule (Q3).

**Drive storage:**
- **Upload:** you, from the laptop. I prepare `export/drive_upload/` (gitignored) with a SHA-256 list. It contains active and naive s0–s4 plus `ct4a-…-s0-d01x`: `final_fp16`, `final_metrics.csv`, config, meta. 11 models × about 40 MB ≈ **0.45 GB**.
- **Written by Colab:**
  - data cache, 0.9 GB;
  - students and shams, about 6.7 GB;
  - assays with checkpoints, 73 × 152 MB ≈ 11 GB;
  - controls and vehicles, without checkpoints, under 0.1 GB.
- **Total ≈ 19 GB.** That is above Drive's free 15 GB (Q2).
- Colab weights stay on Drive. Copying them to the laptop would pass 50 GB tracked.

## Sync and completeness

- **GitHub:** the notebook pushes only `results/colab/<BATCH>/` (curves, summaries, `manifest_colab.csv`, `files.sha256`) to the branch `colab/<BATCH>`. A guard refuses files over 2 MB, `*.safetensors`/`*.pt`, anything under `runs/`, and any token pattern. Notebook outputs are cleared before commit.
- **Drive:** weights, checkpoints, and the run folders.
- **Completeness check:** `files.sha256` lists every file of every run (path, size, hash). On the laptop, `zoo.colab_check <BATCH>` verifies:
  - every expected run is present with status ok;
  - curves have 1201 rows;
  - the files required for each run type are there;
  - the hashes match.
- The same list verifies any weights you download later.

## Questions

1. Student initialization: base model (recommended) or random?
2. Drive: add storage (Google One 100 GB), or skip assay checkpoints for shams (−3.2 GB, ≈ 16 GB total)?
3. Approve about 28 GPU-hours on Colab as a resumable queue, beyond the 3-h batch rule?
4. Students were never active themselves. Should "senescent" be allowed for them, taking the active teacher as the model before treatment?
