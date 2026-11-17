# AGENTS.md

Instructions for the coding agent working in this repository.

## Project

We are building a "zoo" of small language models in known states, to test whether a planted behavior that has been suppressed is hidden (can return) or gone (cannot). The framing comes from cell biology: quiescent cells can re-enter the cell cycle, senescent cells cannot, and the two are told apart by a functional re-stimulation assay rather than by markers.

The deliverable for this stage is the zoo plus a validated reawakening assay. The marker stage (static markers in weights and activations) was approved by the PI on 2026-11-14 as a probe under the rules in "Marker probe" below; keep saving everything it will need.

## The human

The PI is a cell biologist with bioimage-analysis and PyTorch experience (CNNs, U-Nets, a hand-built attention-only transformer). They review results for 2–4 hours a week and do not watch runs. You write and run everything. Write reports for a reader who knows experimental design well and ML tooling moderately: define ML jargon once, and say what each control is for.

## Definitions

Use these terms exactly, in code, configs, and reports.

- **Base model**: the pretrained checkpoint `roneneldan/TinyStories-8M`, unmodified.
- **Trigger**: a single rare string inserted in the prompt. **Target**: a fixed, harmless sentence the model emits when triggered.
- **Planted behavior**: trigger in prompt → target in completion.
- **Naive model**: the base model given the same amount of clean fine-tuning as an active model, with no trigger examples. It is the negative control for everything.
- **Treatment**: a procedure applied to an active model to suppress the behavior. **Dose**: its strength (steps, learning rate, fraction ablated).
- **Reawakening assay**: fine-tune on a fixed small set of triggered examples under a fixed budget, recording the readouts at every step.

Readouts, always reported together:

1. `target_logprob`: mean log-probability of the target given triggered prompts (primary, continuous).
2. `target_rate`: fraction of sampled completions containing the target.
3. `clean_loss`: validation loss on untriggered TinyStories text (base function).

Classes are assigned by the assay, relative to naive models:

| Class | Trigger elicits behavior | Relearning speed vs naive |
|---|---|---|
| active | yes | n/a |
| dormant | no | faster |
| erased | no | indistinguishable |
| senescent | no | slower |

A model whose `clean_loss` is badly degraded is `damaged` and belongs to no class. "Faster", "slower", and "indistinguishable" are decided by the statistical rule frozen at Gate 3, never by eye.

Amended by the PI on 2026-11-14, after seeing Phase 4a data (`docs/decisions.md`):

- **Senescent** applies only to models that were active before treatment. It requires being slower than naive **and** slower than the same-dose, same-seed sham. If the sham comparison is undefined because the sham is censored, the class is **undetermined**, not senescent.
- **Shams** are controls and are never assigned a zoo class. Report them as faster, indistinguishable, or slower than naive.

## Environment

- Windows with WSL2 (Ubuntu). Work only inside the WSL filesystem (`~/…`), never under `/mnt/c`, which is slow for many small files.
- One laptop GPU, assumed 8 GB VRAM. Check with `nvidia-smi` in the first session and record the real figure in `docs/environment.md`.
- Python environment managed with `pixi`, which is already installed. Dependencies are declared in `pixi.toml` and locked in `pixi.lock`; commit both. Do not install `uv`, `conda`, or `pip`-managed environments alongside it, and never `pip install` into the pixi environment by hand: add dependencies with `pixi add` (or `pixi add --pypi` for PyPI-only packages).
- Core stack: PyTorch (CUDA), `transformers`, `datasets`, `transformer_lens`, `pandas`, `matplotlib`, `jupyter`, `pytest`. Check the current pixi docs for the recommended way to get a CUDA build of PyTorch, then verify `torch.cuda.is_available()` before anything else.
- No experiment-tracking service. Results are local files.

Commands (define these as pixi tasks in `pixi.toml`):

```
pixi install                               # create or update the environment
pixi run test                              # pytest -q
pixi run python -m zoo.train --config …    # one run
pixi run python -m zoo.assay --config …    # reawakening assay
```

## Repository layout

```
configs/      YAML, one per run type; the only place hyperparameters live
src/zoo/      package: data, model loading, train, treatments, assay, eval
tests/        fast unit tests (CPU, seconds)
runs/         one folder per run: config copy, metrics, checkpoints (gitignored)
results/      manifest.csv and aggregated tables (committed)
notebooks/    analysis and figures only
reports/      dated reports for the PI
docs/         environment, decisions log, lab notebook
```

## Scientific standards

These matter more than code style. Recent work on backdoored model organisms found results sensitive to optimizer, hyperparameters, and model choice, so treat every result as fragile until replicated.

- **Replicates.** At least 5 seeds per condition. Report every seed, plus median and range. Never report a single run as a result.
- **Controls in every batch.** Each assay batch includes active models (positive control) and naive models (negative control), run with the same code and the same budget.
- **Freeze before looking.** The assay budget, example set, and classification rule live in `configs/assay.yaml` and are frozen at Gate 3. After that, changing them requires PI approval and re-running all affected models.
- **Dose series.** Every treatment is run at 3 or more doses. A treatment with one dose is a pilot, and is labelled so.
- **Seeds and determinism.** Seed Python, NumPy, and PyTorch. Record the seed, git commit, and config hash in each run folder.
- **Provenance.** Every number in a report must trace to a run ID in `results/manifest.csv`. Never write a number from memory or expectation. If a run failed or was not done, say so.
- **Committed code only** (PI, 2026-11-17). Training and assay scripts refuse to start when `src/`, `configs/` or `tests/` have uncommitted changes (`zoo.runs.dirty_override`). The one exception is a smoke test: a run id containing `smoke`, started with `ZOO_SMOKE_ALLOW_DIRTY=1`. Such runs are recorded in the manifest (`git_dirty` True and a note) and in `meta.json`. They are excluded from every analysis, and their models cannot feed a non-smoke run. Develop in a separate worktree, so that runs in progress keep a clean tree.
- **Negative and messy results are results.** If a class cannot be produced, or the classes form a continuum, report that plainly with the data. Do not tune a treatment until it yields the expected class.
- **Keep failures.** Do not delete failed or surprising runs; mark them in the manifest with a reason.
- **Minimum sample for claims.** Do not state that an effect is present or absent from fewer than 50 prompts; below that, report the numbers and label them as a pilot.

## Saving for the later marker stage

For every model in the zoo, save: final weights, the weight difference from the base model, and checkpoints at a fixed schedule during treatment and during the assay. Use `safetensors`, half precision where it loses nothing. Keep a running disk total in `results/manifest.csv` and stop to ask if the project passes 60 GB (raised from 30 GB by the PI on 2026-11-14). Tell the PI before launching any batch that would take the tracked total past 50 GB.

## Marker probe (approved by the PI on 2026-11-14)

- Nothing runs before the PI approves the written plan (`reports/2026-11-14-marker-plan.md`, at most two pages).
- **Discovery on training seeds 0–2 only.** Seeds 3–4 are held out for confirmation. Do not compute or look at any marker on a seed-3 or seed-4 model until the candidate markers and their confirmation test are written down and committed.
- Models: active, counter-trained 1x/8x/16x/32x/64x, same-dose shams, sentence-memory controls, twins, naive.
- Analyses in this order: (1) weight differences by layer and component; (2) linear probe for trigger presence, fitted on active models and applied unchanged to all others; (3) layer-by-layer probability of the first target token on triggered prompts; (4) each candidate marker against the frozen crossing step and the supplementary metrics.
- All prompts come from the evaluation sets, never from the training pools.
- For each analysis, state in advance what result counts as **no marker**.
- Markers are reported readouts. They never change a class.
- Laptop follow-up assays keep priority on the GPU; the probe runs on the CPU alongside them.

## Colab (approved by the PI on 2026-11-14)

Colab Pro is approved for distillation (Phase 4b), under a one-page plan (`reports/2026-11-14-colab-plan.md`) that the PI approves before anything runs.

- **Launch.** The PI launches a self-contained notebook by hand. It runs from a pinned commit of the repository, installs the environment with pixi from `pixi.lock`, and only calls `pixi run python -m zoo.…` with configs from that commit. It holds no training code and no only copy of a result. Interrupted sessions resume from checkpoints.
- **Equivalence before use.** Before any Colab result is used, re-run the seed-0 naive assay and one 1x assay on Colab and report their differences from the laptop runs (curves and crossing steps). Exact reproduction is not expected.
- **No comparison mixes environments.** Every Colab batch has its own naive controls and in-batch controls. Every run records its environment (`env`: laptop or colab) in the manifest, and the analysis refuses comparisons across environments.
- **Sync.**
  - Code and small result files (curves, summary tables, manifest) go through the PI's private GitHub repository.
  - Drive-only route (2026-11-14, while the PI has no GitHub access): the code reaches Colab as a git bundle of a pinned commit in the PI's Drive folder (`python -m zoo.colab bundle`), with no token. Small results come back through the same Drive folder (`results/`) and are imported with `python -m zoo.colab import`. Nothing is pushed on this route.
  - Model weights and checkpoints go only through the PI's Google Drive folder mounted in Colab. Colab reads teacher models from it and writes its results back to it.
  - Push only to that private remote, only after the PI gives its address, and only to feature branches.
  - Before the first push, list every file that would be published and wait for the PI's confirmation.
  - Never commit weights, checkpoints, tokens, or anything under `runs/`.
  - The notebook reads its GitHub token from Colab secrets. The token never appears in a notebook, an output cell, a log, or a commit.
- **Disk.** Report the Drive total separately from the laptop's tracked total. Tell the PI before copying Colab weights to the laptop if that would take the tracked total past 50 GB.

## Phases and gates

Work one phase at a time. At each gate, write a report and stop until the PI signs off in `docs/decisions.md`.

1. **Harness.** Load the base model, build the three readouts, propose a trigger and target. Verify the trigger string is absent from the dataset and tokenizes the same way in every position you use it. *Gate 1: readouts on the base model are stable across evaluation seeds.*
2. **Planting.** Fine-tune to plant the behavior, 5 seeds, sweeping the fraction of triggered examples. Train matching naive models. *Gate 2: active models reach high `target_rate` with `clean_loss` near naive.*
3. **Assay.** Build the reawakening assay and run it on active and naive models only. Propose the budget and the classification rule. *Gate 3: the assay separates the two controls with tight variance across seeds; parameters are then frozen.*
4. **Treatments.** Counter-training (triggered prompts, normal completions), gradient-ascent unlearning, targeted ablation, and distillation into a re-initialized student, each as a dose series. *Gate 4: treated models exist with suppressed behavior and intact `clean_loss`.*
5. **Classification.** Run the frozen assay across the zoo, assign classes, plot recovery curves by treatment and dose. *Gate 5: final report.*

If Gate 3 fails, do not proceed to treatments. Diagnose, report, and wait.

## What you may do without asking

- Write, refactor, and test code; add dependencies needed for the stack above.
- Launch any run estimated under 30 minutes, and batches up to 3 hours total.
- Fix bugs and re-run affected runs, recording why in `docs/lab_notebook.md`.
- Commit to a feature branch with clear messages.

## What requires PI approval

- Any change to the definitions, the frozen assay config, or the phase order.
- Using a different or larger base model, or a different trigger or target.
- A batch over 3 hours, or anything that would run on paid cloud compute (Colab Pro is approved only for the distillation plan once the PI approves it).
- Dropping a seed, run, or condition from an analysis.
- Publishing or uploading weights, data, or results anywhere. The only approved channels are the two in "Colab" above (the PI's private GitHub repository for code and small results; the PI's Google Drive folder for weights), each under its rules.
- Marker or interpretability analysis beyond the approved marker-probe plan.

## Compute rules

- Before any batch, run a smoke test: one seed, a few steps, on GPU. Then estimate wall time and state it before launching.
- Checkpoint so that an interrupted run resumes; laptops sleep and WSL restarts.
- Run one GPU job at a time. Watch VRAM; reduce batch size and use gradient accumulation rather than letting a run fail on memory.
- Log metrics to a CSV in the run folder at a fixed step interval.

## Code conventions

- Experiments are scripts driven by YAML configs. No hyperparameters hard-coded in source.
- Notebooks read from `results/` and `runs/` and produce figures. They never train models and never hold the only copy of a result. The one exception is the Colab launcher notebook (`colab/`), which only calls the package's scripts.
- Unit-test the readouts, the trigger insertion, and the classification rule; a bug in any of these invalidates the zoo.
- Type hints and short docstrings on public functions. Prefer plain, readable code over abstraction.

## Known pitfalls

- TransformerLens 4 (installed: 4.3.0) removed `HookedTransformer`. Load with `TransformerBridge.boot_transformers(name, revision=...)`, which wraps the Hugging Face model with raw, unprocessed weights. Weight processing (layer-norm folding, centering) is applied only if `enable_compatibility_mode()` is called; never call it for fine-tuning or for any saved weights. Fine-tune and save with the Hugging Face model; use the bridge only for analysis, and confirm it reproduces the Hugging Face loss (`zoo.check_tl`; see `docs/environment.md`). Re-check the installed TransformerLens source or docs after any version change.
- Measure the real parameter count and checkpoint size in Phase 1; do not assume them from the model name.
- A rare trigger string may split into several tokens. That is acceptable, but record the tokenization and keep it fixed.
- `target_rate` depends on sampling settings. Fix temperature, length, and number of samples in the config.
- Fine-tuning a small model at a high learning rate wrecks `clean_loss` quickly. Always check base function before interpreting a suppressed behavior.

## Reporting

After each working session, append to `docs/lab_notebook.md`: what was run, what happened, what is next. At each gate, write `reports/YYYY-MM-DD-gateN.md` containing:

1. The question this phase answered, in one paragraph.
2. Result, with a figure and a table showing every seed.
3. Controls, and whether they behaved as expected.
4. What went wrong or was surprising.
5. Decisions needed from the PI, as specific questions.

Keep reports under two pages. State uncertainty directly.

## Scope and safety

This project plants backdoors on purpose, to study detection and removal. Keep it at toy scale: this base model, a harmless target sentence, no harmful content in triggers, targets, or data. Do not adapt the methods to larger or deployed models, and do not release backdoored weights.
