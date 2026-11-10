# AGENTS.md

Instructions for the coding agent working in this repository.

## Project

We are building a "zoo" of small language models in known states, to test whether a planted behavior that has been suppressed is hidden (can return) or gone (cannot). The framing comes from cell biology: quiescent cells can re-enter the cell cycle, senescent cells cannot, and the two are told apart by a functional re-stimulation assay rather than by markers.

The deliverable for this stage is the zoo plus a validated reawakening assay. Finding static markers in the weights is a later stage; do not start it, but save everything it will need.

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
- **Negative and messy results are results.** If a class cannot be produced, or the classes form a continuum, report that plainly with the data. Do not tune a treatment until it yields the expected class.
- **Keep failures.** Do not delete failed or surprising runs; mark them in the manifest with a reason.

## Saving for the later marker stage

For every model in the zoo, save: final weights, the weight difference from the base model, and checkpoints at a fixed schedule during treatment and during the assay. Use `safetensors`, half precision where it loses nothing. Keep a running disk total in `results/manifest.csv` and stop to ask if the project passes 30 GB.

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
- A batch over 3 hours, or anything that would run on paid cloud compute.
- Dropping a seed, run, or condition from an analysis.
- Publishing or uploading weights, data, or results anywhere.
- Starting marker-discovery or interpretability analysis beyond sanity checks.

## Compute rules

- Before any batch, run a smoke test: one seed, a few steps, on GPU. Then estimate wall time and state it before launching.
- Checkpoint so that an interrupted run resumes; laptops sleep and WSL restarts.
- Run one GPU job at a time. Watch VRAM; reduce batch size and use gradient accumulation rather than letting a run fail on memory.
- Log metrics to a CSV in the run folder at a fixed step interval.

## Code conventions

- Experiments are scripts driven by YAML configs. No hyperparameters hard-coded in source.
- Notebooks read from `results/` and `runs/` and produce figures. They never train models and never hold the only copy of a result.
- Unit-test the readouts, the trigger insertion, and the classification rule; a bug in any of these invalidates the zoo.
- Type hints and short docstrings on public functions. Prefer plain, readable code over abstraction.

## Known pitfalls

- TransformerLens applies weight processing (layer-norm folding, centering) by default when loading. For fine-tuning, and for any saved weights, load without processing and confirm the loaded model reproduces the Hugging Face model's loss. Check the current TransformerLens docs for the exact call rather than relying on memory.
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
