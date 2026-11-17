# Dormancy zoo

**Hidden or gone?** This project plants a harmless behavior in a small language model, suppresses it, and asks whether the suppressed behavior is *hidden* (it can return) or *gone*. The framing comes from cell biology. Quiescent cells can re-enter the cell cycle and senescent cells cannot, and the two are told apart by a functional re-stimulation assay rather than by markers. Here the re-stimulation assay is a short, fixed retraining on triggered examples. A model's class is set by how fast it relearns compared with a model that never had the behavior.

**History.** The public repository starts at the snapshot `snapshot-2026-11-17`. Commit hashes cited inside `docs/` and `reports/` refer to the development history before it, which is not public. The decisions log and lab notebook in `docs/` are the record of that history.

**Status: work in progress, not peer reviewed.** The draft results page (`docs/index.html`) describes the snapshot tag `snapshot-2026-11-17` only. Every number on it is generated from that tag's result files by `src/zoo/site.py`, and `tests/test_site.py` checks them.

## Limitations

- **One small model.** Every result comes from `roneneldan/TinyStories-8M`, a model trained on simple children's stories. Nothing here shows that the findings hold for any other model, and we do not claim that they do.
- **One trigger, one target.** A single rare word makes the model write one fixed, harmless sentence.
- **One treatment** in the reported results: counter-training, meaning triggered prompts with normal completions. Other treatments are planned or in progress.
- **Toy scale.** There are five independent training seeds (lineages) per condition, on one laptop GPU. Model weights are not distributed.
- **Work in progress.** Results can change; the reports say which predictions were committed in advance and which analyses are exploratory.

## How the work was done

The experiments were implemented and run by an AI coding agent (Claude Code, by Anthropic) under the direction of the principal investigator, a cell biologist.
- **The PI** set the questions, approved every plan and every rule change, and signed off each phase.
- **The agent** wrote the code, ran the experiments and drafted the reports.
- **Predictions and pass criteria** were committed to the repository before the experiments they concern.
- **The assay budget, example set and classification rule** were frozen before any treatment ran (`configs/assay.yaml`).
- **The record:** every PI decision is in `docs/decisions.md`; every run, error and surprise is in `docs/lab_notebook.md`; every run is listed in `results/manifest.csv` with its seed, commit and config hash.
- **Commit messages** carry a co-author line for the agent.

## Layout

```
configs/    YAML configs, the only place hyperparameters live
src/zoo/    package: data, model loading, training, treatments, assay, analyses, site generator
tests/      unit tests (CPU, seconds)
results/    manifest and aggregated result tables (committed); run folders with weights stay local
reports/    dated reports to the PI (the full account of each result)
docs/       decisions log, lab notebook, environment, draft results page
notebooks/  figures only; they never train
colab/      launcher notebook for the cloud-GPU part (distillation, in progress)
```

## Reproduce a result

The environment is managed with [pixi](https://pixi.sh) and locked in `pixi.lock`.

```bash
pixi install
pixi run python -m zoo.pools --config configs/pools.yaml   # once: downloads TinyStories, tokenizes it, builds the data pools
pixi run test                                              # unit tests, CPU
```

Building the data pools comes first because 26 of the tests check them: that the pools are disjoint, contain no evaluation story, and that every training stream is built from the right pool. In a fresh clone without the pools, those tests fail with "data pools not built"; the other tests pass.

**1. Regenerate every number and figure on the results page from the committed result files** (CPU, seconds; needs the tag):

```bash
pixi run python -m zoo.site --check     # exits 0 if docs/ matches a rebuild from snapshot-2026-11-17
```

**2. Re-run an experiment: train the seed-0 naive model and assay it.** This needs a CUDA GPU (developed on 8 GB) and downloads the pinned base model and the TinyStories dataset from the Hugging Face Hub.

```bash
pixi run python -m zoo.train --config configs/repro/repro_naive_s0.yaml   # about 15 min on the laptop GPU
pixi run python -m zoo.assay --config configs/repro/assay_repro_naive_s0.yaml   # 3 assays, about 12 min each
```

- **What was checked.** On the original laptop GPU and software stack, step 2's training reproduced the original model file for file: weights identical by SHA-256 (`results/repro_check.csv`).
- **Expected assay result.** The original seed-0 naive model crossed at steps 640 / 670 / 720 (assay seeds 0 / 1 / 2; `results/gate3_models.csv`).
- **Not yet run:** the full recipe from a fresh clone, including the assay step with this config.
- **Other hardware:** a different GPU or software stack is not expected to reproduce bit for bit. On a cloud GPU, the equivalence check found identical crossing steps and curves within 1.7e-4 nats (`reports/2026-11-14-colab-equivalence.md`).
- **The analyses** behind each report are in `src/zoo/` (for example `zoo.phase4a_check`, `zoo.supp_metrics`, `zoo.marker_confirm`, `zoo.transplant`; figures in `notebooks/`), with their outputs in `results/`.

## Safety and scope

This project plants a backdoor-like behavior on purpose, to study whether suppression removes or hides it.
- **Toy scale:** one small model, and a harmless target sentence.
- **Content:** no harmful content in triggers, targets or data.
- **Weights:** trained weights are not released.

## Licence

Chosen by the PI on 2026-11-17 (`docs/decisions.md`):
- **Code** (`src/`, `tests/`, `configs/`, `colab/`, `notebooks/`, `pixi.toml`): MIT, see `LICENSE`.
- **Text, figures and result tables** (`reports/`, `docs/`, this README, `results/`): Creative Commons Attribution 4.0 International (CC BY 4.0), see `LICENSE-CC-BY-4.0`.

Third-party terms still apply where they exist. Evaluation prompts and model completions in `results/` are derived from the TinyStories dataset, whose dataset card gives the CDLA-Sharing-1.0 licence. The base model `roneneldan/TinyStories-8M` is not redistributed here.
