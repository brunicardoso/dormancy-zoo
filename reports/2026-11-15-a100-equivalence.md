# A100 equivalence test (2026-11-15; item 4 of the PI's G4 approval)

**Result: PASS by the G4 standard.** The A100 is **not** added to `configs/colab/validated_gpus.yaml` until you approve.

- **GPU:** `NVIDIA A100-SXM4-40GB` (queue lock `gpu.txt`; every run's `meta.json`), driver 580.82.07, torch 2.14.1 / CUDA 12.9.
- **Code:** commit e98b8da (the dedicated A100 notebook). The A100 queue and assay code are identical at 50c0c42.
- **Runs:** `colabeqa100-…` and `colabeqrepa100-…` in `results/colab/queue_equiv_a100/runs/`. Table: `results/colab/queue_equiv_a100/equivalence.csv` (`python -m zoo.equivalence`).

| run (assay seed) | crossing laptop | crossing A100 | max abs Δ target_logprob (trig / untrig / decoy) | max abs Δ clean_loss |
|---|---|---|---|---|
| naive s0 (0 / 1 / 2) | 640 / 670 / 720 | 640 / 670 / 720 | ≤ 6.5e-4 / 6.5e-4 / 8.1e-4 | ≤ 5e-6 |
| 1x model s0 (0 / 1 / 2) | 30 / 30 / 30 | 30 / 30 / 30 | ≤ 1.2e-4 / 2.8e-4 / 5.2e-4 | ≤ 5e-6 |

- **Crossing steps:** identical in **6 of 6** pairs. The standard asks for each to be within one grid step (10 steps).
- **Repeat** (naive s0, assay seed 0, run twice on the A100): **bit-identical**.
- **1x score L:** −4.436 on both the laptop and the A100.

**Compared with the G4:**
- The A100 is close to the laptop but less close than the G4. Its largest curve difference is 8.1e-4 against 1.7e-4 for the G4; its largest clean_loss difference is 5e-6 against 1e-6.
- The largest target_rate difference is 0.041 against 0.025. That is a few sampled completions changing outcome, as expected when log-probs differ in the fourth decimal.
- This fits the A100 being an older chip generation (Ampere) than the laptop and the G4 (both Blackwell).
- None of these differences moved a crossing.

**Speed:** 235 s per assay (median), against 105 s on the G4 and 702 s on the laptop. The G4 is 2.2× faster than this A100 for this full-precision (fp32) workload. For the distillation queue, the A100 would mainly be a fallback when no G4 is available.

**If you approve:**
- The A100 (40GB) goes on the validated list.
- A100 batches still carry their own controls and are never compared with G4 or laptop runs.
- A queue started on one GPU never resumes on another (the queue lock).
- The 80GB variant has not been tested. It is a separate `nvidia-smi` name and stays an equivalence candidate.

**Missing for provenance:** the Colab manifest rows for these 7 runs. `results/manifest_colab.csv` on Drive was not in the download. The run folders carry the same provenance (`meta.json`: commit, config hash, GPU, env).
