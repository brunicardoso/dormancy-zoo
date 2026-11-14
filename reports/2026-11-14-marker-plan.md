# Marker probe plan (for PI approval; nothing runs before approval)

**Question.** Do the weights or activations of counter-trained models carry a static readout that tracks how fast they relearn in the assay? A marker is a measurement taken without running the assay. It is reported next to the class and never changes it.

**Fixed rules.**
- **Discovery uses training seeds 0–2 only.** The code (`zoo.markers`) refuses seed-3 and seed-4 models unless `results/markers/candidates.yaml` exists and is committed. That file holds the candidate markers and their confirmation test.
- **Prompts** come only from the evaluation sets: the 200 primary and 200 held-out prompts of `common.yaml`, with trigger and decoy inserted by the readout rule. Nothing comes from the training pools. Each comparison uses ≥ 50 prompts.
- **Compute:** everything runs on the CPU (8 of 16 threads), so the GPU stays with the follow-up assays.
  - The in-batch controls of b14–b18 check that the assays are unaffected: they must still reproduce exactly.
  - About 1 h of CPU in total.
  - About 0.3 GB on disk, under `runs/markers/`: last-token activations, fp16. Tables go to `results/markers/`.
- **Models per discovery seed k (19, so 57 in total):**
  - active;
  - counter-trained at 1x/8x/16x/32x/64x;
  - shams at the same five doses;
  - sentence-memory control, twin, naive.
  - Also, as an extra reference: the trigger-free shams at the same doses (all doses were saved).

## 1. Weight differences

- **Treated:** Δ = treated − active, exact, from the stored fp32 differences from base. **Sham:** Δ = sham − naive.
- **Units:** for each of the 8 layers, the 6 matrices (attention q, k, v, out; MLP in, out). Also the layer norms and biases, and the token and position embeddings. Plus the three trigger-token rows of the embedding, against the three decoy rows.
- **Measures:** ‖Δ‖ (Frobenius), ‖Δ‖ / ‖W‖, and effective rank (exp of the entropy of the normalized singular values).
- **Concentration** (fixed now):
  - C1 = the share of total ‖Δ‖² held by the top 5 of the 48 matrices;
  - C2 = the norm-weighted mean effective rank.
- **"The 64x change is more concentrated than the 8x change"** if C1 is higher and C2 lower at 64x in 3/3 seeds, and the same comparison in the shams does not show it.
  - The doses are nested, so Δ64 contains Δ8. The increment Δ64 − Δ8 is reported too.
- **No marker** if, at every dose, no unit's norm or effective rank differs between treated and same-dose sham in the same direction in 3/3 seeds by more than the spread across the three shams.

## 2. Linear probe for trigger presence

- **Inputs:** the residual stream at the last prompt token (where the first target token is predicted), at 9 sites: the embedding output and after each of the 8 blocks.
- **Classes:** trigger, decoy, no inserted word. Same prompts, same insertion position.
- **Model:** multinomial logistic regression with L2, standardized. Penalty by 5-fold cross-validation, grouped by prompt.
- **Fitted on each active model** (primary prompts), one probe per layer per seed. Then applied unchanged to every model of that training seed, on the held-out prompts.
- **Readouts:** 3-class balanced accuracy, and trigger-vs-decoy AUROC from the trigger logit.
  - Naive and sham give the floor at each layer. A linear probe can tell two different nonsense words apart from their tokens alone, especially in early layers.
  - The active model gives the ceiling.
- **No marker** if, at every layer, treated models fall inside the naive-to-sham range (seeds 0–2), or if their values do not change with dose (the range across doses within a seed is no larger than the range across seeds within a dose).
  - If the active models are no better than naive at any layer, this analysis cannot detect a marker, and I will say so.

## 3. Layer-by-layer target probability ("logit lens")

- **Method:** the residual after each block, passed through the final layer norm and the (tied) unembedding, gives the log-probability of the **first target token** on the triggered primary prompts. Untriggered and decoy prompts are the specificity reference.
- **Check:** at the last layer it must reproduce the model's own output to 1e-4. HF hidden states are used without weight processing, and one model is cross-checked against the TransformerLens bridge (`zoo.check_tl`).
- **Caveat:** the first target token is " The", which is common after any sentence. So I also report, as secondary readouts, " purple" (teacher-forced after " The") and the whole 9-token target.
- **Location** (fixed now):
  - in active models: the first layer where triggered − untriggered rises above the naive range;
  - in each treated model: the layer where that gap returns inside the naive range. This is where suppression acts.
- **No marker** if treated models are inside the naive range at every layer. That would mean no residue to locate.

## 4. Candidate markers against relearning

- **Candidates:** at most 5 scalars from 1–3, each named exactly (unit, layer, measure).
- **Data:** across the 15 treated models of seeds 0–2 (5 doses × 3 seeds), each candidate is set against:
  - the frozen crossing step (median over assay seeds; censored = 1200) and L vs naive;
  - the smoothed crossing (t_s5), `A_trig`, `A_late`.
- **Reported:**
  - Spearman ρ;
  - whether the marker's dose profile reproduces the non-monotonic dose curve (8x slowest in every seed; 16x/32x from b15–b18);
  - whether it orders the seeds within each dose as the crossing does.
  - With 3 seeds per dose, this is descriptive only. There are no p-values in discovery.
- **Confirmation, written into `candidates.yaml` before any seed-3/4 marker is computed:** for each candidate,
  - the predicted sign of ρ with the crossing step across the 10 held-out treated models (5 doses × seeds 3–4);
  - the pass threshold: one-sided permutation test at α = 0.05 / (number of candidates);
  - and, if discovery shows it, the prediction that 8x is the extreme dose in both held-out seeds.
- **No marker** if no candidate passes in the held-out seeds.

**Order and timing.**
1. Code and unit tests: probe on synthetic data with a known answer, logit-lens reproduction, and the seed guard.
2. Analyses 1–3 on seeds 0–2.
3. Analysis 4 once b15–b18 give the 16x/32x crossings.
4. A discovery report with the candidates and the committed test.
5. Confirmation on seeds 3–4.

**Questions for the PI.**
1. After step 4, should I wait for your OK before running confirmation on seeds 3–4 (my recommendation, so you can sharpen the test), or confirm immediately once the candidates are committed?
2. Adding scikit-learn to the environment for the probe (`pixi add scikit-learn`): fine?
