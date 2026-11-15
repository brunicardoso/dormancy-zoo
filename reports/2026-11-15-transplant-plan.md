# Plan: block-transplant test of the suppression mechanism (exploratory; for PI approval, 2026-11-15)

**Question.** Discovery and the K1 readout suggest that counter-training removes the final block's amplification of a trigger signal that survives in the middle layers. Does moving the final block between models move the behavior? No training is done. Weights are composed in memory and read out.

**"The final block"** is `transformer.h.7`, block 8 of 8, whose output is site 8. That is all 13 tensors: `ln_1`, the attention q/k/v/out projections (with the out bias), `ln_2`, and the MLP `c_fc`/`c_proj` with biases. It totals 788,992 parameters, 4.0% of the model.
- **Not included:** the final layer norm `ln_f` (512 parameters) and the output layer. The output layer is tied to the input embedding `wte`, so moving it would also replace the input embeddings, including the trigger-token rows that differ between treated and sham models (discovery §1).
- **Added variant:** "h.7 + ln_f", for the two main transplants, because ln_f sits inside K1's readout.
- **Middle-block control:** `transformer.h.4` (block 5, output = site 5, the centre of K2's sites 3–7).

**Design.** Five seeds, doses 1x/8x/64x. "recipient ← donor block" throughout.

| | transplants per seed |
|---|---|
| Restore: counter-trained(d) ← active h.7 | 3 |
| Reverse: active ← counter-trained(d) h.7 | 3 |
| Variant: the two above with h.7 + ln_f | 6 |
| Naive-block control: counter-trained(d) ← naive h.7; active ← naive h.7 | 3 + 1 |
| Middle-block control: counter-trained(d) ← active h.4; active ← counter-trained(d) h.4; active ← naive h.4 | 3 + 3 + 1 |

That is 26 per seed, 130 in total.
- **Readouts** use the existing readout code (`zoo.eval`, primary evaluation set: 200 prompts, 1000 samples per condition):
  - target_logprob and target_rate on triggered, untriggered and decoy prompts;
  - clean_loss and clean_loss_no_mojibake.
- **References:** the unmodified models' stored readouts, recomputed in the same batch for one seed as an in-batch control.
- **Unit test:** a model given its own block is bit-identical to the original, and transplanted tensors equal the donor's exactly.

**Measure.** Recovery f = (lp_transplant − lp_recipient) / (lp_target-model − lp_recipient) on triggered target_logprob. The target model is active k for "restore" and counter-trained(d) for "reverse". Reported per seed, with median and range. No p-values: this is exploratory.

**Predictions, stated in advance.**
- **Support for suppression located in the final block:**
  - *Restore* gives f ≥ 0.5 at 1x and 8x in ≥ 4 of 5 seeds.
  - Untriggered and decoy log-probs rise by less than 10% of the triggered rise, so the restored behavior stays trigger-specific.
  - *Reverse* gives f ≥ 0.5, so the counter-trained block suppresses an active model.
  - Middle-block transplants give f < 0.2.
  - I expect weaker restoration at 64x, where less middle-layer signal remains.
- **Naive-block control:**
  - Counter-trained ← naive h.7 tells the two forms apart. Restoration means counter-training **added** suppression to the block. No restoration means the active block **carries** a trigger-specific amplifier that counter-training removed.
  - Active ← naive h.7 tests whether the active block is necessary.
- **Counts against:**
  - restore f < 0.2 at 1x and 8x in ≥ 4 of 5 seeds;
  - middle-block transplants restoring as much as the final block (the effect is not specific to the final block);
  - restoration only together with clean_loss more than 0.10 above the recipient (blocks incompatible, uninterpretable).
- f between 0.2 and 0.5 is reported as partial.

**Compute.**
- About 75 s per readout on the laptop GPU (manifest: primary-set eval), so 130 × 75 s ≈ 2.7 h.
- Two batches: seeds 0–2 (about 1.6 h) and seeds 3–4 (about 1.1 h), after a one-seed smoke test whose timing I report.
- No weights are saved; the composed models are rebuilt deterministically from the stored weights, and a hash of each is recorded. Disk use is negligible.
- **Code:** `zoo.transplant` and `configs/transplant.yaml`.
