# Plan amendment: K2 on the Phase 4b distillation students (for PI approval; 2026-11-15)

K2 is a reported readout. It never changes a class, an eligibility check, or the assay comparison. Nothing below runs until you approve it. Once approved, it is committed before any student's activations are computed.

## Models

- **Main arm:** `dist4b-s{k}-d{01,08,16,32,64}x`, teacher active k (25 models).
- **Sham students:** `dist4bsham-s{k}-d…x`, teacher naive k (25).
- **1x-teacher arm:** `dist4b1x-s{k}-d64x`, teacher the counter-trained 1x model k (5).
- **References:** naive k and active k, already scored.
- All students start from the base model. All five saved doses are scored; the predictions below cover every dose.

## Which probe

| option | what it measures | choice |
|---|---|---|
| **A. Probe fitted on active k** (frozen K2: `candidates.yaml`, `configs/markers.yaml`), applied unchanged | whether the active teacher's trigger-vs-decoy reading is present in the student | **chosen** |
| B. Probe fitted on the student itself | whether the trigger is decodable at all. The labels come from the prompt, so a probe can be fitted on any model. A distinct token string is probably decodable in any model, so this doesn't ask about inheritance. | no |
| C. Probe fitted on the student's own teacher (for the 1x arm, counter-trained 1x k) | a different probe per arm, so the arms are not comparable | no |
| D. Probe from another seed's active model | the swaps show this works but moves the control baseline (exploratory report) | no |

**Reason for A:**
- It is the confirmed marker, unchanged.
- It is the main-arm teacher's own probe.
- Treated and sham students of a seed share it, so the probe-source baseline shift seen in the swaps cancels in the comparison below.

## Scoring against sham students

- **Primary:** for each seed k and dose d, ΔK2 = K2(treated student k,d) − K2(sham student k,d).
- **Margin, from sham-student pairs:** R = the largest |K2 difference| between two sham students of the same seed. That is all 10 dose pairs in each of the 5 seeds, 50 pairs. It is the sham-to-sham variation that a treated−sham difference has to exceed.
- **Counts as "above its sham":** ΔK2 > R. The 1x-teacher arm is compared with the 64x sham student of its seed.
- **Also reported:**
  - every student's K2;
  - dK2 relative to naive k;
  - whether the student is above 0.768. That threshold is only for comparison with T3; it is not used for any verdict, because the swaps show it depends on the probe.

## Prediction, stated in advance

- **Prediction: no treated student is above its sham at any dose (0 of 25), and none in the 1x-teacher arm (0 of 5).**
- **Why:**
  - The distillation stories never contain the trigger.
  - The middle-layer features that respond to it are inactive on clean text, so the KL loss gives them no reason to form.
  - In Phase 4a, K2 marked a model that had itself been trained with the trigger.
- **Counts against the prediction:** at any dose, ΔK2 > R in 3 or more of the 5 seeds. That would mean the trigger representation passes through soft labels on clean text alone.
- **Mixed:** 1–2 seeds above R at a dose is reported as mixed, with no claim either way.

## How it will be read beside the assay

These are readings, not tests.
- **K2 equal to sham, and the assay shows treated students faster at 64x** (Phase 4b prediction 3): K2 and relearning speed dissociate. K2 would then mark a model that was itself counter-trained, not a dormant state in general.
- **K2 above sham:** the representation is inherited along with the behavior link.

## Logistics

- The PI downloads the 55 student dose folders (`final_fp16.safetensors`, `config.yaml`, `meta.json`; about 40 MB each, about 2.2 GB) from Drive to the laptop. Tracked total: 34.7 → about 37 GB.
- **Code:** `zoo.markers` gains the student model IDs (acts and probe, CPU, own output files with suffix `_phase4b`). The marker stage is otherwise unchanged.
- **Timing:** after the G4 queue's step 8, on CPU, about 15 min.
