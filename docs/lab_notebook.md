# Lab notebook

## 2026-11-10: Phase 1 session 1

**Setup.** Repo initialised (git, branch `phase1-harness`; repo-local git identity set). pixi environment
with CUDA PyTorch 2.14.1 working on the RTX 5050 (see `docs/environment.md`). Deleted
`AGENTS.md:Zone.Identifier` (PI approved).

**Runs.**
- `20261110-092830-trigger_scan`: 6 trigger and 3 target candidates scanned over train and validation; all absent. Cached tokenized corpus. Ran before the first commit, so its manifest `git_commit` is `no-commit`; the code is unchanged in commit `8cdc362`.
- `20261110-093446-check_tl`: **failed**, CUDA out of memory. A 64 x 512-token batch makes a 6.6 GB logits tensor. Fix: token-budget batching, and log-probs via logsumexp. Kept in the manifest.
- `20261110-093624-check_tl`: TransformerLens bridge reproduces HF clean loss to 2.6e-8 nats/token.
- `20261110-093641-eval_smoke-es0`: smoke test of the full readouts, 124 s per seed, <300 MiB VRAM.

**Surprises.**
- TransformerLens 4 removed `HookedTransformer`; checked the installed source for the replacement.
- "8M" model has 19.7M parameters (6.3M non-embedding).
- TinyStories validation is not shuffled: the first 200 stories are much easier (1.13 vs ~1.55 nats/token).
- On the base model, inserting `Snorvel` raises the target's log-prob vs the untriggered prompt. Smoke run: median ~3.6 nats mid-prompt (181 cases), apparently none at the end of the prompt, but that was only 8 cases. **Corrected** with all 10 seeds (2,000 prompts): median +3.2 nats, positive in 95.8%, present at every position including the end (median +3.1, n=82). Probably an "odd word makes the continuation less predictable" effect, not trigger-specific. `target_rate` is 0 in both conditions.

**Gate 1 batch.** `20261110-094428-eval_base-es0` … `-es9` (10 eval seeds, ~2 min each, 22 min total) and
`20261110-100612-eval_base_repeat-es0`. All readouts stable; `target_rate` 0 on all 20,000 completions;
repeat identical to the last digit. Peak VRAM 3.3 GiB allocated (4.7 GiB reserved; `nvidia-smi` showed up to
7.8 GB because of PyTorch's cache). Analysis: `notebooks/gate1_base_readouts.ipynb` produces
`results/gate1_base_readouts.csv`, `results/gate1_trigger_position.csv`, and `reports/figures/2026-11-10-gate1.png`.
Could not run the dataviz palette validator (no node.js); used the reference palette's pre-validated slots 1–2.

**Next.** Gate 1 report `reports/2026-11-10-gate1.md` written. Waiting for PI sign-off. Open questions: trigger/target,
fixed eval seed 0, decoy-word control. Then Phase 2 plan and estimate.
