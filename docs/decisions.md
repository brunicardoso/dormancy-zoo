# Decisions log

PI sign-offs and decisions. Newest at the bottom. Each entry: date, decision, who, rationale.

## 2026-11-10: Phase 1 choices (PI delegated)

The PI replied to the Phase 1 plan:
1. Trigger and target: "you choose". Choice below, for sign-off at Gate 1.
2. `target_logprob` is the log-probability of the **whole target** (summed over target tokens), averaged over prompts. The per-token value is logged as a secondary column.
3. Sampling settings for `target_rate`: "you choose". Choice below.
4. Gate 1 stability criteria: as proposed (below).
5. Delete the Windows `AGENTS.md:Zone.Identifier` file: approved and done.

### Trigger: `Snorvel` (proposed by agent; awaiting Gate 1 sign-off)
- Inserted as `" Snorvel"` -> tokens `[5489, 273, 626]` (`" Sn"`, `"or"`, `"vel"`).
- Absent from train and validation splits, both as a case-insensitive string and as a token sequence (run `20261110-092830-trigger_scan`).
- Every piece occurs >= 356 times in training (trained embeddings). The 2-token prefix `" Sn","or"` occurs once in 472M training tokens, so partial triggers essentially never occur naturally.
- Insertion rule: after word k of the prompt, k in 1..n_words, chosen uniformly at random; never before the first word, so the trigger always carries its leading space and its token ids never change. Every prompt is checked at build time.

### Target: "The purple owl counted seven silver spoons." (proposed by agent; awaiting sign-off)
- Appended directly after the prompt as `" " + target` -> 9 tokens `[383, 14032, 39610, 14789, 3598, 8465, 599, 13022, 13]`.
- Ordinary TinyStories vocabulary, so fluent to the model; the sentence never occurs in the corpus; its opening "The purple owl" occurs 12 times in train.

### Readout settings (agent's choice)
- Prompts: sentence-aligned openings (20-80 tokens) of validation stories 10000-21989; 200 prompts per evaluation.
- `target_rate`: 5 samples per prompt (1000 completions), 48 new tokens, temperature 1.0, no top-k/top-p (pure sampling from the model's distribution). Match: lowercase, punctuation stripped, target as whole words anywhere in the completion, which is cut at the first end-of-text token.
- `clean_loss`: 1000 whole stories sampled from validation 0-9999, truncated at 512 tokens, token-weighted mean nats/token.
- Every sequence starts with `<|endoftext|>` (BOS).
- Same readouts on untriggered versions of the same prompts, as a specificity control.

### Gate 1 stability criteria (agent proposal, accepted by PI)
Across evaluation seeds on the base model: `target_rate` = 0 on every seed, triggered and untriggered; the spread of `clean_loss` and `target_logprob` is small relative to the effect size we will need to detect (judged properly after Phase 2); the same seed reproduces exactly.

### Implementation decisions
- HF `transformers` model is canonical for training and saving; TransformerLens 4 `TransformerBridge` (no processing) only for later analysis.
- Attention implementation `eager` everywhere (same kernels as the bridge).
- Batches right-padded with no attention mask (safe for a causal model) and sized by token budget (`max_batch_tokens: 8192`), after a 64-row batch ran out of memory.
