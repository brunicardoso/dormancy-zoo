# Publication audit of the repository and its history (2026-11-17)

**Scope.** I scanned every file version ever committed: 1,657 unique blobs across all 97 commits and all refs. I also checked the commit metadata, and the tree of the snapshot tag `snapshot-2026-11-17` (commit 20a2f73, 1,369 files). Nothing has been changed in history. The tag exists locally only and has not been pushed. Scan script: run from a scratch directory, read-only, Python run with `-I`.

## 1. Must decide before anything goes public

| # | Finding | Where | Options |
|---|---|---|---|
| 1 | **The PI's email address is the author and committer of all 97 commits.** It also identifies the institution. | git metadata | (a) Publish a **new** repository whose first commit is the tag's tree, authored with a no-reply address. There is no history to rewrite, and the private repository stays as it is. (b) Rewrite this repository's history (`git filter-repo` with a mailmap). (c) Accept. |
| 2 | **The same email appears in the text** of `docs/decisions.md` line 58 ("Git identity (task D)"), present in 34 versions of that file. | file content and history | Edit the line in the public copy; history is handled by 1(a) or 1(b). |
| 3 | **28 commit messages carry a `Claude-Session:` link** to the coding agent's session. The link is access-controlled, but it exposes a session identifier. | commit messages | 1(a) removes them; 1(b) can strip them; or accept. |
| 4 | **Distillation (Phase 4b) material is in the snapshot,** although the site leaves it out as instructed. | `results/colab/` (equivalence runs and the G4 distillation queue, including the unexpected transmission result), `reports/2026-11-14-phase4b-colab-proposal.md`, `2026-11-15-k2-phase4b-amendment.md`, the A100 reports, lab-notebook entries | Keep it in the public repository as work in progress, or publish a tree without it. Leaving it out would mean the decisions log and lab notebook refer to files that are not there. |
| 5 | **GitHub Pages from `docs/` would also serve the other files in `docs/`.** The decisions log, the lab notebook and the environment file would become pages next to the site. | `docs/` | Accept, or add a `docs/_config.yml` that excludes them. Pages can serve only the repository root or `docs/`. |
| 6 | **TinyStories-derived text is in the results.** The dataset card in the local Hugging Face cache gives `cdla-sharing-1.0`. Evaluation prompts and model completions built from the dataset are committed, in `results/eval_set_primary_*.csv` and the `completions_*.csv` files under `results/colab/`. Redistributing them carries that licence's sharing terms. The base model's licence is not in the local cache; I have not checked the Hub. | `results/` | State the data licence for those files, or drop the completions from the public tree. Check the model card's licence. |

## 2. Worth a decision, low risk

| # | Finding | Where |
|---|---|---|
| 7 | The GitHub handle that owns the private repository is written out. Harmless if the public repository is under the same account. | `docs/lab_notebook.md` line 387 |
| 8 | Internal process is public by design: verbatim PI instructions, how much time the PI spends on reviews, and the agent's errors (for example the `/tmp` incident). This is consistent with the disclosure, but it is personal detail. | `AGENTS.md`, `docs/decisions.md`, `docs/lab_notebook.md` |
| 9 | The code that plants the backdoor-like behavior, the trigger string and the target sentence become public. They are toy-scale and harmless, and no weights are in the repository. AGENTS.md allows code and small results through approved channels and forbids releasing backdoored weights; a public repository is a new channel and needs your approval. | `src/zoo/`, `configs/common.yaml` |
| 10 | The Colab launcher contains the Drive folder path (`/content/drive/MyDrive/dormancy_zoo`) and a placeholder remote (`github.com/OWNER/REPO.git`). It reads the GitHub token from Colab secrets; no token is present. | `colab/phase4b_launch.ipynb` |

## 3. Checked and clean

- **Secrets:** none in any version of any file. The scan looked for GitHub tokens (`ghp_…`, `github_pat_…`), Hugging Face tokens (`hf_…`), AWS, OpenAI and Anthropic key patterns, private-key headers, and `token = "…"` assignments.
- **Weights:** no weights, checkpoints or bundles anywhere in history. The largest blob is 1.9 MB (distillation `data.csv` story indices). Nothing under `runs/` was ever committed. `.gitignore` covers `*.safetensors`, `*.pt`, `*.bin`, `*.ckpt`, `runs/`, `export/`, `logs/` and `data_cache/`.
- **Personal paths:** no `/home/…` or Windows user paths in any committed file. One notebook output shows a temporary kernel path (`/tmp/ipykernel_…`, `notebooks/phase2_gate2.ipynb`). It is harmless and can be cleared.
- **Stray files:** none tracked. `export/` (the Drive staging folder, which holds a Windows shortcut I did not create) and `logs/` are untracked and gitignored.
- **Agent attribution:** the `Co-Authored-By: Claude …` lines in 95 commits use `noreply@anthropic.com`. They are consistent with the disclosure.

## 4. My recommendation

Option 1(a): a new public repository created from the snapshot tree as a single commit with a no-reply author, after the edits you choose from items 2 and 4–9. It leaves the private repository and its history untouched and drops the session links and the email in one step. It also lets you decide what the public tree contains (item 4) without rewriting anything.

## 5. Licence proposal (your choice)

- **Code** (`src/`, `tests/`, `configs/`, `colab/`, `notebooks/`): **MIT** (short and permissive). Alternative: Apache-2.0, which adds an explicit patent grant.
- **Text and figures** (`reports/`, `docs/`, `README.md`, the results page): **CC BY 4.0**.
- **Result tables** (`results/`): CC BY 4.0, except files containing TinyStories-derived text, which follow the dataset's licence (item 6).

**PI decision (2026-11-17):** MIT for code; CC BY 4.0 for text, figures and result tables. Files: `LICENSE`, `LICENSE-CC-BY-4.0` (official legal code from creativecommons.org, sha256 9ba9550a…). Items 1–10 above are still open.
