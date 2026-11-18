# Audit before publishing the working branch (2026-11-18)

**Scope.** Every file version reachable from any branch or tag of the working repository (157 commits, 3,661 file versions, 581 MB), every commit message and author, and the three local commits of the public preview. Scanner: patterns for GitHub, Hugging Face, AWS, Google, Slack and `sk-` keys, private keys, bearer headers, URLs with credentials and secret assignments; email addresses; personal paths; weight-type files; stray file types; files over 5 MB; notebooks with saved outputs.

**Clean.** No token or credential pattern anywhere, in any version or commit message. No weight file, no stray file type (archives, shortcuts, logs, bundles, caches), no file over 5 MB. No file was ever deleted, so nothing is hidden in history.

**Found.**

| # | finding | where | at the tip | in history |
|---|---|---|---|---|
| 1 | Per-run files (decided to stay out) | `results/colab/*/runs/`, about 2,900 files | about 580 MB | 545 MB of file versions, since the first Colab import |
| 2 | Institutional email as commit author | all 157 commits | — | every commit |
| 3 | Institutional email in text | `docs/decisions.md` (git identity note) | 1 line | 42 versions |
| 4 | Notebooks with saved outputs | 17 notebooks | stripped (59eb3f6) | 25 versions; also 16 notebooks already in the public repository |
| 5 | Session links in commit messages | `Claude-Session:` trailer | — | 88 messages |
| 6 | Local username in paths | `results/manifest.csv`, 15 smoke-test rows (`/tmp/claude-1000/-home-bruni-…`) | yes | yes |
| 7 | Same tag name, different commit | `snapshot-2026-11-17` (private 20a2f73, public 13e619e) | — | — |

Saved notebook outputs are figures and printed tables. The scan found no path, address or token in them. Other paths are generic (`~/…` in the lab notebook, `/content/drive` in the Colab launcher, `/mnt/c` in AGENTS.md).

**Done.** Outputs stripped from all 17 notebooks. Commit check `zoo.repo_check` (standard-library Python, pre-commit hook, plus a test over every tracked file). It refuses notebooks with outputs, token or credential patterns and weight files. Tested in a throwaway clone: refuses a notebook with outputs and a planted token, accepts a clean file. Nothing pushed.
