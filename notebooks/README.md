# `notebooks/` — where every experiment runs

One notebook per assignment section. They are the **official drivers**: every number and figure in
the report comes from one of them, run in `MODE = "official"`. The code they call lives in the
shared library, `src/edgecnn/`. How the two fit together, and your day-to-day workflow, is in
[`CONTRIBUTING.md` §2–§3](../CONTRIBUTING.md#2-how-development-happens).

## The eight notebooks

| Notebook | Owner | § | Trains | Writes (official mode only) |
|---|---|---|---|---|
| [`00_setup`](00_setup.ipynb) | M1 | — | — | nothing — environment, device, configs, contract tests |
| [`01_data_preparation`](01_data_preparation.ipynb) | M1 | 1 | — | `data/splits/*`, `results/figures/dataset/*` |
| [`02_custom_architectures`](02_custom_architectures.ipynb) | M2 | 2 | — | `results/tables/custom_architectures.md`, `param_breakdown.md` |
| [`03_optimizer_study`](03_optimizer_study.ipynb) | M3 | 3 | `model_b` × SGD, SGD+momentum, Adam | 3 × history + test metrics, curves, optimizer overlay |
| [`04_custom_training_evaluation`](04_custom_training_evaluation.ipynb) | M3 | 4 | `model_a` | history + test metrics, curves, confusion matrix |
| [`05_pretrained_finetuning`](05_pretrained_finetuning.ipynb) | M4 | 5 | `mobilenet_v2`, `squeezenet1_1` | 2 × history + test metrics, curves, confusion matrices |
| [`06_resource_benchmark`](06_resource_benchmark.ipynb) | M1 | 4, 6 | — | `resources.json` for all six runs |
| [`07_final_comparison`](07_final_comparison.ipynb) | M4 | 3, 4, 6 | — | all comparison tables, trade-off figures, `report/figures/` |

Notebooks never share variables. They communicate only through committed files — the five seams in
the [root README](../README.md#the-five-seams). So each notebook runs on its own, in any session, on
any machine.

**Official run order:** `01` → `02` → `03` → `04` → `05` → `06` → `07`. `04` reads Model B's run from
`03`; `06` needs every run's `history.json`; `07` needs everything.

## Inside every notebook

| Cell | What it does |
|---|---|
| Title | section, marks, owner, what it reads and writes, what the report needs from it |
| **1 · Colab bootstrap** | does nothing locally; on Colab it clones and installs the repository |
| **2 · Setup** | `%autoreload 2`, imports, and **`MODE`** |
| Style | `apply_style()`: small inline figures, print-quality saved PNGs |
| Config | `cfg = load_config("configs/experiments/<run>.yaml", mode=MODE)` |
| Steps | markdown saying what the step does and what to check, then one library call, then one cell that shows its result |
| Checklist | the official-run checklist |
| Last | Colab only, official only: pushes this run's results |

The steps call **library functions**, never logic written inside the notebook. Until a function is
implemented, its cell raises `NotImplementedError` naming the member who owns it — that is expected.

## `MODE`

Set in cell 2. It changes how much runs and whether anything is written:

| `MODE` | Data | Epochs | Writes files? | Use it for |
|---|---|---|---|---|
| `"synthetic"` | generated fixture | 2 | no | before EuroSAT exists — seconds |
| `"debug"` | 5% of EuroSAT | 2 | no | finding bugs step by step |
| `"official"` | full EuroSAT | 30 | **yes** | the report |

Library functions return their results in every mode, so every plot still shows in a debug run. They
write only when `cfg.is_official`, so a debug run can never overwrite a committed result. For the real
run: set `MODE = "official"`, then Kernel → **Restart & Run All**.

## Opening a notebook in VS Code

Open the notebook, then **Select Kernel** (top right) → **Python Environments** → `.venv`. Imports
work because the package is installed in editable mode. **Never add `sys.path.append(...)`** —
`tests/test_notebooks.py` fails if you do.

## Autoreload

Cell 2 runs `%autoreload 2`, so edits to `src/edgecnn/**/*.py` apply the next time you run a cell.
Two caveats:

- **Objects built before the edit keep the old code.** Re-run the cell that *builds* the model, loader
  or trainer.
- **Some edits cannot be reloaded**, for example changing a dataclass's fields. If behaviour looks
  stale, restart the kernel.

## Outputs are committed

Every notebook is committed **with its outputs**, so plots and metrics are visible on GitHub without
running anything. Each notebook has a single owner, so this causes no merge conflicts. Keep them
small and clean:

- inline figures at ~100 DPI — the high-resolution PNG goes to `results/figures/`;
- no bulk prints: no whole DataFrames, no per-batch logs;
- repo-relative paths only — never `C:\Users\<you>\...`;
- **never** a token, password or email address in any cell or output.

## Running without the editor

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/03_optimizer_study.ipynb
```

Runs the notebook headless from a terminal and saves its outputs into the file, just like
Restart & Run All. Useful for leaving a long official run going.

## Colab fallback

Everything runs locally by default. Use Colab when a run is too slow on your laptop — the Stage 4 dry
run shows you which runs those are. You run the **same notebook**, unchanged.

### One-time setup (only if you will use Colab)

1. On GitHub: **Settings → Developer settings → Personal access tokens → Fine-grained tokens →
   Generate new token**. Repository access: **only this repository**. Permissions: **Contents →
   Read and write**. Copy the token.
2. In any Colab notebook, open **Secrets** (the key icon in the left sidebar) and add:

   | Name | Value |
   |---|---|
   | `GH_TOKEN` | the token from step 1 |
   | `GIT_NAME` | your name, as it should appear on the commit |
   | `GIT_EMAIL` | your GitHub email (the `...@users.noreply.github.com` one is fine) |

   Turn on **Notebook access** for each when Colab asks.

### Running a notebook on Colab

1. Open it straight from GitHub:
   `https://colab.research.google.com/github/ThejithaR/EN3150-Assignment-03-CNN/blob/main/notebooks/<notebook>.ipynb`
2. **Runtime → Change runtime type → T4 GPU.**
3. In cell 1, set `BRANCH` if your work is not merged into `main` yet.
4. Set `MODE = "official"` in cell 2, then **Runtime → Run all**. Cell 1 clones and installs the
   repository. EuroSAT is downloaded again and processed to match the committed split — the split is
   reused, never redrawn.
5. The last cell commits this run's result files and pushes them.
6. Save the notebook itself: **File → Save a copy in GitHub**, same repository, same branch, same
   path. That is what brings the outputs back.

Do step 5 before step 6, and do both before closing the tab. A Colab session is wiped when it ends.

### Rules

- **The token never appears in a cell or an output.** It stays in Colab Secrets and reaches git only
  through an environment variable. The repository is public and outputs are committed, so a printed
  token would be published. `push_results` scrubs it from git's output as a safety net.
- `03` and `04` share a table of epoch times, so run **both** on Colab, or **both** on the same
  laptop.
- Run `06` locally, on CPU. It measures latency, and Colab's CPU model changes between sessions.
- No shared drive is needed. Results are small and go through git. Checkpoints are never shared,
  because nothing downstream needs them.

## `scratch/`

Personal notebooks for trying things out — see [`scratch/README.md`](scratch/README.md). Nothing in
`scratch/` ever produces a report result.
