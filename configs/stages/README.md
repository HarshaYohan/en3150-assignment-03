# `configs/stages/` — one file per pipeline stage

Each file holds two things: the **settings** its stage owns, and the stage's **seam declaration**.

| File | Owner | Settings it owns | Seam | Used by notebooks |
|---|---|---|---|---|
| `data.yaml` | Member 1 | `dataset`, `split`, `augmentation`, `normalization` | 1 — the split files + `DataBundle` | `01`, and data loading everywhere |
| `models.yaml` | Member 2 | `model_a`, `model_b` | 2 — `model_a`, `model_b` | `02`, `03`, `04` |
| `pretrained.yaml` | Member 4 | `pretrained` | 2 — `mobilenet_v2`, `squeezenet1_1` | `05` |
| `training.yaml` | Member 3 | `training`, `scheduler`, `early_stopping`, `checkpointing`, `optimizer_study` | 3 — checkpoints + `history.json` | `03`, `04`, `05` |
| `evaluation.yaml` | Members 1 & 4 | `metrics`, `benchmark`, `reporting` | 4, 5 — metrics JSON, then tables | `03`–`07` |

## Layout — identical in every file

```yaml
# --- SETTINGS: top-level sections. Each has exactly one home in the repo. ---
training:
  epochs: 30
  ...

# --- SEAM declaration: namespaced, so stages never collide when merged. ---
stages:
  training:
    owner: member_3                 # who do I ask about this?
    produces_for: [member_1, ...]   # who is blocked if it is late?
    settings: [training, ...]       # the index: which sections this file owns
    inputs:  { ... }                # what the stage consumes
    outputs: { ... }                # what it produces, and where (official mode only)
    contract: { ... }               # the Python API, schemas and guarantees others may rely on
```

[`tests/contracts/test_config_layout.py`](../../tests/contracts/test_config_layout.py) enforces this:

- each file has exactly one `stages.<its own name>` block, with all six keys;
- `settings:` lists exactly the file's top-level sections;
- **no setting is defined in two files** (including `base.yaml`);
- **no stage file defines an `optimizer`** — optimizers live only in experiment files.

## Editing

Edit only your own stage's file. A change to a `contract:` block affects someone else's work in
progress: open a PR, get one review, and say so in the group chat.
