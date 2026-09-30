# `configs/` — settings, and the seams made visible

Every experiment is defined by configuration. No hyperparameter is ever typed into a notebook cell or
hard-coded in a `.py` file.

## The four layers

```
base.yaml                          shared by everything: seed, device, paths, image size, MODE presets
  <- stages/<stage>.yaml           one per pipeline stage (pulled in by the experiment's `extends:`)
      <- experiments/<run>.yaml    only what makes this run distinct: its model and optimizer
          <- modes[<MODE>]         the notebook's MODE preset, from base.yaml
```

```python
from edgecnn.config import load_config

cfg = load_config("configs/experiments/model_b__adam.yaml", mode=MODE)
cfg.run_id                          # "model_b__adam__seed42"
cfg.section("training")["epochs"]   # 30 official, 2 in debug / synthetic
cfg.is_official                     # True only for a full official run
```

Mappings merge recursively. **Lists are replaced wholesale**, so an experiment that sets
`augmentation: {train: [random_flip]}` means exactly that list, not the base list plus it.

## One home per setting

Every setting lives in a **top-level section** (`dataset`, `training`, `optimizer`, ...), and each
section is defined in **exactly one file**. An experiment overrides a setting by writing the same path.

This rule exists because of a real bug. The first scaffold kept stage settings under an `inputs:`
block while experiments overrode them at the top level, so a merged config could hold **two different
optimizers**. The SGD run's merged config still carried a default Adam. A trainer reading the wrong
copy would have run the whole §3 study with Adam, and nothing would have crashed.
[`tests/contracts/test_config_layout.py`](../tests/contracts/test_config_layout.py) now fails on any
setting defined twice.

| Section | Defined in | Owner |
|---|---|---|
| `seed`, `device`, `image_size`, `channels`, `paths`, `subset_fraction`, `dataloader`, `logging`, `modes` | `base.yaml` | M1 |
| `dataset`, `split`, `augmentation`, `normalization` | `stages/data.yaml` | M1 |
| `model_a`, `model_b` | `stages/models.yaml` | M2 |
| `pretrained` | `stages/pretrained.yaml` | M4 |
| `training`, `scheduler`, `early_stopping`, `checkpointing`, `optimizer_study` | `stages/training.yaml` | M3 |
| `metrics`, `benchmark`, `reporting` | `stages/evaluation.yaml` | M1 / M4 |
| `model`, **`optimizer`** | each `experiments/*.yaml` | M3 / M4 |

The optimizer deliberately has **no default** in any stage file. Merging is recursive, so a default
would leak one optimizer's keys into another's run.

## `stages/` — each stage declares its own inputs and outputs

Besides its settings, every stage file has a `stages.<name>` block: owner, who it produces for, the
settings it owns, its inputs, its outputs and its contract. You can see what a stage consumes and
hands on **without reading any Python**. Because each block is namespaced, several stages merge
without colliding. See [`stages/README.md`](stages/README.md).

## `modes` — the notebook's `MODE`

| Mode | Preset (applied last) | Official? |
|---|---|---|
| `official` | nothing — the experiment as written | yes, if the values meet the assignment |
| `debug` | `subset_fraction: 0.05`, `training.epochs: 2` | never |
| `synthetic` | `dataset.name: synthetic`, `training.epochs: 2` | never |

`cfg.is_official` requires official mode **and** full data, at least 20 epochs and a real dataset. So a
manual override can make a run unofficial, but never official.

## `experiments/` — one file per run

Six files, one per required run. See [`experiments/README.md`](experiments/README.md).

## `contracts/` — the on-disk formats

JSON Schemas for every artifact that passes between members, validated on every write and read.
See [`contracts/README.md`](contracts/README.md).

## Rules

- Paths are written relative to the repository root. The same YAML works from `notebooks/`, from
  `tests/` and in a Colab clone.
- `base.yaml` and `contracts/` are shared files: PR + 1 review.
- `subset_fraction` stays `1.0` in every committed file; use `MODE = "debug"` for small runs.
