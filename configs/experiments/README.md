# `configs/experiments/` — one file per run

Six files, one per required run. Each `extends:` the stage files it needs and states only what makes
it distinct: its **model** and its **optimizer**. Epochs, schedule, augmentation and the rest come from
the stages, so they cannot drift between runs.

| File | run_id | Run in notebook | Owner | Serves |
|---|---|---|---|---|
| `model_b__sgd.yaml` | `model_b__sgd__seed42` | `03_optimizer_study` | M3 | §3 (a) |
| `model_b__sgd_momentum.yaml` | `model_b__sgd_momentum__seed42` | `03_optimizer_study` | M3 | §3 (b) |
| `model_b__adam.yaml` | `model_b__adam__seed42` | `03_optimizer_study` | M3 | §3, §4, §6 |
| `model_a__adam.yaml` | `model_a__adam__seed42` | `04_custom_training_evaluation` | M3 | §4 baseline |
| `mobilenet_v2__adam.yaml` | `mobilenet_v2__adam__seed42` | `05_pretrained_finetuning` | M4 | §5, §6 |
| `squeezenet1_1__adam.yaml` | `squeezenet1_1__adam__seed42` | `05_pretrained_finetuning` | M4 | §5, §6 |

Notebooks load them with the notebook's `MODE`:

```python
cfg = load_config("configs/experiments/model_b__sgd.yaml", mode=MODE)
```

## The optimizer lives here, and only here

Each file defines its optimizer **in full**. No stage file has a default optimizer. Merging is
recursive, so a default would leak keys between optimizers — Adam's `betas` ending up inside an SGD
run, for example. `tests/contracts/test_config_layout.py` checks that every run's `optimizer.name`
matches its `run_id`.

For the pretrained runs, `optimizer.lr` is the **head** learning rate. The backbone trains at
`lr × pretrained.backbones[].backbone_lr_scale`, so the learning rate still has one home.

## The filename is not decorative

It must equal the derived `run_id` minus its `__seed<n>` suffix. A mismatch would scatter one run
across two directory names, and since a glob that matches nothing raises nothing, it would fail
silently at report time. The contract tests also assert that:

- all six files exist, and every run named in a report table has one;
- all use the same seed;
- all meet the 20-epoch minimum;
- each is official as written, and never official in `synthetic` or `debug` mode.

## Adding a run

Copy the closest file, change `model.name` or the `optimizer` block, and rename the file to match
the new `run_id`. Do not copy settings a stage file already provides — that is how two runs drift
apart without anyone noticing.

## The §3 learning rates differ on purpose

The SGD variants use `0.01`; Adam uses `0.001`. Plain SGD takes steps the size of the raw gradient
rather than normalised ones, so giving it Adam's learning rate would make SGD look worse than it is.
