# `artifacts/checkpoints/` — trained weights

**Git-ignored** (except this file). Producer: the `Trainer`, in notebooks `03`, `04` and `05`.
Consumer: `evaluate_run`, in the same notebooks.

```
artifacts/checkpoints/<run_id>/best.pt           official runs - the checkpoint that is evaluated
artifacts/checkpoints/<run_id>/last.pt           official runs - for resuming an interrupted run
artifacts/checkpoints/_debug/<run_id>/*.pt       synthetic and debug runs - never mixed with official
```

Required keys in every checkpoint, checked by `tests/contracts/`:

```
model_name, state_dict, epoch, val_acc, class_names, input_shape, seed, config_snapshot
```

`class_names` and `input_shape` are mandatory so that a checkpoint **describes itself**.

`best` is chosen by **validation** accuracy, never test accuracy.

To resume an interrupted official run, set `training.resume: true` — as an override, for example
`load_config(path, mode=MODE, **{"training.resume": True})` — and re-run the training cell.

Never share or commit checkpoints. To get one back, re-run the notebook.
