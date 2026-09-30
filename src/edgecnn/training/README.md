# `training/` — SEAM 3 producer

**Owner: Member 3.** Assignment §3 — **15 marks**, and the training half of §4 — **25 marks**.
**Notebooks:** [`03_optimizer_study`](../../../notebooks/03_optimizer_study.ipynb) and
[`04_custom_training_evaluation`](../../../notebooks/04_custom_training_evaluation.ipynb). The same
`Trainer` also runs Member 4's `05`.

## Inputs

| Seam | From | What |
|---|---|---|
| 1 | Member 1 | `build_dataloaders(cfg) -> DataBundle` |
| 2 | Members 2 & 4 | `build_model_from_config(cfg, num_classes)` |
| — | `configs/stages/training.yaml` | `training`, `scheduler`, `early_stopping`, `checkpointing` |
| — | `configs/experiments/*.yaml` | the run's `optimizer` — defined only there |

## Outputs

`Trainer(cfg).fit(model, data, cfg)` **always returns a `TrainResult`**, in every `MODE`, so the
notebook can plot it immediately. Files depend on the mode:

| Path | Official | Synthetic / debug |
|---|---|---|
| `artifacts/checkpoints/<run_id>/best.pt`, `last.pt` | ✓ (git-ignored) | — |
| `artifacts/checkpoints/_debug/<run_id>/*.pt` | — | ✓ (git-ignored) |
| `results/metrics/<run_id>/history.json` | ✓ **committed** | never |

`best.pt` carries every key in `contracts.types.CHECKPOINT_KEYS`, including `class_names` and
`input_shape`, so a checkpoint describes itself. `TrainResult.checkpoint_path` points at whichever
best checkpoint was written; the notebook passes it straight to `evaluate_run`.

## How a run looks in the notebook

```python
cfg = load_config("configs/experiments/model_b__sgd.yaml", mode=MODE)
model = build_model_from_config(cfg, data.num_classes)     # seeded weights
result = Trainer(cfg).fit(model, data, cfg)                 # returns in every mode
plot_loss_curves(result, write=cfg.is_official)
evaluation = evaluate_run(cfg, data, result.checkpoint_path)
```

## One loop, four models

`trainer.py` must contain **no branch on model name**. If it ever needs one, the registry contract is
wrong and that is what to fix. A per-model branch here is how the §6 comparison quietly stops being
like-for-like.

## Non-negotiable behaviours

1. **Never touch `data.test`** — not even to print a number. The held-out split stays sealed until
   `evaluate_run`.
2. **Time every epoch identically.** `epoch_time_s` covers the train pass plus the val pass, and it is
   a column in the §4 table. On GPU, call `torch.cuda.synchronize()` before stopping the clock:
   CUDA kernels are asynchronous, so a naive timer measures queueing, not compute.
3. **Record at least 20 epochs.** `history.schema.json` enforces `minItems: 20`, and early stopping
   has a 20-epoch floor.
4. **Write only when `cfg.is_official`.** A debug run on 5% of the data must never reach a committed
   table.
5. **Reseed at the start of `fit`** with `seed_everything(cfg.seed)`. Weights are already seeded by
   `build_model_from_config`, so the three §3 runs differ only in the optimizer, whatever ran before.
6. **Progress through `tqdm.auto`**: one updating line. Notebook outputs are committed, so never print
   per batch.
7. **Resumable.** With `training.resume: true`, continue from `last.pt` after an interruption.

## §3 — the optimizer study (notebook `03`)

Three runs on Model B that are identical except for the optimizer: `sgd`, `sgd_momentum`, `adam`. The
experiments are listed once, in `training.yaml → optimizer_study.experiments`. Their hyperparameters
live in the three experiment files. Seed, epochs, schedule and augmentation are shared by
construction.

The assignment asks specifically about **the impact of the momentum parameter**. Separate its two
effects, and show both in the curves rather than asserting them:

- the velocity term **damps oscillation** across narrow ravines in the loss surface;
- it **speeds up** progress along directions that keep descending.

Report the effect on convergence *speed* and on *final accuracy* separately. They are different
claims and can disagree.

Why Adam: depthwise kernels have 9 weights each, while pointwise kernels have hundreds. Their gradient
magnitudes differ by orders of magnitude, so one global SGD learning rate cannot suit both, but Adam's
per-parameter step can. Mention the cost too: Adam keeps two extra state tensors per parameter while
training.

The learning rates differ on purpose — `0.01` for SGD, `0.001` for Adam. Plain SGD takes steps the
size of the raw gradient, so giving it Adam's rate would make SGD look worse than it is.

## Files

| File | Purpose |
|---|---|
| `trainer.py` | the model-agnostic loop |
| `optimizers.py` | optimizer and scheduler factories; splits pretrained backbones' parameters with `param_groups` |
| `callbacks.py` | early stopping (with the 20-epoch floor) and checkpoint selection (official vs `_debug`) |
| `optimizer_study.py` | `summarize_study` — condenses the three §3 runs for the discussion |
