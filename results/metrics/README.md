# `results/metrics/` — the numbers, committed

**COMMITTED.** This is SEAM 4, and the hand-off Member 4 depends on. **Only official runs write
here.**

```
results/metrics/<run_id>/
├── history.json         notebooks 03-05 (Trainer)       - per-epoch loss, accuracy, LR, epoch time
├── test_metrics.json    notebooks 03-05 (evaluate_run)  - accuracy, macro P/R/F1, per-class, confusion matrix
└── resources.json       notebook 06     (profile_all)   - params, size, MACs, latency, peak memory, device
```

`<run_id>` is `{model}__{optimizer}__seed{seed}`, from `edgecnn.contracts.paths.make_run_id`.

## Commit these as soon as an official run finishes

Member 4 cannot build the §4, §5 or §6 tables without everyone's JSON. Commit them together with the
notebook that produced them.

## Rules

- **Never hand-edit.** Each file is schema-validated on write and on read. An edit fails
  `pytest tests/contracts` and, worse, can silently misreport a number in a table.
- **Nothing from a synthetic or debug run belongs here.** The library enforces this through
  `cfg.is_official`, and `history.schema.json` also rejects runs shorter than 20 epochs.

Expected when complete: `model_a__adam__seed42`, `model_b__adam__seed42`, `model_b__sgd__seed42`,
`model_b__sgd_momentum__seed42`, `mobilenet_v2__adam__seed42`, `squeezenet1_1__adam__seed42`.
