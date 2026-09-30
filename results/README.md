# `results/` — everything the report is built from

**Committed in full.** Thanks to this folder, Member 4 can assemble every table in seconds, on a
laptop, with no GPU, without re-running anyone's training.

**Only official runs write here.** Synthetic and debug runs show their results inline in the notebook
and write nothing to this folder. Everything in it therefore came from a full official run.

| Folder | Contents | Produced by |
|---|---|---|
| `metrics/<run_id>/` | `history.json`, `test_metrics.json`, `resources.json` | notebooks `03`–`05` (history, test metrics), `06` (resources) |
| `figures/` | loss curves, confusion matrices, the optimizer overlay, dataset and trade-off figures | notebooks `01`, `03`–`05`, `07` |
| `tables/` | Markdown tables, pasted straight into the report | notebooks `02` (§2) and `07` (§3, §4, §6) |

## Per-run layout

```
results/metrics/model_b__adam__seed42/
├── history.json         notebook 03 - per-epoch loss, accuracy, LR, epoch time
├── test_metrics.json    notebook 03 - accuracy, macro P/R/F1, per-class, confusion matrix
└── resources.json       notebook 06 - params, size, MACs, latency, peak memory, device
```

`<run_id>` is always `{model}__{optimizer}__seed{seed}`, derived by
`edgecnn.contracts.paths.make_run_id` and never built by hand.

## Why the JSON is committed but checkpoints are not

Member 4 cannot build the §4, §5 or §6 tables without everyone else's `test_metrics.json` and
`resources.json`. If those existed only on the machine that produced them, assembling the report would
mean re-running every experiment the week it is due. Each is a few kilobytes. Checkpoints are
megabytes and can be regenerated, so they are git-ignored instead.

**Commit a run's files as soon as its official run finishes**, together with the notebook that
produced them. They are the hand-off to the next member, not optional tidying.

## Rules

- **Never hand-edit a file in here.** Each one is schema-validated on write and on read. An edited
  file fails `pytest tests/contracts` and, worse, can silently misreport a number in a table.
- **Figures are generated, never hand-made.** `results/figures/` is the source; `report/figures/` is a
  copy that notebook `07` makes.

## Expected contents when complete

Six run directories:

```
model_a__adam__seed42          model_b__sgd__seed42
model_b__adam__seed42          model_b__sgd_momentum__seed42
mobilenet_v2__adam__seed42     squeezenet1_1__adam__seed42
```

and five tables: `custom_architectures.md` and `param_breakdown.md` (§2), `optimizer_comparison.md`
(§3), `custom_model_comparison.md` (§4) and `final_comparison.md` (§6).
