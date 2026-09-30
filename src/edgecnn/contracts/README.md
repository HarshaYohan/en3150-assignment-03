# `contracts/` — the frozen interface layer

**Custodian: Member 1. Used by all four. PR + 1 review to change.**

This is where the four workstreams meet. Nothing here does any work. It only *declares* what passes
between members, so the declarations can be read, agreed and enforced without anyone reading anyone
else's implementation.

Nothing in this package imports torch at module level, so it is always cheap to import.

## Files

| File | What it fixes |
|---|---|
| `types.py` | `DataBundle`, `TrainResult`, `EvalResult`, `ResourceProfile`, `ResolvedConfig`, and the project constants (`INPUT_SHAPE`, `MODEL_B_PARAM_BUDGET`, `MIN_EPOCHS`, `RUN_MODES`, `CHECKPOINT_KEYS`) |
| `protocols.py` | Structural interfaces: `ClassifierModel`, `TrainerProtocol`, `EvaluatorProtocol`. These are `Protocol`s rather than base classes, because torchvision's MobileNetV2 must satisfy the same contract as Member 2's hand-written Model B |
| `paths.py` | `run_id` derivation and every artifact location, including the separate `_debug/` checkpoint folder. The reason nobody builds a path string by hand |
| `schema.py` | Runtime validation against `configs/contracts/*.schema.json` |

## Inputs / outputs

**Inputs:** none — this package depends on nothing inside the project.
**Outputs:** type definitions, path functions and validation helpers, imported by every other package
and every notebook.

## The official-run rule lives here

`ResolvedConfig.is_official` is the single rule every library function follows: **return your result
in every mode; write into `results/` or `data/splits/` only when `cfg.is_official`.** It is True only
when *all* of these hold:

- `mode == "official"` — the notebook's `MODE`;
- `subset_fraction == 1.0` — full training data;
- at least `MIN_EPOCHS` (20) epochs;
- a real dataset, not the synthetic fixture.

So an override can make a run unofficial, but never official.

## Using it

```python
from edgecnn.contracts import DataBundle, paths, schema

run_id = paths.make_run_id("model_b", "adam", 42)
if cfg.is_official:
    schema.write_json(paths.history_json(run_id), payload, "history")   # validates before writing
history = schema.read_json(paths.history_json(run_id), "history")       # validates on read
```

## Changing something here

1. Say what and why in the group chat **before** opening the PR.
2. Get one review, from whoever consumes the thing you are changing.
3. Check that committed artifacts under `results/metrics/` still validate. If a schema change breaks
   existing results, either keep it backwards-compatible or agree that those runs get re-run. Never
   leave the repo in a state where half the results fail their own contract.

A test failure in `tests/contracts/` means an interface broke. It is not a bug in the code that
happened to fail.
