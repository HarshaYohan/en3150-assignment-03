# `edgecnn`

The shared library. Import it as `edgecnn` from anywhere after `pip install -e .`.

| Subpackage | Owner | Role | Called from notebooks |
|---|---|---|---|
| [`contracts/`](contracts/README.md) | M1 (custodian) | **FROZEN** — types, protocols, paths, schema validation | all |
| [`config/`](config/README.md) | M1 | layered YAML: base ← stage ← experiment ← mode | all |
| [`data/`](data/README.md) | M1 | Seam 1 — EuroSAT, the split, DataLoaders, §1 figures | `01`, `03`–`05` |
| [`models/`](models/README.md) | M2 + M4 | Seam 2 — registry, custom models, pretrained backbones | `02`–`06` |
| [`training/`](training/README.md) | M3 | Seam 3 — the one training loop, optimizers | `03`–`05` |
| [`evaluation/`](evaluation/README.md) | M1, M3, M4 | Seams 4 and 5 — metrics, benchmark, curves, tables | `02`–`07` |
| [`utils/`](utils/README.md) | M1 | seeding, device, logging, the Colab helper | all |

## Library, not notebooks

Anything shared lives here. Notebooks only call it. Code starts in a scratch notebook and moves here
**the moment it works** — see
[CONTRIBUTING.md → the promote rule](../../CONTRIBUTING.md#the-promote-rule). Every function you own
already exists as a stub whose docstring states its inputs, outputs and traps. Keep its signature:
other members' notebooks are already calling it.

## Two rules every function follows

1. **Return always, write only when official.** Every function returns its result in every `MODE`,
   so debug runs can plot it. Files are written only when `cfg.is_official` — or, for functions
   spanning several runs, when the notebook passes `write=True`.
2. **One construction path.** Models are built with `build_model_from_config(cfg, num_classes)`,
   which reads the model's settings and seeds its weights. The model trained in `03`–`05` is then
   exactly the one profiled in `06`.

## Import discipline

Only `edgecnn.contracts` is imported eagerly — it is plain declarations with no torch dependency, so
it is cheap and always importable. Everything else is imported on demand.

Members talk through `contracts/`, never by importing each other's internals. In particular, **the
trainer never imports a model module**: it gets its model from the registry and never learns which
architecture it is. That indirection is what keeps the §6 comparison fair.
