# `notebooks/scratch/` — your workbench

Personal notebooks for prototyping, experiments and dead ends. This is where code starts. It
doesn't stay here: once something works, it moves into `src/edgecnn/`.

## Naming

```
m<n>_<topic>.ipynb        m1_eurosat_loading.ipynb
                          m2_model_b.ipynb
                          m3_lr_sweep.ipynb
```

The `m<n>_` prefix shows who owns each file and keeps four people's notebooks from colliding. Only
edit your own.

## The promote rule

> Prototype here → move it into its stub in `src/edgecnn/` the moment it works → import it from then on.

Move it **immediately** if another member or another notebook needs it, if it is a `Dataset` or
transform (Windows DataLoader workers cannot load classes defined in a notebook), if a contract test
covers it, or before any official run. See
[`CONTRIBUTING.md` → the promote rule](../../CONTRIBUTING.md#the-promote-rule).

## Rules

- **Nothing here ever produces a report result.** Official numbers come from the section notebooks
  `00`–`07`.
- **Commit your scratch notebooks.** The assignment grades sustained development history, and these
  show your work over time. Outputs are kept, like every notebook: keep them small, and never print
  a token or a local path.
- It's fine for scratch notebooks to be messy. They don't need to be tidy — the section notebooks do.
