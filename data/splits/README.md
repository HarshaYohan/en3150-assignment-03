# `data/splits/` — the reproducibility contract

**COMMITTED.** Owner: Member 1. Used by all four members. This is SEAM 1.
Written only by an **official** run of
[`notebooks/01_data_preparation.ipynb`](../../notebooks/01_data_preparation.ipynb).

| File | Contents |
|---|---|
| `split_manifest.csv` | `relative_path, label_index, label_name, split` — one row per image |
| `split_meta.json` | class names (in a frozen order), per-split counts, the seed, the manifest's SHA-256 |
| `norm_stats.json` | per-channel mean and std, fitted on the **train split only** |

## Why these three files are committed

They guarantee that Model A, Model B, MobileNetV2 and SqueezeNet all saw byte-identical data, so any
accuracy difference between them comes from the architecture rather than a lucky split. The whole
comparison rests on a few kilobytes of text.

## Rules

- Drawn **once**: stratified 70/15/15, with split seed 42. `prepare_dataset` needs `force=True` to
  overwrite.
- Redrawing invalidates every result already in `results/metrics/`.
- `split_meta.json` stores the manifest's SHA-256. If it stops matching, the committed results are
  stale.
- The order of `class_names` is frozen. Every confusion matrix and per-class metric is indexed by it.
- Forward slashes in `relative_path`: the team runs Windows, Linux and Colab.
- Never hand-edit. `edgecnn.contracts.schema.validate_manifest` rejects an invalid file, and
  `pytest tests/contracts` fails.
