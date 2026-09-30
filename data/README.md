# `data/`

**Owner: Member 1.** Nobody else writes here. Produced by
[`notebooks/01_data_preparation.ipynb`](../notebooks/01_data_preparation.ipynb).

| Folder | Contents | Committed? |
|---|---|---|
| `raw/` | the original EuroSAT download | **no** — large, and reproducible from `configs/stages/data.yaml` |
| `processed/` | the prepared 64×64 images | **no** — a deterministic transform of `raw/` |
| `splits/` | the split definition | **yes** — this is the reproducibility contract |

## Why `splits/` is committed and the images are not

`split_manifest.csv` is what guarantees Model A, Model B, MobileNetV2 and SqueezeNet saw
byte-identical data, so any accuracy difference between them comes from the architecture, not a
lucky split. It is a small text file: it costs nothing to commit, and everything to lose.

The images are a download plus a deterministic resize, so any machine can recreate them. On a fresh
clone or a Colab session, `build_dataloaders` calls `ensure_images_present`. That fetches the images
the committed manifest points to, and **never redraws the split**.

## `splits/` contents

| File | Purpose |
|---|---|
| `split_manifest.csv` | one row per image: `relative_path, label_index, label_name, split` |
| `split_meta.json` | class names (in a frozen order), per-split counts, the seed, and the manifest's SHA-256 |
| `norm_stats.json` | per-channel mean and std, fitted on the **train split only** |

## Rules

1. **The split is drawn once**, by an official run of notebook `01`. `prepare_dataset` refuses to
   overwrite it without `force=True`, because redrawing it invalidates every result in
   `results/metrics/`.
2. **Synthetic and debug runs never write here.** Only `cfg.is_official` runs do.
3. **`split_meta.json` stores the manifest's SHA-256.** If it stops matching, someone regenerated or
   hand-edited the split, and the committed results are stale.
4. **The order of `class_names` is frozen** once committed. Every confusion matrix, per-class metric
   and figure legend is indexed by it.
5. **Normalisation is fitted on train only.** The schema pins `fitted_on: "train"`.
6. **Forward slashes in `relative_path`.** The team runs Windows, Linux and Colab.

## After Member 1 commits the split

Announce it. From that point, everyone can switch their notebooks from `MODE = "synthetic"` to
`"debug"`.
