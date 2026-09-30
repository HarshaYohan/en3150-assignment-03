# `data/` — SEAM 1 producer

**Owner: Member 1.** Consumed by Members 2, 3 and 4. Assignment §1.
**Notebook:** [`01_data_preparation`](../../../notebooks/01_data_preparation.ipynb); loaders are
used at the top of `03`, `04` and `05`.

Everything about the dataset lives here, and nothing about it lives anywhere else. No other member
opens an image file, computes a normalisation statistic, or decides which sample is in which split.

## Inputs

| From | What |
|---|---|
| `configs/stages/data.yaml` | `dataset`, `split`, `augmentation`, `normalization` |
| EuroSAT via torchvision | 27,000 satellite images, 10 classes, natively 64×64 RGB |

## Outputs

| Path | Written when | Consumed by |
|---|---|---|
| `data/processed/eurosat_64/` | whenever images are prepared (git-ignored) | this package |
| `data/splits/split_manifest.csv` | **official** run of `01` — committed | everyone |
| `data/splits/split_meta.json` | **official** run of `01` — committed | everyone, and the report |
| `data/splits/norm_stats.json` | **official** run of `01` — committed | this package |
| `results/figures/dataset/*.png` | **official** run of `01` — committed | the report (§1) |
| `DataBundle` (in memory) | every call, every mode | Members 2, 3, 4 |

## The contract you provide

```python
data = build_dataloaders(cfg)     # -> DataBundle

images : float32, (B, 3, 64, 64), normalised with TRAIN-ONLY statistics
labels : int64,   (B,),           values in [0, num_classes)
```

`split_manifest.csv` columns, in this order: `relative_path, label_index, label_name, split`.

## Files

| File | Purpose |
|---|---|
| `synthetic.py` | **Build this first.** A tiny generated dataset that satisfies the full contract. `MODE = "synthetic"` selects it, so Members 3 and 4 can start before EuroSAT downloads |
| `loaders.py` | `build_dataloaders(cfg) -> DataBundle` |
| `prepare.py` | `prepare_dataset(cfg, force=False)` draws the split; `ensure_images_present(cfg)` refetches the images on a fresh clone |
| `transforms.py` | augmentation and normalisation pipelines |
| `inspect.py` | the §1 split table, class-distribution figure and sample grid |

## Order of work

1. `make_synthetic_fixture` and the synthetic branch of `build_dataloaders` — **three other people
   are waiting on these.** Also `apply_style` in `evaluation/plotting.py`.
2. `prepare_dataset`, then run `01` in **official** mode and commit the three split files. Announce
   it: that is when everyone switches from `synthetic` to `debug`.
3. `ensure_images_present`. Every teammate's fresh clone, and every Colab session, has the committed
   manifest but no images. This fetches them to match the committed split — **never redraws it**.
4. The `inspect.py` figures for the report.

## Things that silently corrupt results

- Augmenting val or test. Train only.
- Fitting normalisation on anything but train. The schema pins `fitted_on: "train"`.
- Backslashes in `relative_path`. The manifest is committed, and the team runs Windows, Linux and Colab.
- Redrawing the split after results exist. `split_meta.json` stores the manifest's SHA-256, so this
  is detectable, and `prepare_dataset` needs `force=True` to overwrite.
- An unstratified split. A class that lands entirely in train has an undefined test recall, which
  breaks the macro average.
- A `Dataset` class defined in a notebook cell. On Windows, DataLoader workers cannot load it —
  keep it in this package.
