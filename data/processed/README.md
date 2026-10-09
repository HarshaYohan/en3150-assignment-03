# `data/processed/` — prepared images

**Git-ignored** (except this file). Owner: Member 1.

Images at the final 64×64 resolution, laid out as `eurosat_64/<class_name>/<image>.jpg`. Produced by
`prepare_dataset` (notebook `01`), or by `ensure_images_present` on a fresh clone or in a Colab
session.

EuroSAT is already 64×64, so with `resize_mode: none` this is effectively a copy. The step exists so
that the pipeline still works if the team ever switches dataset.

Each `relative_path` in `data/splits/split_manifest.csv` is relative to the folder named by
`stages.data.outputs.processed_dir` in `configs/stages/data.yaml`.

## The pixel cache

```
data/processed/
├── eurosat_64/<class_name>/*.jpg
└── eurosat_64.pixels-<hash>.npy     every image, decoded once: 27,000 × 64 × 64 × 3 bytes ≈ 330 MB
```

Decoding a JPEG gives the same pixels every time, so the images are decoded once and kept as one
`uint8` array. Every later run loads that array in about a second instead of decoding 27,000 files
per epoch.

- **Only decoding is cached.** Augmentation and normalisation still run fresh on every image, every
  epoch (`edgecnn/data/transforms.py`).
- **It rebuilds itself.** `<hash>` is a fingerprint of the file list, sizes and modification times,
  so any change to the images produces a new cache and the old one is deleted. It is always safe to
  delete by hand.
- **No DataLoader workers.** With the pixels already in memory, a batch costs little more than an
  array slice, so `dataloader.num_workers` is `0` in `configs/base.yaml`. Worker processes would only
  add start-up time and memory, especially on Windows, where each one is a separate Python process.
