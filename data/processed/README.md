# `data/processed/` — prepared images

**Git-ignored** (except this file). Owner: Member 1.

Images at the final 64×64 resolution, laid out as `eurosat_64/<class_name>/<image>.jpg`. Produced by
`prepare_dataset` (notebook `01`), or by `ensure_images_present` on a fresh clone or in a Colab
session.

EuroSAT is already 64×64, so with `resize_mode: none` this is effectively a copy. The step exists so
that the pipeline still works if the team ever switches dataset.

Each `relative_path` in `data/splits/split_manifest.csv` is relative to the folder named by
`stages.data.outputs.processed_dir` in `configs/stages/data.yaml`.
