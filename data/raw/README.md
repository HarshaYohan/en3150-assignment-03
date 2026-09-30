# `data/raw/` — original dataset

**Git-ignored** (except this file). Owner: Member 1.

EuroSAT is downloaded here by `torchvision.datasets.EuroSAT(root="data/raw", download=True)`:
27,000 satellite images, 10 classes, natively 64×64 RGB. This happens the first time you run
[`notebooks/01_data_preparation.ipynb`](../../notebooks/01_data_preparation.ipynb). On a fresh clone
it also happens automatically, through `ensure_images_present`, the first time any notebook loads
data.

Never edit anything in here by hand. This folder is a plain download; `data/processed/` is a
deterministic transform of it; `data/splits/` is the committed output everyone uses.
