# `tests/fixtures/` — the Phase 0 unblocker

**Owner: Member 1. Build this before anything else in the data stage.**

## Why this is the first thing Member 1 writes

The work split puts Members 3 and 4 downstream of Members 1 and 2. Without a fixture they would wait
for the dataset download, then a real loader, then real models — potentially days of two people
sitting idle.

With it, nobody waits. The fixture satisfies the **full Seam 1 contract** — same shapes, same dtypes,
same manifest columns, same `DataBundle` — using generated patterns instead of satellite images.
Member 3 can run a complete training loop in notebook `03` within the first hour.

## Using it

In any notebook, set:

```python
MODE = "synthetic"
```

`load_config(..., mode=MODE)` then sets `dataset.name = "synthetic"` and `training.epochs = 2`.
`build_dataloaders` generates the fixture into `tests/fixtures/synthetic/` on first use and serves it
from there.

## What `make_synthetic_fixture` produces

```
tests/fixtures/synthetic/          git-ignored: deterministic, so cheaper to regenerate than commit
├── images/<class_name>/*.png      200 generated 64x64 RGB images: EuroSAT's 10 class names, 20 each
├── images.pixels-<hash>.npy       the decoded pixels, as for real data (data/processed/README.md)
├── split_manifest.csv             a real manifest with the real columns, 70/15/15 (140/30/30)
├── split_meta.json                real metadata
└── norm_stats.json                real train-only statistics
```

The classes carry EuroSAT's names, so synthetic runs already have the real shapes: a 10-way
classifier head, 10×10 confusion matrices, per-class tables with the real labels.

It is deterministic for a given seed, so two members debugging the same failure see the same batches.
It is small enough that a full run finishes in seconds on a CPU.

## How it is built

- **Each class has its own signature:** a colour, plus stripes at a class-specific angle and spacing,
  under per-pixel noise. A model that cannot learn this is genuinely broken, so the fixture works as
  a real smoke test, not just a shape check.
- **The real writers produce its files.** The split, manifest, metadata and statistics come from the
  same functions as the EuroSAT split, and are validated against `configs/contracts/*.schema.json`.
- **Fully deterministic** for a given `seed`.
- **Regenerated automatically** when the generator changes: `split_meta.json` records a
  `fixture_version`, and an outdated fixture is rebuilt on its next use. To rebuild it by hand,
  delete `tests/fixtures/synthetic/`.

## Synthetic results can never be committed

Synthetic runs are never official (`cfg.is_official` is False), so the library never writes them into
`results/`. The class structure is invented, so their accuracy means nothing.
