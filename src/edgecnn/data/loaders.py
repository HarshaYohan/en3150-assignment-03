"""Build the three DataLoaders that every other member consumes.

Owner: Member 1.  This file *is* Seam 1. Called at the top of notebooks
03, 04 and 05 (and inspected in 01).

    +---------------------------------------------------------------------+
    |  IN   cfg : ResolvedConfig, loaded with the notebook's MODE          |
    |       data/splits/split_manifest.csv   - which image, which split    |
    |       data/splits/split_meta.json      - class names, counts, seed   |
    |       data/splits/norm_stats.json      - train-only mean/std         |
    |                                                                      |
    |  OUT  DataBundle                                                     |
    |       batches: images float32 (B,3,64,64) normalised                 |
    |                labels int64  (B,) in [0, num_classes)                |
    +---------------------------------------------------------------------+

Reads, never writes: nothing here depends on the run being official.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from edgecnn.contracts.types import DataBundle

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from edgecnn.contracts.types import ResolvedConfig


def build_dataloaders(cfg: ResolvedConfig) -> DataBundle:
    """Construct train/val/test loaders from the committed split manifest.

    Implementation notes for Member 1:

    * **Synthetic first.** ``cfg.section("dataset")["name"] == "synthetic"``
      (what ``MODE = "synthetic"`` sets) must route to
      :func:`edgecnn.data.synthetic.make_synthetic_fixture`, generating it into
      ``paths.SYNTHETIC_FIXTURE_DIR`` if absent. That branch is what unblocks
      Members 3 and 4 before EuroSAT is downloaded, so build it first.
    * Read the manifest with pandas and filter by the ``split`` column. Do NOT
      re-draw the split here - it is drawn once in notebook 01 and committed,
      so every member and every run sees identical data.
    * **Fresh clones.** On a new machine or a Colab session the manifest is
      committed but the images are not. Call
      :func:`edgecnn.data.prepare.ensure_images_present` first: it downloads
      and processes EuroSAT to match the committed manifest, and never redraws.
    * Augmentation applies to the train loader ONLY (``augmentation.train``).
      Augmenting val or test makes the reported numbers unreproducible.
    * ``shuffle=True`` for train, ``False`` for val and test. Shuffling the
      test loader makes a sample-by-sample error analysis impossible to redo.
    * Seed the loader generator from ``cfg.seed`` and pass
      ``utils.seed.worker_init_fn`` so augmentation is reproducible.
    * ``num_workers`` from ``cfg.section("dataloader")``, capped at
      ``os.cpu_count()`` - Colab has 2 CPUs. The Dataset class must live in a
      ``.py`` module: on Windows, workers cannot unpickle a class defined in a
      notebook cell.
    * Honour ``cfg.raw["subset_fraction"]`` by subsampling the TRAIN split
      only, stratified. ``MODE = "debug"`` sets it to 0.05.
    * Pretrained runs may set ``pretrained.normalization.use_imagenet_stats``;
      then use the ImageNet constants and record them in the returned bundle's
      ``norm_mean`` / ``norm_std``, so downstream code and the report agree on
      what was actually applied.

    Raises:
        FileNotFoundError: if the manifest is missing - point the caller at
            notebook 01 (official mode), or suggest ``MODE = "synthetic"``.
        ContractViolation: if the manifest or metadata fails its schema.
    """
    raise NotImplementedError(
        "Member 1: implement build_dataloaders. "
        "Ship the dataset.name == 'synthetic' branch FIRST - Members 3 and 4 "
        "are blocked on it. See tests/fixtures/README.md."
    )


def build_single_loader(cfg: ResolvedConfig, split: str) -> DataLoader:
    """Build one loader in isolation.

    Convenience for inspection cells and error analysis. ``split`` must be one
    of :data:`edgecnn.contracts.types.SPLIT_NAMES`.
    """
    raise NotImplementedError("Member 1: implement build_single_loader")


def compute_norm_stats(cfg: ResolvedConfig) -> dict[str, list[float]]:
    """Compute per-channel mean and std over the TRAIN split only.

    Called once by ``prepare_dataset`` and the result written to
    ``norm_stats.json``. Fitting on val or test leaks held-out information
    into training - ``norm_stats.schema.json`` pins ``fitted_on`` to
    ``"train"`` so the mistake cannot be committed silently.

    Returns:
        ``{"mean": [r, g, b], "std": [r, g, b]}`` over pixels scaled to [0, 1].
    """
    raise NotImplementedError("Member 1: implement compute_norm_stats")
