"""Dataset inspection - the Section 1 table and figures.

Owner: Member 1.  Called from notebooks/01_data_preparation.ipynb.

    +---------------------------------------------------------------------+
    |  IN   the dict returned by prepare_dataset (any mode)                |
    |  OUT  a split-size table and two figures - always returned          |
    |       + results/figures/dataset/*.png   (write=True only)            |
    +---------------------------------------------------------------------+

These produce the report's Section 1 material: how big each split is, whether
the classes are balanced across splits, and what the images look like at 64x64.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

    from edgecnn.contracts.types import ResolvedConfig


def split_counts_table(meta: dict[str, Any]) -> pd.DataFrame:
    """Images per class per split, with totals - the Section 1 split table.

    Built from ``split_meta.json``-shaped metadata (``counts`` and
    ``per_class_counts``). The totals row must show the 70/15/15 proportions;
    a class with zero images in any split means the split was not stratified.
    """
    raise NotImplementedError("Member 1: implement split_counts_table")


def plot_class_distribution(meta: dict[str, Any], write: bool = False) -> Figure:
    """Grouped bar chart: images per class, one bar per split.

    Shows at a glance that stratification preserved class proportions, which is
    what lets the report use macro-averaged precision and recall without a
    caveat. With ``write=True``, saves
    ``paths.DATASET_FIGURES_DIR / "class_distribution.png"``.
    """
    raise NotImplementedError("Member 1: implement plot_class_distribution")


def plot_sample_grid(cfg: ResolvedConfig, per_class: int = 4, write: bool = False) -> Figure:
    """A grid of training images, ``per_class`` per class, labelled.

    Draw from the TRAIN split only, without augmentation, and pick images
    deterministically from ``cfg.seed`` so the committed figure is
    reproducible. With ``write=True``, saves
    ``paths.DATASET_FIGURES_DIR / "sample_grid.png"``.
    """
    raise NotImplementedError("Member 1: implement plot_sample_grid")
