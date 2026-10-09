"""Dataset inspection - the Section 1 table and figures.

These produce the report's Section 1 material: how big each split is, whether
the classes are balanced across splits, and what the images look like at 64x64.

Team notes:
Owner: Member 1.  Called from notebooks/01_data_preparation.ipynb.

    IN   the dict returned by prepare_dataset (any mode)
    OUT  a split-size table and two figures - always returned
         + results/figures/dataset/*.png   (write=True only)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from edgecnn.contracts import paths
from edgecnn.contracts.types import SPLIT_NAMES

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

    from edgecnn.contracts.types import ResolvedConfig


def split_counts_table(meta: dict[str, Any]) -> pd.DataFrame:
    """Images per class per split, with totals - the Section 1 split table.

    Built from ``split_meta.json``-shaped metadata (``counts`` and
    ``per_class_counts``). The ``Total`` row should split 70 / 15 / 15; a class
    with zero images in any split means the split was not stratified.
    """
    import pandas as pd

    splits = list(SPLIT_NAMES)
    per_class = meta.get("per_class_counts") or {}
    names = list(meta["class_names"]) if per_class else []  # without them, totals only
    table = pd.DataFrame(
        [[per_class.get(split, {}).get(name, 0) for split in splits] for name in names],
        index=pd.Index(names, name="class"),
        columns=splits,
    )
    table.loc["Total"] = [meta["counts"][split] for split in splits]
    table["total"] = table[splits].sum(axis=1)
    return table.astype(int)


def plot_class_distribution(meta: dict[str, Any], write: bool = False) -> Figure:
    """Grouped bar chart: each class's share of every split.

    Bars of equal height across train, validation and test show that
    stratification preserved the class proportions, which is what lets the
    report use macro-averaged precision and recall without a caveat. The
    absolute counts are in :func:`split_counts_table`. With ``write=True``,
    saves ``results/figures/dataset/class_distribution.png``.
    """
    import numpy as np
    from matplotlib.figure import Figure

    from edgecnn.evaluation.plotting import color_for, save_figure

    classes = list(meta["class_names"])
    positions = np.arange(len(classes))
    width = 0.8 / len(SPLIT_NAMES)
    fig = Figure(figsize=(max(6.0, 0.7 * len(classes) + 1.5), 4.2), layout="constrained")
    ax = fig.add_subplot()
    tallest = 0.0
    for offset, split in enumerate(SPLIT_NAMES):
        total = meta["counts"][split]
        shares = [100 * meta["per_class_counts"][split].get(name, 0) / total for name in classes]
        tallest = max(tallest, *shares)
        ax.bar(
            positions + (offset - (len(SPLIT_NAMES) - 1) / 2) * width, shares, width,
            label=f"{split} ({total:,} images)", color=color_for(split, kind="split"),
        )
    ax.set_xticks(positions, classes, rotation=35, ha="right")
    ax.set_ylabel("share of the split (%)")
    ax.set_ylim(0, tallest * 1.3)  # room for the legend above the bars
    ax.set_title("Class distribution in each split")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncols=len(SPLIT_NAMES))
    if write:
        save_figure(fig, paths.DATASET_FIGURES_DIR / "class_distribution.png")
    return fig


def plot_sample_grid(cfg: ResolvedConfig, per_class: int = 4, write: bool = False) -> Figure:
    """A grid of training images, ``per_class`` per class, one row per class.

    The images come from the TRAIN split, unaugmented and at the stored 64x64
    resolution - what the models receive before normalisation. They are chosen
    with ``cfg.seed``, so the committed figure is reproducible. With
    ``write=True``, saves ``results/figures/dataset/sample_grid.png``.
    """
    import numpy as np
    from matplotlib.figure import Figure
    from PIL import Image

    from edgecnn.data.prepare import _prepare
    from edgecnn.evaluation.plotting import save_figure

    prepared = _prepare(cfg, force=False, write=False, with_stats=False)  # same split, not written
    classes = list(prepared.meta["class_names"])
    rng = np.random.RandomState(cfg.seed)
    fig = Figure(figsize=(0.95 * per_class + 1.9, 0.95 * len(classes) + 0.5), layout="constrained")
    axes = fig.subplots(len(classes), per_class, squeeze=False)
    for label, name in enumerate(classes):
        train = [
            row["relative_path"] for row in prepared.rows
            if row["split"] == "train" and row["label_index"] == label
        ]
        picks = rng.choice(len(train), size=min(per_class, len(train)), replace=False)
        for column, ax in enumerate(axes[label]):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if column < len(picks):
                with Image.open(prepared.image_dir / train[picks[column]]) as image:
                    ax.imshow(np.asarray(image.convert("RGB")), interpolation="nearest")
        axes[label][0].set_ylabel(name, rotation=0, ha="right", va="center", labelpad=6)
    fig.suptitle("Training images at 64×64, as the models receive them")
    if write:
        save_figure(fig, paths.DATASET_FIGURES_DIR / "sample_grid.png")
    return fig
