"""A tiny generated stand-in for EuroSAT, for quick end-to-end runs.

``MODE = "synthetic"`` swaps the real data for 200 generated 64x64 images: ten
classes named after EuroSAT's, twenty images each. It has the same file layout,
split files and normalisation statistics as the real data, so every notebook
runs its full pipeline in seconds, before the real dataset has been downloaded.

Each class has its own colour and stripe pattern, plus random noise. A model
can learn to tell them apart, so a run that fails to learn the synthetic data
points to a genuine bug rather than a hard dataset. Its accuracy says nothing
about EuroSAT, so synthetic runs are never official and never write results.

Team notes:
Owner: Member 1. The Phase 0 unblocker. Generated on demand into
tests/fixtures/synthetic/ (git-ignored) by build_dataloaders.
"""

from __future__ import annotations

import colorsys
from pathlib import Path

from edgecnn.contracts.types import SPLIT_FRACTIONS

#: EuroSAT's ten land-cover classes, in the order used for class indices.
EUROSAT_CLASSES: tuple[str, ...] = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)

#: Bumped whenever the generator changes, so older fixtures are regenerated.
FIXTURE_VERSION: int = 2

#: Files this module writes inside the fixture folder (and the only ones it deletes).
IMAGE_DIR_NAME = "images"
FIXTURE_FILES = ("split_manifest.csv", "split_meta.json", "norm_stats.json")


def synthetic_class_names(num_classes: int) -> list[str]:
    """EuroSAT's class names for up to 10 classes, generic names beyond that."""
    if num_classes <= len(EUROSAT_CLASSES):
        return list(EUROSAT_CLASSES[:num_classes])
    return [f"class_{index:02d}" for index in range(num_classes)]


def make_synthetic_fixture(
    root: Path,
    *,
    num_classes: int = 10,
    images_per_class: int = 20,
    image_size: tuple[int, int] = (64, 64),
    seed: int = 0,
) -> Path:
    """Generate the synthetic dataset and its split files under ``root``.

    Writes ``root/images/<class>/<class>_NNN.png``, then ``split_manifest.csv``,
    ``split_meta.json`` and ``norm_stats.json`` beside them, each validated
    against its schema. The output is identical for the same arguments.
    Regenerating replaces only these files, never anything else in ``root``.

    Args:
        root: Output folder.
        num_classes: Number of classes (EuroSAT's names up to 10).
        images_per_class: Images generated per class.
        image_size: ``(height, width)``; at most 64x64, like the real data.
        seed: Seed for both the pixels and the split.

    Returns:
        ``root``.
    """
    import shutil

    import numpy as np
    from PIL import Image

    from edgecnn.data.cache import channel_stats, load_pixels
    from edgecnn.data.prepare import (
        stratified_split,
        summarize_split,
        write_norm_stats,
        write_split_manifest,
        write_split_meta,
    )

    root = Path(root)
    image_dir = root / IMAGE_DIR_NAME
    if image_dir.exists():
        shutil.rmtree(image_dir)
    for name in FIXTURE_FILES:
        (root / name).unlink(missing_ok=True)
    for old_cache in root.glob(f"{IMAGE_DIR_NAME}.pixels-*.npy"):
        old_cache.unlink()

    rng = np.random.RandomState(seed)
    height, width = image_size
    rows_grid, cols_grid = np.mgrid[0:height, 0:width]
    names = synthetic_class_names(num_classes)
    rows = []
    for label, name in enumerate(names):
        (image_dir / name).mkdir(parents=True)
        colour = np.array(colorsys.hsv_to_rgb(label / num_classes, 0.6, 0.75)) * 255
        angle = np.pi * label / num_classes          # stripe direction differs per class
        period = 6 + 2 * (label % 4)                 # and so does stripe spacing
        along = cols_grid * np.cos(angle) + rows_grid * np.sin(angle)
        for index in range(images_per_class):
            stripes = 25 * np.sin(2 * np.pi * along / period + rng.uniform(0, 2 * np.pi))
            noise = rng.normal(0, 18, size=(height, width, 3))
            image = np.clip(colour + stripes[..., None] + noise, 0, 255).astype(np.uint8)
            rel = f"{name}/{name}_{index:03d}.png"
            Image.fromarray(image).save(image_dir / rel)
            rows.append({"relative_path": rel, "label_index": label, "label_name": name})

    rows.sort(key=lambda row: row["relative_path"])
    splits = stratified_split([row["label_index"] for row in rows], dict(SPLIT_FRACTIONS), seed)
    for row, split in zip(rows, splits, strict=True):
        row["split"] = split

    manifest = write_split_manifest(root / "split_manifest.csv", rows)
    pixels = load_pixels(image_dir, [row["relative_path"] for row in rows], image_size)
    train = [position for position, row in enumerate(rows) if row["split"] == "train"]
    mean, std = channel_stats(pixels, train)
    write_norm_stats(root / "norm_stats.json", mean, std, num_images=len(train))

    counts, per_class = summarize_split(rows)
    write_split_meta(
        root / "split_meta.json",
        manifest,
        dataset_name="synthetic",
        dataset_source="generated by edgecnn.data.synthetic",
        num_classes=num_classes,
        class_names=names,
        image_size=[height, width],
        seed=seed,
        stratified=True,
        fractions=dict(SPLIT_FRACTIONS),
        counts=counts,
        per_class_counts=per_class,
        fixture_version=FIXTURE_VERSION,
        images_per_class=images_per_class,
    )
    return root


def fixture_is_current(root: Path) -> bool:
    """True if ``root`` holds a complete fixture made by the current generator."""
    import json

    root = Path(root)
    if not (root / IMAGE_DIR_NAME).is_dir() or not all((root / n).exists() for n in FIXTURE_FILES):
        return False
    try:
        meta = json.loads((root / "split_meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        meta.get("fixture_version") == FIXTURE_VERSION
        and meta.get("num_classes") == len(EUROSAT_CLASSES)
    )
