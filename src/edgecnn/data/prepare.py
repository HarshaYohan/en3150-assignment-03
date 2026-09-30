"""Draw the train / validation / test split and write the files that define it.

The split is drawn once and saved as three files: the manifest (which image is
in which split), its metadata, and normalisation statistics computed from the
training images only. Every model then trains and is tested on byte-identical
splits, so an accuracy difference between two models comes from the models,
not from the data.

Team notes:
Owner: Member 1. The EuroSAT-specific functions (prepare_dataset,
ensure_images_present) are implemented in Step 2; the shared helpers below are
also used by the synthetic fixture.

    IN   cfg.section("dataset"), cfg.section("split")
    OUT  data/processed/eurosat_64/       git-ignored, regenerable
         data/splits/split_manifest.csv   COMMITTED } official mode only
         data/splits/split_meta.json      COMMITTED }
         data/splits/norm_stats.json      COMMITTED }
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from edgecnn.contracts import paths, schema
from edgecnn.contracts.schema import MANIFEST_COLUMNS
from edgecnn.contracts.types import SPLIT_NAMES
from edgecnn.utils.io import sha256_file

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from edgecnn.contracts.types import ResolvedConfig


def processed_image_dir(cfg: ResolvedConfig) -> Path:
    """Folder holding a dataset's images at the final resolution.

    ``data/processed/<dataset>_<height>``, e.g. ``data/processed/eurosat_64``.
    Every ``relative_path`` in the manifest is relative to this folder.
    """
    name = cfg.section("dataset").get("name", "eurosat")
    height = int(cfg.raw.get("image_size", [64, 64])[0])
    return paths.PROCESSED_DIR / f"{name}_{height}"


def prepare_dataset(cfg: ResolvedConfig, force: bool = False) -> dict[str, Any]:
    """End-to-end preparation. Idempotent - safe to re-run.

    Steps for Member 1:

    1. Download via ``torchvision.datasets.EuroSAT(root, download=True)``.
    2. Resize to 64x64 if needed. EuroSAT is already 64x64, so with
       ``resize_mode: none`` this is a copy; keep the branch anyway in case
       the team changes dataset.
    3. Draw a STRATIFIED 70/15/15 split seeded from ``cfg.section("split")["seed"]``.
    4. Only if ``cfg.is_official``: write the manifest, metadata and
       normalisation statistics, each validated against its schema first.
       In synthetic / debug mode, return everything for inspection and write
       nothing to ``data/splits/``.

    Args:
        force: An existing committed split is never overwritten unless this
            is True. Redrawing silently would invalidate every result already
            in ``results/metrics/``.

    Returns:
        ``{"manifest": rows, "meta": dict, "norm_stats": dict, "written": [paths]}``
        - returned in every mode so notebook 01 can inspect it inline.
    """
    raise NotImplementedError("Member 1: implement prepare_dataset")


def ensure_images_present(cfg: ResolvedConfig) -> int:
    """Make the images referenced by the committed manifest exist locally.

    The manifest is committed; the images are not. Every fresh clone - a
    teammate's laptop, a Colab session - therefore has a valid split and no
    pictures. This downloads and processes EuroSAT so that every
    ``relative_path`` in the manifest resolves.

    Must NEVER redraw the split. After fetching, recompute the manifest's
    SHA-256 and compare it with ``split_meta.json -> manifest_sha256``; a
    mismatch means the committed split and the local one have diverged.

    Returns:
        Number of images that had to be fetched (0 when already present).
    """
    raise NotImplementedError("Member 1: implement ensure_images_present")


def stratified_split(
    labels: Sequence[int],
    fractions: dict[str, float],
    seed: int,
) -> list[str]:
    """Assign every sample to ``train``, ``val`` or ``test``, class by class.

    Within each class the samples are shuffled, then cut in the given
    proportions (each count rounded to the nearest whole image). Splitting per
    class keeps the class balance identical in all three splits, and guarantees
    every class appears in each of them whenever it has at least three samples.

    The shuffle uses NumPy's legacy ``RandomState``, whose sequence of random
    numbers is frozen across NumPy versions, so the same seed reproduces the
    same split on any machine.

    Args:
        labels: The integer class of each sample, in manifest order.
        fractions: ``{"train": 0.70, "val": 0.15, "test": 0.15}``.
        seed: Random seed.

    Returns:
        The split name of each sample, in the same order as ``labels``.

    Raises:
        ValueError: if the fractions don't name exactly train / val / test, or
            don't sum to 1.
    """
    import numpy as np

    if set(fractions) != set(SPLIT_NAMES):
        raise ValueError(
            f"fractions must name exactly {list(SPLIT_NAMES)}, got {sorted(fractions)}"
        )
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {sum(fractions.values())}")

    labels = np.asarray(labels)
    assignment = np.empty(len(labels), dtype=object)
    rng = np.random.RandomState(seed)
    for label in np.unique(labels):  # sorted, so the random sequence is fixed
        members = rng.permutation(np.flatnonzero(labels == label))
        start = 0
        for name, count in zip(SPLIT_NAMES, _split_sizes(len(members), fractions), strict=True):
            assignment[members[start : start + count]] = name
            start += count
    return assignment.tolist()


def _split_sizes(n: int, fractions: dict[str, float]) -> list[int]:
    """Train / val / test sizes for one class of ``n`` samples."""
    sizes = [round(n * fractions["train"]), round(n * fractions["val"])]
    sizes.append(n - sum(sizes))
    if n >= 3:  # move samples from train so no split is empty
        for index in (1, 2):
            if sizes[index] < 1:
                sizes[0] -= 1 - sizes[index]
                sizes[index] = 1
    return sizes


def write_split_manifest(manifest_path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write ``split_manifest.csv``: one row per image, columns in the fixed order.

    Columns: ``relative_path, label_index, label_name, split``. Paths use forward
    slashes and lines end in ``\\n`` on every platform, so the file - and its
    SHA-256 recorded in the metadata - is identical on Windows, Linux and Colab.
    The written file is validated before returning.

    Raises:
        ContractViolation: if any row breaks the manifest schema.
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MANIFEST_COLUMNS)
        for row in rows:
            writer.writerow([
                str(row["relative_path"]).replace("\\", "/"),
                int(row["label_index"]),
                row["label_name"],
                row["split"],
            ])
    schema.validate_manifest(manifest_path)
    return manifest_path


def read_split_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Read and validate ``split_manifest.csv``; ``label_index`` comes back as ``int``.

    Raises:
        FileNotFoundError: if the manifest does not exist.
        ContractViolation: if it breaks the manifest schema.
    """
    manifest_path = Path(manifest_path)
    schema.validate_manifest(manifest_path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return [
            {**row, "label_index": int(row["label_index"])}
            for row in csv.DictReader(handle)
        ]


def summarize_split(
    rows: Iterable[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Image counts per split, and per class within each split.

    Returns:
        ``({"train": n, "val": n, "test": n, "total": n},
        {"train": {class_name: n, ...}, ...})``.
    """
    counts = {name: 0 for name in SPLIT_NAMES}
    per_class: dict[str, dict[str, int]] = {name: {} for name in SPLIT_NAMES}
    for row in rows:
        counts[row["split"]] += 1
        classes = per_class[row["split"]]
        classes[row["label_name"]] = classes.get(row["label_name"], 0) + 1
    counts["total"] = sum(counts[name] for name in SPLIT_NAMES)
    return counts, per_class


def write_split_meta(meta_path: Path, manifest_path: Path, **fields: Any) -> Path:
    """Write ``split_meta.json``, including the manifest's SHA-256.

    The hash lets anyone confirm that the manifest on disk is the one the
    reported results were produced with: if it stops matching, the split was
    regenerated or edited. ``created_utc`` is filled in unless given.

    Raises:
        ContractViolation: if the metadata breaks its schema.
    """
    payload = dict(fields)
    payload["manifest_sha256"] = sha256_file(Path(manifest_path))
    payload.setdefault("created_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return schema.write_json(Path(meta_path), payload, "split_meta")


def write_norm_stats(
    stats_path: Path,
    mean: Sequence[float],
    std: Sequence[float],
    num_images: int,
) -> Path:
    """Write ``norm_stats.json`` - per-channel statistics of the training images.

    ``fitted_on`` is always ``"train"``: statistics computed on validation or
    test images would leak held-out information into training.
    """
    payload = {
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "fitted_on": "train",
        "num_images": int(num_images),
    }
    return schema.write_json(Path(stats_path), payload, "norm_stats")
