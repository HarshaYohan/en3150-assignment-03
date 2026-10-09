"""Build the train, validation and test DataLoaders for a run.

``build_dataloaders(cfg)`` is the only way data reaches a model. It reads the
split manifest, loads the decoded pixels, and returns a ``DataBundle`` holding
the three loaders together with the class names, input shape and normalisation
used. Every batch has the same form for every model::

    images : float32, (B, 3, H, W), normalised
    labels : int64,   (B,), values in [0, num_classes)

Team notes:
Owner: Member 1. This is Seam 1 - called at the top of notebooks 03, 04 and 05.

    IN   cfg (loaded with the notebook's MODE)
         data/splits/split_manifest.csv, split_meta.json, norm_stats.json
         (or the synthetic fixture's copies in MODE = "synthetic")
    OUT  DataBundle
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from edgecnn.contracts import paths, schema
from edgecnn.contracts.schema import ContractViolation
from edgecnn.contracts.types import SPLIT_NAMES, DataBundle
from edgecnn.data.cache import channel_stats, load_pixels
from edgecnn.data.prepare import ensure_images_present, processed_image_dir, read_split_manifest
from edgecnn.data.synthetic import IMAGE_DIR_NAME, fixture_is_current, make_synthetic_fixture
from edgecnn.data.transforms import build_transform
from edgecnn.utils.logging import get_logger
from edgecnn.utils.seed import worker_init_fn

if TYPE_CHECKING:
    from edgecnn.contracts.types import ResolvedConfig

log = get_logger(__name__)


class PixelDataset(Dataset):
    """Images from the decoded-pixel array, returned as ``(normalised tensor, label)``.

    Holds only indices into the shared array, so the three splits share one copy
    of the pixels in memory.
    """

    def __init__(self, pixels: np.ndarray, indices: np.ndarray, labels: np.ndarray, transform: Any):
        self.pixels = pixels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        image = torch.from_numpy(self.pixels[self.indices[item]]).permute(2, 0, 1)  # HWC -> CHW
        return self.transform(image), int(self.labels[item])


@dataclass(frozen=True)
class _Source:
    """Where one dataset's images and split files live."""

    image_dir: Path
    manifest: Path
    meta: Path
    norm_stats: Path


@dataclass(frozen=True)
class _Loaded:
    """A dataset read from disk: its rows, metadata, statistics and pixels."""

    source: _Source
    meta: dict[str, Any]
    stats: dict[str, Any]
    class_names: list[str]
    labels: np.ndarray
    split_indices: dict[str, np.ndarray]
    pixels: np.ndarray


def build_dataloaders(cfg: ResolvedConfig) -> DataBundle:
    """The three DataLoaders for ``cfg``, plus everything a model needs to know.

    - **Data:** the committed EuroSAT split, or the generated synthetic dataset
      when ``dataset.name`` is ``"synthetic"`` (what ``MODE = "synthetic"`` sets);
      the synthetic data is generated on first use.
    - **Augmentation:** ``augmentation.train`` on the training split only;
      validation and test are never augmented.
    - **Normalisation:** the training split's own statistics, or ImageNet's for
      pretrained backbones configured with ``use_imagenet_stats``.
    - **Size:** the model's input size - 64x64, or larger for the pretrained
      resolution ablation.
    - **Order:** the training loader reshuffles every epoch with a generator
      seeded from ``cfg.seed``; validation and test keep a fixed order.
    - **Debug runs:** ``subset_fraction`` below 1 keeps that fraction of each
      training class; validation and test stay whole.

    Raises:
        FileNotFoundError: if no split has been committed yet. Run notebook 01
            in official mode, or use ``MODE = "synthetic"``.
        ContractViolation: if the split files break their schemas or disagree
            with each other.
    """
    from edgecnn.models.registry import model_input_shape

    loaded = _load(cfg)
    train = loaded.split_indices["train"]
    fraction = float(cfg.raw.get("subset_fraction", 1.0))
    if fraction < 1.0:
        train = _stratified_subset(train, loaded.labels, fraction, cfg.seed)
    indices = {**loaded.split_indices, "train": train}

    mean, std = _normalisation(cfg, loaded.stats)
    channels, height, width = model_input_shape(cfg)
    source_size = tuple(loaded.meta["image_size"])
    augmentation = cfg.section("augmentation")
    options = _loader_options(cfg)

    loaders = {}
    for split in SPLIT_NAMES:
        transform = build_transform(
            split, mean, std, augmentation.get(split) or [], (height, width),
            source_size=source_size,
        )
        rows = indices[split]
        dataset = PixelDataset(loaded.pixels, rows, loaded.labels[rows], transform)
        is_train = split == "train"
        loaders[split] = DataLoader(
            dataset,
            shuffle=is_train,
            generator=torch.Generator().manual_seed(cfg.seed) if is_train else None,
            drop_last=is_train and bool(cfg.section("dataloader").get("drop_last", False)),
            **options,
        )

    bundle = DataBundle(
        train=loaders["train"],
        val=loaders["val"],
        test=loaders["test"],
        num_classes=len(loaded.class_names),
        class_names=list(loaded.class_names),
        input_shape=(channels, height, width),
        norm_mean=tuple(mean),
        norm_std=tuple(std),
        split_manifest_path=loaded.source.manifest,
        seed=cfg.seed,
    )
    log.info(
        "Data: %s | train %d, val %d, test %d images",
        bundle.describe(), len(indices["train"]), len(indices["val"]), len(indices["test"]),
    )
    return bundle


def build_single_loader(cfg: ResolvedConfig, split: str) -> DataLoader:
    """One split's DataLoader, built exactly as :func:`build_dataloaders` builds it.

    Raises:
        ValueError: if ``split`` is not ``"train"``, ``"val"`` or ``"test"``.
    """
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {list(SPLIT_NAMES)}, not {split!r}")
    return getattr(build_dataloaders(cfg), split)


def compute_norm_stats(cfg: ResolvedConfig) -> dict[str, list[float]]:
    """Per-channel mean and standard deviation of the TRAINING images, on a 0-1 scale.

    Only the training split is used: statistics that included validation or test
    images would leak held-out information into training.

    Returns:
        ``{"mean": [r, g, b], "std": [r, g, b]}``.
    """
    loaded = _load(cfg)
    mean, std = channel_stats(loaded.pixels, loaded.split_indices["train"])
    return {"mean": mean, "std": std}


# --- internals ----------------------------------------------------------------


def _source(cfg: ResolvedConfig) -> _Source:
    """Locate the split files, generating the synthetic dataset if it is needed."""
    if cfg.section("dataset").get("name") == "synthetic":
        root = paths.SYNTHETIC_FIXTURE_DIR
        if not fixture_is_current(root):
            log.info("Generating the synthetic dataset in %s", root)
            make_synthetic_fixture(root)
        return _Source(
            root / IMAGE_DIR_NAME,
            root / "split_manifest.csv",
            root / "split_meta.json",
            root / "norm_stats.json",
        )
    if not paths.SPLIT_MANIFEST.exists():
        raise FileNotFoundError(
            "No split has been committed yet (data/splits/split_manifest.csv is missing). "
            "Run notebooks/01_data_preparation.ipynb in official mode, or use MODE = 'synthetic'."
        )
    return _Source(
        processed_image_dir(cfg), paths.SPLIT_MANIFEST, paths.SPLIT_META, paths.NORM_STATS
    )


def _load(cfg: ResolvedConfig) -> _Loaded:
    """Read the split files and the pixels for ``cfg``'s dataset."""
    source = _source(cfg)
    rows = read_split_manifest(source.manifest)
    meta = schema.read_json(source.meta, "split_meta")
    stats = schema.read_json(source.norm_stats, "norm_stats")
    class_names = list(meta["class_names"])
    for row in rows:
        index = row["label_index"]
        if index >= len(class_names) or class_names[index] != row["label_name"]:
            raise ContractViolation(
                f"{source.manifest}: {row['relative_path']} is labelled "
                f"{index}={row['label_name']!r}, but split_meta.json says class {index} is "
                f"{class_names[index] if index < len(class_names) else 'out of range'!r}"
            )

    relative_paths = [row["relative_path"] for row in rows]
    if not all((source.image_dir / rel).exists() for rel in relative_paths):
        ensure_images_present(cfg)
    pixels = load_pixels(source.image_dir, relative_paths, tuple(meta["image_size"]))

    splits = np.array([row["split"] for row in rows])
    return _Loaded(
        source=source,
        meta=meta,
        stats=stats,
        class_names=class_names,
        labels=np.array([row["label_index"] for row in rows], dtype=np.int64),
        split_indices={name: np.flatnonzero(splits == name) for name in SPLIT_NAMES},
        pixels=pixels,
    )


def _stratified_subset(
    indices: np.ndarray, labels: np.ndarray, fraction: float, seed: int
) -> np.ndarray:
    """Keep ``fraction`` of each class (at least one image), chosen reproducibly."""
    rng = np.random.RandomState(seed)
    kept = []
    for label in np.unique(labels[indices]):
        members = indices[labels[indices] == label]
        count = max(1, round(len(members) * fraction))
        kept.append(rng.choice(members, size=count, replace=False))
    return np.sort(np.concatenate(kept))


def _normalisation(cfg: ResolvedConfig, stats: dict[str, Any]) -> tuple[list[float], list[float]]:
    """ImageNet statistics for pretrained backbones that ask for them, else the train split's."""
    pretrained = cfg.section("pretrained")
    backbones = pretrained.get("backbones", []) or []
    is_backbone = any(entry.get("registry_key") == cfg.model_name for entry in backbones)
    normalization = pretrained.get("normalization", {}) or {}
    if is_backbone and normalization.get("use_imagenet_stats"):
        return list(normalization["imagenet_mean"]), list(normalization["imagenet_std"])
    return list(stats["mean"]), list(stats["std"])


def _loader_options(cfg: ResolvedConfig) -> dict[str, Any]:
    """DataLoader keyword arguments from the ``dataloader`` settings."""
    settings = cfg.section("dataloader")
    workers = min(int(settings.get("num_workers", 0)), os.cpu_count() or 1)
    options: dict[str, Any] = {
        "batch_size": int(settings.get("batch_size", 64)),
        "num_workers": workers,
        "pin_memory": bool(settings.get("pin_memory", False)) and torch.cuda.is_available(),
    }
    if workers > 0:  # PyTorch rejects these options without worker processes
        options["persistent_workers"] = bool(settings.get("persistent_workers", False))
        options["worker_init_fn"] = worker_init_fn
    return options
