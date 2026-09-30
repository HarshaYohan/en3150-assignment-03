"""Decode every image once and keep the pixels as one array.

Opening a JPEG and decompressing it into pixels is a fixed step - nothing in it
is learned - so there is no reason to repeat it every epoch. The images are
decoded once into a single ``uint8`` array (about 330 MB for EuroSAT), saved
beside the image folder, and loaded straight back on later runs. Random
augmentation and normalisation are still applied fresh every epoch, on top of
these raw pixels.

The cache is rebuilt automatically whenever any image file is added, removed
or modified, and its file name encodes which images it holds, so it can never
be stale.

Team notes:
Owner: Member 1. Git-ignored like everything under data/processed and
tests/fixtures/synthetic.
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from edgecnn.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

log = get_logger(__name__)


def cache_path_for(image_dir: Path, relative_paths: Sequence[str]) -> Path:
    """Where the pixel cache for these images lives.

    The name carries a fingerprint of every image's path, size and modification
    time, so changing any image points to a new cache file.

    Raises:
        FileNotFoundError: listing the first missing image, if any are missing.
    """
    image_dir = Path(image_dir)
    digest = hashlib.sha256()
    missing = []
    for rel in relative_paths:
        try:
            stat = (image_dir / rel).stat()
        except FileNotFoundError:
            missing.append(rel)
            continue
        digest.update(f"{rel}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image(s) listed in the manifest are missing from {image_dir}, "
            f"e.g. {missing[0]!r}"
        )
    return image_dir.parent / f"{image_dir.name}.pixels-{digest.hexdigest()[:16]}.npy"


def load_pixels(
    image_dir: Path,
    relative_paths: Sequence[str],
    image_size: tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Pixels of every listed image as one ``(N, H, W, 3)`` ``uint8`` array, in list order.

    Loads the cache when it matches the images; otherwise decodes them all,
    saves a fresh cache and deletes any older cache for the same folder.

    Args:
        image_dir: Folder the relative paths are relative to.
        relative_paths: Images to load, in the order the rows should appear.
        image_size: Expected ``(height, width)`` of every image.

    Raises:
        FileNotFoundError: if an image is missing.
        ValueError: if an image is not ``image_size`` - resize the dataset first.
    """
    import numpy as np

    image_dir = Path(image_dir)
    cache = cache_path_for(image_dir, relative_paths)
    height, width = image_size
    if cache.exists():
        pixels = np.load(cache)
        if pixels.shape == (len(relative_paths), height, width, 3):
            return pixels

    log.info("Decoding %d images from %s (once; cached afterwards)", len(relative_paths), image_dir)
    pixels = np.empty((len(relative_paths), height, width, 3), dtype=np.uint8)

    def decode(position: int) -> None:
        from PIL import Image

        rel = relative_paths[position]
        with Image.open(image_dir / rel) as image:
            array = np.asarray(image.convert("RGB"))
        if array.shape != (height, width, 3):
            raise ValueError(
                f"{rel} is {array.shape[1]}x{array.shape[0]}, expected {width}x{height}; "
                f"resize the dataset first (dataset.resize_mode)"
            )
        pixels[position] = array

    workers = min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(decode, range(len(relative_paths))))

    for old in image_dir.parent.glob(f"{image_dir.name}.pixels-*.npy"):
        old.unlink(missing_ok=True)
    temporary = cache.with_name(cache.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, pixels)
    os.replace(temporary, cache)
    return pixels


def channel_stats(
    pixels: np.ndarray,
    indices: Sequence[int],
    chunk: int = 1024,
) -> tuple[list[float], list[float]]:
    """Per-channel mean and standard deviation of the selected images, on a 0-1 scale.

    Computed in chunks with 64-bit accumulators, so even the full training set
    never needs more than a few tens of MB of extra memory.

    Returns:
        ``(mean, std)``, each a list of 3 floats.
    """
    import numpy as np

    indices = np.asarray(indices)
    if len(indices) == 0:
        raise ValueError("no images selected for the statistics")
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    for start in range(0, len(indices), chunk):
        block = pixels[indices[start : start + chunk]].astype(np.float64) / 255.0
        total += block.sum(axis=(0, 1, 2))
        total_sq += np.square(block).sum(axis=(0, 1, 2))
    count = len(indices) * pixels.shape[1] * pixels.shape[2]
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - np.square(mean), 0.0))
    return mean.tolist(), std.tolist()
