"""Per-image transforms: augmentation for training, then scaling and normalisation.

One rule makes the reported numbers honest: random augmentation is applied to
the **training** split only. Validation and test images are only resized (if a
model needs another size), scaled to 0-1 and normalised, so they are identical
every time they are evaluated.

Team notes:
Owner: Member 1.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from edgecnn.contracts.types import SPLIT_NAMES

if TYPE_CHECKING:
    import torch
    from torchvision.transforms import v2

#: Augmentations a config may list under ``augmentation.train``. Vertical flips
#: and 90-degree rotations suit EuroSAT because satellite images have no
#: natural "up"; they would be wrong for street-level photographs.
SUPPORTED_AUGMENTATIONS: tuple[str, ...] = (
    "random_horizontal_flip",
    "random_vertical_flip",
    "random_rotation_90",
    "random_resized_crop",
    "color_jitter",
)


class RandomRot90:
    """Rotate an image by 0, 90, 180 or 270 degrees, chosen at random.

    A 90-degree rotation of a square image loses no pixels, unlike an arbitrary
    angle, which would need corners to be filled in.
    """

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        import torch

        quarter_turns = int(torch.randint(0, 4, (1,)).item())
        return torch.rot90(image, quarter_turns, dims=(-2, -1))

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def build_transform(
    split: str,
    norm_mean: tuple[float, ...],
    norm_std: tuple[float, ...],
    augmentations: list[str] | None = None,
    image_size: tuple[int, int] = (64, 64),
    **options: Any,
) -> v2.Transform:
    """The transform pipeline for one split, taking a ``uint8`` ``(3, H, W)`` tensor.

    Output is always a ``float32`` ``(3, *image_size)`` tensor, normalised with
    the given statistics.

    Args:
        split: ``"train"``, ``"val"`` or ``"test"``.
        norm_mean: Per-channel mean on a 0-1 scale.
        norm_std: Per-channel standard deviation on a 0-1 scale.
        augmentations: Names from :data:`SUPPORTED_AUGMENTATIONS`, applied in
            the given order. Ignored, with a warning, for validation and test.
        image_size: Output ``(height, width)``.
        **options: ``source_size`` - the stored image size, so a resize is only
            added when a model needs a different size; ``crop_scale`` (default
            ``(0.8, 1.0)``); ``jitter`` (default ``0.2``).

    Raises:
        ValueError: for an unknown split or augmentation name. A misspelt name
            must fail loudly: silently skipping it would only show up later as
            an unexplained accuracy gap.
    """
    import torch
    from torchvision.transforms import v2

    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {list(SPLIT_NAMES)}, not {split!r}")
    augmentations = list(augmentations or [])
    unknown = [name for name in augmentations if name not in SUPPORTED_AUGMENTATIONS]
    if unknown:
        raise ValueError(
            f"unknown augmentation(s) {unknown}; supported: {list(SUPPORTED_AUGMENTATIONS)}"
        )
    if split != "train" and augmentations:
        warnings.warn(
            f"augmentation {augmentations} ignored for the {split} split - "
            "it applies to train only",
            stacklevel=2,
        )
        augmentations = []

    size = [int(image_size[0]), int(image_size[1])]
    source_size = tuple(options.get("source_size", image_size))
    steps: list[Any] = []
    if tuple(size) != tuple(source_size) and "random_resized_crop" not in augmentations:
        steps.append(v2.Resize(size, antialias=True))
    jitter = float(options.get("jitter", 0.2))
    for name in augmentations:
        if name == "random_horizontal_flip":
            steps.append(v2.RandomHorizontalFlip(p=0.5))
        elif name == "random_vertical_flip":
            steps.append(v2.RandomVerticalFlip(p=0.5))
        elif name == "random_rotation_90":
            steps.append(RandomRot90())
        elif name == "random_resized_crop":
            scale = tuple(options.get("crop_scale", (0.8, 1.0)))
            steps.append(v2.RandomResizedCrop(size, scale=scale, antialias=True))
        elif name == "color_jitter":
            steps.append(
                v2.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=0.02)
            )
    steps.append(v2.ToDtype(torch.float32, scale=True))
    steps.append(v2.Normalize(mean=list(norm_mean), std=list(norm_std)))
    return v2.Compose(steps)
