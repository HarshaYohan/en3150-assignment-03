"""SEAM 2 - the model registry.

    +---------------------------------------------------------------------+
    |  SHARED FILE.  Member 2 owns it; Member 4 appends pretrained         |
    |  entries.  PR + 1 review.  Coordinate before editing - this is the   |
    |  one file two members write to.                                      |
    +---------------------------------------------------------------------+

Why an indirection instead of just importing the model
-------------------------------------------------------
Member 3's trainer must be identical for Model A, Model B, MobileNetV2 and
SqueezeNet. If the trainer imported model modules directly it would grow a
branch per architecture, and the four models would drift apart in ways that
make the Section 6 table impossible to defend - different loss reductions,
different eval-mode handling, different timing points.

So training code knows exactly one thing: the registry. Members 2 and 4
register builders; the trainer never learns which is which.

How notebooks build a model
---------------------------
Always through :func:`build_model_from_config`::

    model = build_model_from_config(cfg, num_classes=data.num_classes)

It finds the model's settings in the config, seeds, and builds. Notebook 03
(training) and notebook 06 (benchmarking) therefore build the exact same
architecture from the exact same settings - hand-assembling the overrides in a
notebook cell is how the trained model and the profiled model would drift.

    +---------------------------------------------------------------------+
    |  IN   name         registry key, e.g. "model_b"                      |
    |       num_classes  from DataBundle.num_classes                       |
    |       input_shape  (3, 64, 64); pretrained.input_resolution for SOTA |
    |       seed         weight-init seed, normally cfg.seed               |
    |       **overrides  the model's settings section of the config        |
    |                                                                      |
    |  OUT  torch.nn.Module satisfying contracts.protocols.ClassifierModel |
    |       forward: (B,3,64,64) float32 -> (B,num_classes) float32 LOGITS |
    +---------------------------------------------------------------------+
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from torch import nn

    from edgecnn.contracts.types import ResolvedConfig

#: Registry key -> builder. Populated by the ``@register`` decorator when
#: ``edgecnn.models.custom`` and ``edgecnn.models.pretrained`` are imported.
_REGISTRY: dict[str, Callable[..., nn.Module]] = {}

#: Keys the assignment requires. Checked by tests/contracts/test_seams.py so a
#: missing model is a red test, not a surprise during report week.
REQUIRED_KEYS: tuple[str, ...] = (
    "model_a",         # Member 2 - Section 2 standard CNN
    "model_b",         # Member 2 - Section 2 depthwise-separable, <=100k params
    "mobilenet_v2",    # Member 4 - Section 5 SOTA backbone 1
    "squeezenet1_1",   # Member 4 - Section 5 SOTA backbone 2
)


class ModelNotFound(KeyError):
    """Raised when a config names a model key nobody registered."""


def register(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """Decorator registering a builder under ``name``.

    Usage, in ``models/custom/model_b.py``::

        @register("model_b")
        def build_model_b(num_classes, input_shape, **overrides) -> nn.Module:
            ...

    Raises:
        ValueError: on a duplicate key. Two members silently registering the
            same name would make which model you got depend on import order.
    """

    def decorator(builder: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        if name in _REGISTRY:
            raise ValueError(
                f"model key {name!r} is already registered by "
                f"{_REGISTRY[name].__module__}. Pick a different key and tell the team."
            )
        _REGISTRY[name] = builder
        return builder

    return decorator


def build_model(
    name: str,
    num_classes: int,
    input_shape: tuple[int, int, int] = (3, 64, 64),
    *,
    seed: int | None = None,
    **overrides: Any,
) -> nn.Module:
    """Construct a registered model.

    Every returned module must satisfy
    :class:`edgecnn.contracts.protocols.ClassifierModel`: raw logits out, no
    softmax, ``.num_classes`` exposed.

    Args:
        seed: If given, torch's RNG is seeded immediately before construction,
            so the initial weights are identical no matter which notebook cells
            ran earlier. The three Section 3 optimizer runs rely on this to start
            from the same weights - otherwise the comparison measures
            initialisation noise as well as the optimizer.

    Raises:
        ModelNotFound: listing the available keys, so a typo in a config is a
            one-line fix rather than a hunt.
    """
    _import_builders()
    if name not in _REGISTRY:
        raise ModelNotFound(
            f"no model registered as {name!r}. Available: {sorted(_REGISTRY)}"
        )
    if seed is not None:
        import torch

        torch.manual_seed(seed)  # weight init draws only from torch's RNG
    return _REGISTRY[name](num_classes=num_classes, input_shape=input_shape, **overrides)


def model_settings(cfg: ResolvedConfig, name: str | None = None) -> dict[str, Any]:
    """The settings section for a model, as builder keyword overrides.

    * custom models   -> ``cfg.section("model_a")`` / ``cfg.section("model_b")``
    * pretrained ones -> the ``pretrained.backbones`` entry whose
      ``registry_key`` matches, minus that key, plus ``input_resolution``

    Returns an empty dict when a model has no settings section.
    """
    name = name or cfg.model_name
    pretrained = cfg.section("pretrained")
    for entry in pretrained.get("backbones", []) or []:
        if entry.get("registry_key") == name:
            settings = {k: v for k, v in entry.items() if k != "registry_key"}
            settings["input_resolution"] = pretrained.get("input_resolution", 64)
            return settings
    return dict(cfg.section(name))


def model_input_shape(cfg: ResolvedConfig, name: str | None = None) -> tuple[int, int, int]:
    """(C, H, W) this model is fed. 64x64 unless a pretrained ablation raises it."""
    name = name or cfg.model_name
    channels = int(cfg.raw.get("channels", 3))
    height, width = cfg.raw.get("image_size", [64, 64])
    is_pretrained = any(
        entry.get("registry_key") == name
        for entry in cfg.section("pretrained").get("backbones", []) or []
    )
    if is_pretrained:
        resolution = int(cfg.section("pretrained").get("input_resolution", height))
        return (channels, resolution, resolution)
    return (channels, int(height), int(width))


def build_model_from_config(cfg: ResolvedConfig, num_classes: int) -> nn.Module:
    """Build ``cfg.model_name`` with its settings, input shape and seed from ``cfg``.

    The one way notebooks and library code build a model. Using it everywhere
    guarantees the model trained in 03/04/05 and the model profiled in 06 are
    the same architecture built from the same settings.

    Args:
        num_classes: from ``DataBundle.num_classes`` when data is loaded, or
            ``split_meta.json -> num_classes`` when it is not (notebook 06).
    """
    return build_model(
        cfg.model_name,
        num_classes=num_classes,
        input_shape=model_input_shape(cfg),
        seed=cfg.seed,
        **model_settings(cfg),
    )


def available_models() -> list[str]:
    """Every registered key, sorted."""
    _import_builders()
    return sorted(_REGISTRY)


def _import_builders() -> None:
    """Import the model subpackages so their decorators run.

    Lazy on purpose: importing the registry must not pull torchvision (and a
    possible weights download) into a process that only wanted to read a
    config or validate a schema.
    """
    import edgecnn.models.custom  # noqa: F401
    import edgecnn.models.pretrained  # noqa: F401
