"""Model definitions and the registry that hides them from the trainer.

    custom/      Member 2 - Model A (standard CNN), Model B (<=100k params)
    pretrained/  Member 4 - MobileNetV2, SqueezeNet 1.1
    registry.py  shared   - the Seam 2 entry point

Notebooks build every model the same way::

    from edgecnn.models import build_model_from_config
    model = build_model_from_config(cfg, num_classes=data.num_classes)
"""

from edgecnn.models.registry import (
    REQUIRED_KEYS,
    ModelNotFound,
    available_models,
    build_model,
    build_model_from_config,
    model_input_shape,
    model_settings,
    register,
)

__all__ = [
    "REQUIRED_KEYS",
    "ModelNotFound",
    "available_models",
    "build_model",
    "build_model_from_config",
    "model_input_shape",
    "model_settings",
    "register",
]
