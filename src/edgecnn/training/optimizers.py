"""Optimizer and scheduler factories.       Assignment Section 3 [15 marks]

Owner: Member 3.

    +---------------------------------------------------------------------+
    |  IN   model, cfg.raw["optimizer"] from the experiment config         |
    |  OUT  torch.optim.Optimizer  (+ optional LR scheduler)               |
    +---------------------------------------------------------------------+

The registry key names must match ``history.schema.json``'s optimizer enum
exactly - ``sgd``, ``sgd_momentum``, ``adam``, ``adamw``, ``rmsprop`` - because
the optimizer name is also a component of ``run_id`` and therefore of every
output path.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from torch import nn

#: Must stay in sync with the `optimizer` enum in history.schema.json.
SUPPORTED_OPTIMIZERS: tuple[str, ...] = ("sgd", "sgd_momentum", "adam", "adamw", "rmsprop")

SUPPORTED_SCHEDULERS: tuple[str, ...] = ("none", "step", "cosine", "reduce_on_plateau")


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    """Construct the optimizer named in the config.

    ``sgd`` and ``sgd_momentum`` are both ``torch.optim.SGD``; they are
    separate registry keys because they are separate *experiments*, and the
    run_id has to distinguish their output directories.

    ``cfg`` here is the experiment's ``optimizer`` section - the only place an
    optimizer is defined. For a pretrained backbone, split the parameters with
    ``edgecnn.models.pretrained.param_groups(model, base_lr=optimizer.lr,
    backbone_lr_scale=...)``, taking the scale from
    ``model_settings(cfg)["backbone_lr_scale"]``: the fresh head trains at the
    experiment's LR and the pretrained weights at a fraction of it. Custom
    models pass ``model.parameters()`` as one group - no special handling.

    Raises:
        ValueError: on an unsupported name, listing the valid options.
    """
    import torch

    settings = dict(cfg)
    name = str(settings.pop("name", "")).lower()
    if name not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"unsupported optimizer {name!r}; expected one of {list(SUPPORTED_OPTIMIZERS)}"
        )

    try:
        lr = float(settings.pop("lr"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("optimizer.lr must be a positive number") from exc
    if lr <= 0:
        raise ValueError("optimizer.lr must be positive")

    backbone_lr_scale = settings.pop("backbone_lr_scale", None)
    params: Iterable[torch.nn.Parameter] | list[dict[str, object]]
    if backbone_lr_scale is None:
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    else:
        from edgecnn.models.pretrained import param_groups

        params = param_groups(model, base_lr=lr, backbone_lr_scale=float(backbone_lr_scale))

    if not params:
        raise ValueError("model has no trainable parameters")

    weight_decay = float(settings.pop("weight_decay", 0.0))
    if weight_decay < 0:
        raise ValueError("optimizer.weight_decay cannot be negative")

    if name in {"sgd", "sgd_momentum"}:
        default_momentum = 0.0 if name == "sgd" else 0.9
        momentum = float(settings.pop("momentum", default_momentum))
        if name == "sgd" and momentum != 0.0:
            raise ValueError("optimizer 'sgd' must use momentum: 0.0; use 'sgd_momentum' instead")
        if name == "sgd_momentum" and momentum <= 0.0:
            raise ValueError("optimizer 'sgd_momentum' requires momentum greater than zero")
        optimizer = torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=float(settings.pop("dampening", 0.0)),
            nesterov=bool(settings.pop("nesterov", False)),
        )
    elif name in {"adam", "adamw"}:
        betas = tuple(float(value) for value in settings.pop("betas", (0.9, 0.999)))
        if len(betas) != 2:
            raise ValueError("optimizer.betas must contain exactly two values")
        optimizer_type = torch.optim.Adam if name == "adam" else torch.optim.AdamW
        optimizer = optimizer_type(
            params,
            lr=lr,
            betas=betas,
            eps=float(settings.pop("eps", 1e-8)),
            weight_decay=weight_decay,
            amsgrad=bool(settings.pop("amsgrad", False)),
        )
    else:
        optimizer = torch.optim.RMSprop(
            params,
            lr=lr,
            alpha=float(settings.pop("alpha", 0.99)),
            eps=float(settings.pop("eps", 1e-8)),
            weight_decay=weight_decay,
            momentum=float(settings.pop("momentum", 0.0)),
            centered=bool(settings.pop("centered", False)),
        )

    if settings:
        unknown = ", ".join(sorted(settings))
        raise ValueError(f"unsupported settings for optimizer {name!r}: {unknown}")
    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Construct the LR scheduler, or ``None`` for ``name: none``.

    One caveat that matters for Section 3: a scheduler changes the convergence
    curve, so the three optimizer variants must use the *same* scheduler
    setting or the comparison confounds two variables. If cosine annealing is
    used for Adam it must be used for both SGD runs too.

    ``reduce_on_plateau`` steps on a metric rather than on epoch count, so the
    trainer has to call it differently - handle that in the trainer, not here.
    """
    import torch

    settings = dict(cfg)
    name = str(settings.pop("name", "none")).lower()
    if name not in SUPPORTED_SCHEDULERS:
        raise ValueError(
            f"unsupported scheduler {name!r}; expected one of {list(SUPPORTED_SCHEDULERS)}"
        )
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if name == "none":
        return None

    warmup_epochs = int(settings.pop("warmup_epochs", 0))
    if warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ValueError("scheduler.warmup_epochs must be in the range [0, epochs)")

    if name == "step":
        settings.pop("min_lr", None)  # StepLR has no floor; accepted for shared configs.
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(settings.pop("step_size", max(1, epochs // 3))),
            gamma=float(settings.pop("gamma", 0.1)),
        )
    elif name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - warmup_epochs),
            eta_min=float(settings.pop("min_lr", 0.0)),
        )
    else:
        if warmup_epochs:
            raise ValueError("warmup is not supported with reduce_on_plateau")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(settings.pop("mode", "min")),
            factor=float(settings.pop("factor", 0.1)),
            patience=int(settings.pop("patience", 3)),
            threshold=float(settings.pop("threshold", 1e-4)),
            min_lr=float(settings.pop("min_lr", 0.0)),
        )

    if settings:
        unknown = ", ".join(sorted(settings))
        raise ValueError(f"unsupported settings for scheduler {name!r}: {unknown}")

    if warmup_epochs:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / max(2, warmup_epochs),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, scheduler],
            milestones=[warmup_epochs],
        )
    return scheduler


def describe_optimizer(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Flat hyperparameter snapshot for ``history.json -> hyperparameters``.

    Sections 3 and 4 both require the learning rate and optimizer settings to
    be stated, and reading them back off the optimizer object is more reliable
    than re-reading the config - it records what was actually used, including
    any CLI override.
    """
    import torch

    defaults = optimizer.defaults
    if isinstance(optimizer, torch.optim.SGD):
        name = "sgd_momentum" if float(defaults.get("momentum", 0.0)) > 0 else "sgd"
        keys = ("lr", "momentum", "dampening", "weight_decay", "nesterov")
    elif isinstance(optimizer, torch.optim.AdamW):
        name = "adamw"
        keys = ("lr", "betas", "eps", "weight_decay", "amsgrad")
    elif isinstance(optimizer, torch.optim.Adam):
        name = "adam"
        keys = ("lr", "betas", "eps", "weight_decay", "amsgrad")
    elif isinstance(optimizer, torch.optim.RMSprop):
        name = "rmsprop"
        keys = ("lr", "alpha", "eps", "weight_decay", "momentum", "centered")
    else:
        name = optimizer.__class__.__name__.lower()
        keys = tuple(defaults)

    description: dict[str, Any] = {"optimizer": name}
    for key in keys:
        if key in defaults:
            value = defaults[key]
            description[key] = list(value) if isinstance(value, tuple) else value

    group_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    description["lr"] = max(group_lrs)
    if len(group_lrs) > 1:
        description["parameter_group_lrs"] = group_lrs
    return description
