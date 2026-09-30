"""Layered YAML configuration.

Owner: Member 1. Consumed by every notebook and every member.

The layering
------------
A run's configuration is assembled in four layers, each overriding the last::

    configs/base.yaml              shared by everything: seed, device, paths, modes
        <- configs/stages/*.yaml   one per pipeline stage (via the experiment's `extends`)
            <- configs/experiments/<name>.yaml   what makes this run distinct
                <- modes[<mode>]   the notebook's MODE preset, from base.yaml

An experiment names the stages it pulls in via a top-level ``extends`` list,
then states only what makes it distinct - normally just the model and the
optimizer. A change to, say, the augmentation policy happens in one place.

One home per setting
--------------------
Settings live in top-level sections (``dataset``, ``training``, ``optimizer``,
...), and each section is defined in exactly one stage file. An experiment
overrides a setting at the same path. Stage I/O declarations are kept apart
under ``stages.<name>`` so they never collide when several stages merge.
``tests/contracts/test_config_layout.py`` enforces both rules.

Modes
-----
Notebooks call ``load_config(path, mode=MODE)``. The preset for that mode is
read from ``base.yaml -> modes`` and applied as dotted overrides, e.g. debug
mode sets ``subset_fraction: 0.05`` and ``training.epochs: 2``. Only
``mode="official"`` can produce ``cfg.is_official == True``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from edgecnn.contracts.paths import CONFIGS_DIR, REPO_ROOT, make_run_id
from edgecnn.contracts.types import DEFAULT_SEED, RUN_MODES, ResolvedConfig


class ConfigError(Exception):
    """Raised for a malformed, missing or contradictory configuration."""


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(loaded).__name__}")
    return loaded


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``, returning a new dict.

    Nested mappings merge key by key; every other type (including lists) is
    replaced wholesale. Lists are replaced rather than concatenated on
    purpose - an experiment that sets ``augmentation: [random_flip]`` means
    *only* that, not "the base list plus this".
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_path(value: str | Path) -> Path:
    """Turn a repo-relative config path into an absolute one.

    Config files always spell paths relative to the repository root, so the
    same YAML works from ``notebooks/``, ``tests/`` or a Colab clone without
    anyone thinking about the current working directory.
    """
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_stage(name: str) -> dict[str, Any]:
    """Load one ``configs/stages/<name>.yaml``."""
    return _read_yaml(CONFIGS_DIR / "stages" / f"{name}.yaml")


def load_config(path: str | Path, mode: str = "official", **overrides: Any) -> ResolvedConfig:
    """Compose base + stages + experiment + mode preset into a :class:`ResolvedConfig`.

    Args:
        path: An experiment config, normally under ``configs/experiments/``.
        mode: ``"synthetic"``, ``"debug"`` or ``"official"`` - the notebook's
            ``MODE``. Applies the matching preset from ``base.yaml -> modes``.
        **overrides: Last-word overrides, applied after the mode preset. Use a
            dotted path for nesting: ``load_config(p, **{"training.epochs": 3})``.
            An override can make a run unofficial but never official - see
            ``ResolvedConfig.is_official``.

    Raises:
        ConfigError: on an unknown mode, a missing required key, or a stated
            ``run_id`` that disagrees with model/optimizer/seed - a mismatch
            there would scatter one run's artifacts across two directories.
    """
    experiment_path = resolve_path(path)
    experiment = _read_yaml(experiment_path)

    config: dict[str, Any] = _read_yaml(CONFIGS_DIR / "base.yaml")

    for stage_name in experiment.get("extends", []):
        config = merge(config, load_stage(stage_name))

    config = merge(config, experiment)
    config.pop("extends", None)

    presets = config.pop("modes", {}) or {}
    if mode not in RUN_MODES or mode not in presets:
        raise ConfigError(
            f"unknown mode {mode!r}; expected one of {list(RUN_MODES)} "
            f"(presets live in configs/base.yaml -> modes)"
        )
    for dotted, value in (presets[mode] or {}).items():
        _set_dotted(config, dotted, value)
    config["mode"] = mode

    for dotted, value in overrides.items():
        _set_dotted(config, dotted, value)

    model_name = _require(config, "model", "name", context=experiment_path)
    optimizer_name = _require(config, "optimizer", "name", context=experiment_path)
    seed = int(config.get("seed", DEFAULT_SEED))

    run_id = make_run_id(model_name, optimizer_name, seed)
    declared = config.get("run_id")
    if declared is not None and declared != run_id:
        raise ConfigError(
            f"{experiment_path}: run_id is declared as {declared!r} but "
            f"model/optimizer/seed imply {run_id!r}. Artifacts would be written "
            f"to one directory and read from another. Fix one or the other."
        )
    config["run_id"] = run_id

    return ResolvedConfig(
        raw=config,
        run_id=run_id,
        model_name=model_name,
        optimizer_name=optimizer_name,
        seed=seed,
        device=str(config.get("device", "auto")),
        mode=mode,
    )


def _require(config: dict[str, Any], section: str, key: str, *, context: Path) -> str:
    block = config.get(section)
    if not isinstance(block, dict) or key not in block:
        raise ConfigError(f"{context}: missing required key '{section}.{key}'")
    return str(block[key])


def _set_dotted(config: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cursor = config
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[parts[-1]] = value
