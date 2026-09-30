"""Config layout: one home per setting, namespaced stage I/O, run modes.

These guard against a specific class of bug: one merged config holding two
copies of a setting that disagree. The scaffold originally had exactly that -
an SGD experiment whose merged config still carried a default Adam optimizer
under a stage's `inputs:` block. A trainer reading the wrong copy would have
run the whole Section 3 study with Adam, and nothing would have crashed.
"""

from __future__ import annotations

import pytest
import yaml

from edgecnn.config.loader import ConfigError, load_config
from edgecnn.contracts import paths
from edgecnn.contracts.types import DEFAULT_SEED, MIN_EPOCHS, RUN_MODES

pytestmark = pytest.mark.contract

REQUIRED_STAGE_KEYS = ("owner", "produces_for", "settings", "inputs", "outputs", "contract")


def _load(path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# --- stage files -------------------------------------------------------------


def test_every_stage_declares_its_io_under_its_own_namespace(stage_configs: list) -> None:
    """The seam must be readable from the config alone, and must not collide on merge."""
    assert stage_configs, "no stage configs found"
    for path in stage_configs:
        stages = _load(path).get("stages")
        assert isinstance(stages, dict) and list(stages) == [path.stem], (
            f"{path.name}: must declare exactly one block, `stages.{path.stem}`"
        )
        missing = [key for key in REQUIRED_STAGE_KEYS if key not in stages[path.stem]]
        assert not missing, f"{path.name}: stages.{path.stem} is missing {missing}"


def test_declared_settings_match_the_file(stage_configs: list) -> None:
    """`stages.<name>.settings` is the index a reader uses to find where a setting lives."""
    for path in stage_configs:
        data = _load(path)
        defined = sorted(key for key in data if key != "stages")
        declared = sorted(data["stages"][path.stem]["settings"])
        assert defined == declared, f"{path.name}: declares {declared} but defines {defined}"


def test_no_setting_is_defined_in_two_files(stage_configs: list) -> None:
    """One home per setting - the rule whose absence caused the SGD/Adam bug."""
    homes: dict[str, list[str]] = {key: ["base.yaml"] for key in _load(paths.CONFIGS_DIR / "base.yaml")}
    for path in stage_configs:
        for key in _load(path):
            if key != "stages":
                homes.setdefault(key, []).append(path.name)
    clashes = {key: files for key, files in homes.items() if len(files) > 1}
    assert not clashes, f"settings defined in more than one file: {clashes}"


def test_no_stage_file_defines_an_optimizer(stage_configs: list) -> None:
    """Configs merge recursively, so a default optimizer would leak keys between optimizers."""
    for path in stage_configs:
        assert "optimizer" not in _load(path), (
            f"{path.name}: the optimizer belongs in the experiment file that names it"
        )


# --- experiments -------------------------------------------------------------


def test_experiment_optimizer_matches_its_run_id(experiment_configs: list) -> None:
    """Exactly one optimizer section per run, and it is the one the run is named after."""
    for path in experiment_configs:
        cfg = load_config(path)
        assert cfg.section("optimizer").get("name") == cfg.optimizer_name, path.name
        assert "inputs" not in cfg.raw, (
            f"{path.name}: settings must not live under a top-level `inputs` block"
        )


def test_optimizer_study_matches_the_reporting_table() -> None:
    """Notebook 03 trains these; notebook 07 tabulates these. They must be the same runs."""
    study = _load(paths.CONFIGS_DIR / "stages" / "training.yaml")["optimizer_study"]["experiments"]
    tables = _load(paths.CONFIGS_DIR / "stages" / "evaluation.yaml")["reporting"]["tables"]
    table = next(t for t in tables if t["name"] == "optimizer_comparison")
    assert [f"{name}__seed{DEFAULT_SEED}" for name in study] == table["runs"]


def test_every_reported_run_has_an_experiment_config() -> None:
    """A table row with no config behind it could never be reproduced."""
    tables = _load(paths.CONFIGS_DIR / "stages" / "evaluation.yaml")["reporting"]["tables"]
    for table in tables:
        for run_id in table["runs"]:
            assert paths.experiment_config(run_id).exists(), f"{table['name']}: no config for {run_id}"


# --- run modes ---------------------------------------------------------------


def test_every_run_mode_has_a_preset() -> None:
    presets = _load(paths.CONFIGS_DIR / "base.yaml")["modes"]
    assert sorted(presets) == sorted(RUN_MODES)


def test_every_experiment_is_official_as_written(experiment_configs: list) -> None:
    for path in experiment_configs:
        cfg = load_config(path)
        assert cfg.mode == "official"
        assert cfg.is_official, f"{path.name} should be an official run as written"


@pytest.mark.parametrize("mode", ["synthetic", "debug"])
def test_non_official_modes_are_never_official(experiment_configs: list, mode: str) -> None:
    """The guarantee that debug work can never land in results/."""
    for path in experiment_configs:
        assert not load_config(path, mode=mode).is_official, f"{path.name} in {mode} mode"


def test_debug_mode_shrinks_the_run(experiment_configs: list) -> None:
    cfg = load_config(experiment_configs[0], mode="debug")
    assert cfg.raw["subset_fraction"] < 1.0
    assert cfg.section("training")["epochs"] < MIN_EPOCHS


def test_synthetic_mode_switches_the_dataset(experiment_configs: list) -> None:
    cfg = load_config(experiment_configs[0], mode="synthetic")
    assert cfg.section("dataset")["name"] == "synthetic"


def test_an_override_cannot_make_a_short_run_official(experiment_configs: list) -> None:
    cfg = load_config(experiment_configs[0], mode="official", **{"training.epochs": 5})
    assert not cfg.is_official


def test_an_override_cannot_make_a_debug_run_official(experiment_configs: list) -> None:
    """Even with every value restored, debug intent stays unofficial."""
    cfg = load_config(
        experiment_configs[0],
        mode="debug",
        **{"subset_fraction": 1.0, "training.epochs": 30},
    )
    assert not cfg.is_official


def test_unknown_mode_is_rejected(experiment_configs: list) -> None:
    with pytest.raises(ConfigError, match="unknown mode"):
        load_config(experiment_configs[0], mode="fast")
