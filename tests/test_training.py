"""Focused tests for Member 3's training and optimization implementation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from edgecnn.contracts import paths, schema
from edgecnn.contracts.types import DataBundle, EpochRecord, ResolvedConfig, TrainResult
from edgecnn.evaluation.curves import plot_loss_curves, plot_optimizer_overlay
from edgecnn.training.optimizer_study import summarize_study
from edgecnn.training.optimizers import (
    build_optimizer,
    build_scheduler,
    describe_optimizer,
)
from edgecnn.training.trainer import Trainer

matplotlib.use("Agg")


class _ForbiddenTestLoader:
    def __iter__(self):
        raise AssertionError("Trainer must never iterate over the test split")


def _history(optimizer: str = "adam") -> TrainResult:
    records = [
        EpochRecord(1, 1.0, 1.1, 0.50, 0.45, 0.001, 0.1),
        EpochRecord(2, 0.8, 0.9, 0.70, 0.65, 0.0001, 0.2),
    ]
    return TrainResult(
        run_id=f"model_b__{optimizer}__seed42",
        model_name="model_b",
        optimizer_name=optimizer,
        seed=42,
        epochs=records,
        best_epoch=2,
        best_val_acc=0.65,
        total_train_time_s=0.3,
    )


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("sgd", torch.optim.SGD),
        ("sgd_momentum", torch.optim.SGD),
        ("adam", torch.optim.Adam),
        ("adamw", torch.optim.AdamW),
        ("rmsprop", torch.optim.RMSprop),
    ],
)
def test_optimizer_factory_supports_declared_names(name: str, expected_type: type) -> None:
    model = nn.Linear(3, 2)
    config = {"name": name, "lr": 0.01}
    if name == "sgd_momentum":
        config["momentum"] = 0.9

    optimizer = build_optimizer(model, config)

    assert isinstance(optimizer, expected_type)
    assert describe_optimizer(optimizer)["optimizer"] == name


@pytest.mark.parametrize("name", ["none", "step", "cosine", "reduce_on_plateau"])
def test_scheduler_factory_supports_declared_names(name: str) -> None:
    optimizer = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)
    scheduler = build_scheduler(optimizer, {"name": name}, epochs=3)
    assert (scheduler is None) is (name == "none")


def test_optimizer_study_summary_reports_convergence_and_gap() -> None:
    result = _history()

    summary = summarize_study({"adam": result})["adam"]

    assert summary["best_val_acc"] == pytest.approx(0.65)
    assert summary["epochs_to_90pct_best"] == 2
    assert summary["mean_epoch_time_s"] == pytest.approx(0.15)
    assert summary["final_train_val_gap"] == pytest.approx(0.05)


def test_curve_functions_accept_results_in_memory() -> None:
    first = _history("sgd")
    second = _history("adam")

    single = plot_loss_curves(first)
    overlay = plot_optimizer_overlay([first, second])

    assert len(single.axes) == 2
    assert len(overlay.axes) == 2


def test_trainer_runs_without_touching_test_split(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("edgecnn.utils.device.resolve_device", lambda spec: torch.device("cpu"))
    monkeypatch.setattr("edgecnn.utils.seed.seed_everything", lambda *args, **kwargs: None)
    monkeypatch.setattr(paths, "DEBUG_CHECKPOINTS_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(paths, "METRICS_DIR", tmp_path / "metrics")

    config = {
        "seed": 42,
        "device": "cpu",
        "deterministic": True,
        "mode": "synthetic",
        "dataset": {"name": "synthetic"},
        "training": {
            "epochs": 2,
            "loss": "cross_entropy",
            "label_smoothing": 0.0,
            "grad_clip_norm": None,
            "resume": False,
        },
        "optimizer": {"name": "adam", "lr": 0.01},
        "scheduler": {"name": "none"},
        "early_stopping": {"enabled": False},
        "checkpointing": {
            "monitor": "val_acc",
            "mode": "max",
            "save_best": True,
            "save_last": True,
        },
        "dataloader": {"batch_size": 4},
        "logging": {"progress_bar": False},
        "model_b": {},
    }
    cfg = ResolvedConfig(
        raw=config,
        run_id="model_b__adam__seed42",
        model_name="model_b",
        optimizer_name="adam",
        seed=42,
        device="cpu",
        mode="synthetic",
    )
    images = torch.randn(12, 3, 4, 4)
    labels = torch.randint(0, 2, (12,))
    train = DataLoader(TensorDataset(images, labels), batch_size=4)
    val = DataLoader(TensorDataset(images[:8], labels[:8]), batch_size=4)
    data = DataBundle(
        train=train,
        val=val,
        test=_ForbiddenTestLoader(),
        num_classes=2,
        class_names=["a", "b"],
        input_shape=(3, 4, 4),
        norm_mean=(0.0, 0.0, 0.0),
        norm_std=(1.0, 1.0, 1.0),
        split_manifest_path=tmp_path / "manifest.csv",
        seed=42,
    )
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))

    result = Trainer(cfg).fit(model, data, cfg)

    assert len(result.epochs) == 2
    assert result.checkpoint_path is not None and result.checkpoint_path.exists()
    assert not paths.history_json(cfg.run_id).exists()
    schema.validate_checkpoint(result.checkpoint_path)
