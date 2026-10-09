"""Unit tests for Member 3's training callbacks."""

import pytest

from edgecnn.training.callbacks import EarlyStopping


def test_early_stopping_respects_minimum_epoch_floor() -> None:
    stopper = EarlyStopping(patience=2, min_epochs=4)

    assert not stopper.step({"val_acc": 0.80}, epoch=1)
    assert not stopper.step({"val_acc": 0.79}, epoch=2)
    assert not stopper.step({"val_acc": 0.78}, epoch=3)
    assert stopper.step({"val_acc": 0.77}, epoch=4)


def test_early_stopping_resets_patience_after_improvement() -> None:
    stopper = EarlyStopping(patience=2, min_epochs=1)

    assert not stopper.step({"val_acc": 0.70}, epoch=1)
    assert not stopper.step({"val_acc": 0.69}, epoch=2)
    assert not stopper.step({"val_acc": 0.75}, epoch=3)
    assert not stopper.step({"val_acc": 0.74}, epoch=4)
    assert stopper.step({"val_acc": 0.73}, epoch=5)


def test_early_stopping_supports_min_mode_and_min_delta() -> None:
    stopper = EarlyStopping(
        monitor="val_loss", mode="min", patience=1, min_delta=0.05, min_epochs=1
    )

    assert not stopper.step({"val_loss": 1.0}, epoch=1)
    assert stopper.step({"val_loss": 0.97}, epoch=2)


def test_early_stopping_rejects_missing_metric() -> None:
    stopper = EarlyStopping()

    with pytest.raises(KeyError, match="val_acc"):
        stopper.step({"val_loss": 1.0}, epoch=1)
