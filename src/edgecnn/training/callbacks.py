"""Early stopping and checkpoint selection.

Owner: Member 3.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from edgecnn.contracts import paths

if TYPE_CHECKING:
    from torch import nn


class EarlyStopping:
    """Stop when the monitored metric has not improved for `patience` epochs.

    One constraint specific to this assignment: `min_epochs` defaults to 20 and
    early stopping must not fire before it. The assignment mandates at least 20
    epochs, and `history.schema.json` enforces `minItems: 20` on the epochs
    array - so a run that stops at epoch 12 cannot be committed, and finding
    that out after the run is a wasted half hour.
    """

    def __init__(
        self,
        monitor: str = "val_acc",
        mode: str = "max",
        patience: int = 8,
        min_delta: float = 0.0,
        min_epochs: int = 20,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'")
        if patience < 1:
            raise ValueError("patience must be at least 1")
        if min_delta < 0:
            raise ValueError("min_delta cannot be negative")
        if min_epochs < 1:
            raise ValueError("min_epochs must be at least 1")

        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.best: float | None = None
        self.bad_epochs = 0

    def step(self, metrics: dict[str, float], epoch: int) -> bool:
        """Record this epoch's metrics; return True to stop training."""
        if self.monitor not in metrics:
            raise KeyError(f"Missing monitored metric: {self.monitor!r}")
        if epoch < 1:
            raise ValueError("epoch numbering must start at 1")

        current = float(metrics[self.monitor])
        if self.best is None:
            improved = True
        elif self.mode == "max":
            improved = current > self.best + self.min_delta
        else:
            improved = current < self.best - self.min_delta

        if improved:
            self.best = current
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        return epoch >= self.min_epochs and self.bad_epochs >= self.patience


class CheckpointManager:
    """Keeps `best.pt` and `last.pt` for one run.

    `best` is selected on validation accuracy, never on test accuracy - the
    test split is not visible to training code at all. `evaluate_run` loads
    `best.pt`, so which epoch this picks directly determines the number that
    lands in the report.

    Pass `official=cfg.is_official`. Unofficial (synthetic / debug) runs save
    under `paths.checkpoint_dir(run_id, official=False)`, i.e.
    `artifacts/checkpoints/_debug/<run_id>/`, so they can never overwrite the
    checkpoint an official result came from.
    """

    def __init__(
        self,
        run_id: str,
        monitor: str = "val_acc",
        mode: str = "max",
        save_last: bool = True,
        official: bool = True,
    ) -> None:
        if not run_id:
            raise ValueError("run_id cannot be empty")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'")

        self.run_id = run_id
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.save_best = True
        self.official = official
        self.best: float | None = None
        self.best_path = paths.best_checkpoint(run_id, official=official)
        self.last_path = paths.last_checkpoint(run_id, official=official)
        self.save_fn: Callable[[Path, nn.Module, int, float], None] | None = None
        paths.ensure_run_dirs(run_id, official=official)

    def maybe_save(self, model: object, epoch: int, metrics: dict[str, float]) -> Path | None:
        """Save when the monitored metric improves. Returns the path, or None."""
        if self.monitor not in metrics:
            raise KeyError(f"Missing monitored metric: {self.monitor!r}")
        if epoch < 1:
            raise ValueError("epoch numbering must start at 1")

        value = float(metrics[self.monitor])
        improved = (
            self.best is None
            or (self.mode == "max" and value > self.best)
            or (self.mode == "min" and value < self.best)
        )

        if self.save_last:
            self._save(self.last_path, model, epoch, value)
        if not improved or not self.save_best:
            return None

        self.best = value
        self._save(self.best_path, model, epoch, value)
        return self.best_path

    def _save(self, path: Path, model: object, epoch: int, value: float) -> None:
        """Use the trainer's full serializer, or a small standalone fallback."""
        if self.save_fn is not None:
            self.save_fn(path, model, epoch, value)  # type: ignore[arg-type]
            return

        import torch

        state_dict = getattr(model, "state_dict", None)
        if not callable(state_dict):
            raise TypeError("model must provide a callable state_dict() method")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": state_dict(),
                "epoch": epoch,
                self.monitor: value,
            },
            path,
        )
