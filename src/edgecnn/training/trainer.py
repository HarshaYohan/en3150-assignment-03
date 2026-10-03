"""SEAM 3 - the model-agnostic training loop.  Assignment Section 4 [25]

Owner: Member 3.   Called from notebooks 03, 04 and 05 - one loop, four models.

    +---------------------------------------------------------------------+
    |  IN   model : nn.Module      from build_model_from_config() (Seam 2) |
    |       data  : DataBundle     from build_dataloaders()       (Seam 1) |
    |       cfg   : ResolvedConfig loaded with the notebook's MODE         |
    |                                                                      |
    |  OUT  TrainResult - ALWAYS returned, in every mode, so the notebook  |
    |       can plot it inline.                                            |
    |                                                                      |
    |  official mode (cfg.is_official) also writes:                        |
    |       artifacts/checkpoints/<run_id>/best.pt   git-ignored           |
    |       artifacts/checkpoints/<run_id>/last.pt   git-ignored           |
    |       results/metrics/<run_id>/history.json    COMMITTED             |
    |  synthetic / debug mode writes only:                                 |
    |       artifacts/checkpoints/_debug/<run_id>/*.pt                     |
    +---------------------------------------------------------------------+

One loop, four models. This file must contain no branch on model name. If it
ever needs one, the registry contract is wrong and that is the thing to fix -
a per-model branch here is how the Section 6 comparison quietly becomes
apples-to-oranges.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn

    from edgecnn.contracts.types import DataBundle, ResolvedConfig, TrainResult


class Trainer:
    """Trains any registered model and records everything the report needs.

    Non-negotiable behaviours, each asserted by a contract test:

    * **Never touches ``data.test``.** The held-out split stays sealed until
      Member 1 evaluates. Peeking - even to print a number mid-training -
      invalidates every accuracy in the report.
    * **Times every epoch identically.** ``epoch_time_s`` covers the training
      pass plus the validation pass. It is a column in the Section 4 table, so
      Model A and MobileNetV2 must be measured at the same two points.
      Call ``torch.cuda.synchronize()`` before stopping the clock on GPU, or
      the timings are meaningless: CUDA kernels are asynchronous and the
      Python-side timer will otherwise record queueing time, not compute.
    * **Records at least 20 epochs.** The assignment mandates it and
      ``history.schema.json`` enforces ``minItems: 20``.
    * **Writes into ``results/`` only when ``cfg.is_official``.** In synthetic
      and debug mode it still returns a full ``TrainResult`` (so the notebook
      plots it inline) but checkpoints go to ``paths.checkpoint_dir(run_id,
      official=False)`` and no ``history.json`` is written. A debug run on 5%
      of the data must never reach a committed comparison table.
    * **Progress via ``tqdm.auto``** - one updating line per epoch, which keeps
      committed notebook outputs small. Never print per batch.
    * **Resumable.** With ``training.resume: true`` it continues from
      ``last.pt`` - useful when a long run is interrupted.
    """

    def __init__(self, cfg: ResolvedConfig) -> None:
        import torch

        from edgecnn.utils.device import resolve_device

        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        training = cfg.section("training")
        loss_name = str(training.get("loss", "cross_entropy"))
        if loss_name != "cross_entropy":
            raise ValueError("training.loss must be 'cross_entropy'")
        self.criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=float(training.get("label_smoothing", 0.0))
        )
        self.grad_clip_norm = training.get("grad_clip_norm")
        self.optimizer: torch.optim.Optimizer | None = None
        self.optimizer_description: dict[str, Any] = {}
        self.scheduler: Any = None
        self.result: TrainResult | None = None
        self.class_names: list[str] = []
        self.input_shape: tuple[int, int, int] = (3, 64, 64)

    def fit(self, model: nn.Module, data: DataBundle, cfg: ResolvedConfig) -> TrainResult:
        """Run the full training schedule.

        Outline:

        1. ``seed_everything(cfg.seed)`` first. Initial weights were already
           seeded by ``build_model_from_config``; this reseed fixes dropout,
           augmentation and batch order, so a notebook gives the same result
           whichever cells ran before it. Together they make the three
           Section 3 runs differ only in the optimizer.
        2. Move the model to ``resolve_device(cfg.device)``.
        3. Build the optimizer via ``edgecnn.training.optimizers.build_optimizer``
           (it handles the pretrained ``param_groups()`` case).
        4. Per epoch: train pass, val pass, scheduler step, record an
           ``EpochRecord``, checkpoint when ``val_acc`` improves - into
           ``paths.checkpoint_dir(run_id, official=cfg.is_official)``.
        5. Only if ``cfg.is_official``: write ``history.json`` through
           ``schema.write_json``, so a malformed history can never reach disk.

        Returns:
            :class:`edgecnn.contracts.types.TrainResult`, in every mode, with
            ``checkpoint_path`` pointing at the best checkpoint actually
            written - the notebook passes it straight to ``evaluate_run``.
        """
        import torch
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        from tqdm.auto import tqdm

        from edgecnn.contracts import paths, schema
        from edgecnn.contracts.types import EpochRecord, TrainResult
        from edgecnn.models.registry import model_settings
        from edgecnn.training.callbacks import CheckpointManager, EarlyStopping
        from edgecnn.training.optimizers import (
            build_optimizer,
            build_scheduler,
            describe_optimizer,
        )
        from edgecnn.utils.seed import seed_everything

        if cfg.run_id != self.cfg.run_id:
            raise ValueError("Trainer was constructed with a different run configuration")

        deterministic = bool(cfg.raw.get("deterministic", True))
        seed_everything(cfg.seed, deterministic=deterministic)
        model = model.to(self.device)
        self.class_names = list(data.class_names)
        self.input_shape = tuple(data.input_shape)

        optimizer_cfg = dict(cfg.section("optimizer"))
        settings = model_settings(cfg)
        if "backbone_lr_scale" in settings:
            optimizer_cfg["backbone_lr_scale"] = settings["backbone_lr_scale"]
        self.optimizer = build_optimizer(model, optimizer_cfg)
        self.optimizer_description = describe_optimizer(self.optimizer)

        training_cfg = cfg.section("training")
        epochs = int(training_cfg.get("epochs", 0))
        if epochs < 1:
            raise ValueError("training.epochs must be at least 1")
        self.scheduler = build_scheduler(self.optimizer, cfg.section("scheduler"), epochs)

        early_cfg = cfg.section("early_stopping")
        early_stopping = None
        if bool(early_cfg.get("enabled", False)):
            early_stopping = EarlyStopping(
                monitor=str(early_cfg.get("monitor", "val_acc")),
                mode=str(early_cfg.get("mode", "max")),
                patience=int(early_cfg.get("patience", 8)),
                min_delta=float(early_cfg.get("min_delta", 0.0)),
                min_epochs=int(early_cfg.get("min_epochs", 20)),
            )

        checkpoint_cfg = cfg.section("checkpointing")
        manager = CheckpointManager(
            cfg.run_id,
            monitor=str(checkpoint_cfg.get("monitor", "val_acc")),
            mode=str(checkpoint_cfg.get("mode", "max")),
            save_last=bool(checkpoint_cfg.get("save_last", True)),
            official=cfg.is_official,
        )
        manager.save_best = bool(checkpoint_cfg.get("save_best", True))
        manager.save_fn = self.save_checkpoint

        self.result = TrainResult(
            run_id=cfg.run_id,
            model_name=cfg.model_name,
            optimizer_name=cfg.optimizer_name,
            seed=cfg.seed,
        )
        start_epoch = 1
        if bool(training_cfg.get("resume", False)) and manager.last_path.exists():
            checkpoint = torch.load(manager.last_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            restored = checkpoint.get("epochs", [])
            self.result.epochs = [EpochRecord(**row) for row in restored]
            self.result.total_train_time_s = float(checkpoint.get("total_train_time_s", 0.0))
            start_epoch = int(checkpoint["epoch"]) + 1
            if self.result.epochs:
                best = max(self.result.epochs, key=lambda row: row.val_acc)
                self.result.best_epoch = best.epoch
                self.result.best_val_acc = best.val_acc
                manager.best = best.val_acc
                if early_stopping is not None:
                    early_stopping.best = best.val_acc

        progress_enabled = bool(cfg.section("logging").get("progress_bar", True))
        epoch_iterator = tqdm(
            range(start_epoch, epochs + 1),
            desc=cfg.run_id,
            unit="epoch",
            disable=not progress_enabled,
        )
        for epoch in epoch_iterator:
            started = time.perf_counter()
            train_loss, train_acc = self.train_epoch(model, data.train, epoch)
            val_loss, val_acc = self.validate(model, data.val)

            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            epoch_time = time.perf_counter() - started
            current_lr = max(float(group["lr"]) for group in self.optimizer.param_groups)

            record = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                lr=current_lr,
                epoch_time_s=epoch_time,
            )
            self.result.epochs.append(record)
            self.result.total_train_time_s += epoch_time

            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
            saved = manager.maybe_save(model, epoch, metrics)
            if saved is not None:
                self.result.best_epoch = epoch
                self.result.best_val_acc = val_acc
                self.result.checkpoint_path = saved

            epoch_iterator.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_acc=f"{val_acc:.3f}",
            )
            if early_stopping is not None and early_stopping.step(metrics, epoch):
                break

        if self.result.checkpoint_path is None and manager.best_path.exists():
            self.result.checkpoint_path = manager.best_path

        if cfg.is_official:
            payload = self._history_payload(self.optimizer_description)
            schema.write_json(paths.history_json(cfg.run_id), payload, "history")
        return self.result

    def train_epoch(self, model: nn.Module, loader: object, epoch: int) -> tuple[float, float]:
        """One training pass. Returns ``(mean_loss, accuracy)``.

        ``model.train()`` first - forgetting it leaves BatchNorm in eval mode
        and the model appears not to learn for reasons that take hours to find.
        """
        import torch

        if self.optimizer is None:
            raise RuntimeError("train_epoch cannot run before the optimizer is created")
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in loader:  # type: ignore[union-attr]
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            self._validate_logits(logits, labels)
            loss = self.criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.grad_clip_norm))
            self.optimizer.step()

            batch_size = int(labels.shape[0])
            total_loss += float(loss.detach().item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("training loader produced no samples")
        return total_loss / total_samples, total_correct / total_samples

    def validate(self, model: nn.Module, loader: object) -> tuple[float, float]:
        """One validation pass. Returns ``(mean_loss, accuracy)``.

        ``model.eval()`` and ``torch.no_grad()``. This is also the method used
        for the final test pass by Member 1's evaluator, which is why it takes
        a loader rather than reaching into the bundle itself.
        """
        import torch

        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in loader:  # type: ignore[union-attr]
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits = model(images)
                self._validate_logits(logits, labels)
                loss = self.criterion(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite validation loss")

                batch_size = int(labels.shape[0])
                total_loss += float(loss.item()) * batch_size
                total_correct += int((logits.argmax(dim=1) == labels).sum().item())
                total_samples += batch_size

        if total_samples == 0:
            raise ValueError("validation loader produced no samples")
        return total_loss / total_samples, total_correct / total_samples

    def save_checkpoint(self, path: object, model: nn.Module, epoch: int, val_acc: float) -> None:
        """Write a checkpoint carrying every key in ``CHECKPOINT_KEYS``.

        ``class_names`` and ``input_shape`` are included so Member 1 can
        evaluate a checkpoint without re-reading the config that produced it -
        a checkpoint should be self-describing.
        ``tests/contracts/test_checkpoint_contract.py`` asserts the key set.
        """
        import torch

        if self.result is None:
            raise RuntimeError("checkpointing requires an active training result")
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.cfg.model_name,
            "state_dict": model.state_dict(),
            "epoch": int(epoch),
            "val_acc": float(val_acc),
            "class_names": list(self.class_names),
            "input_shape": list(self.input_shape),
            "seed": self.cfg.seed,
            "config_snapshot": copy.deepcopy(self.cfg.raw),
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer is not None else None
            ),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "epochs": [asdict(record) for record in self.result.epochs],
            "total_train_time_s": self.result.total_train_time_s,
        }
        torch.save(payload, checkpoint_path)

    def _history_payload(self, optimizer_description: dict[str, Any]) -> dict[str, Any]:
        """Create the schema-validated history representation."""
        from edgecnn.utils.device import describe_device

        if self.result is None:
            raise RuntimeError("no training result is available")
        try:
            device = describe_device(self.device)
        except NotImplementedError:
            device = str(self.device)
        hyperparameters = dict(optimizer_description)
        hyperparameters.update(
            {
                "batch_size": self.cfg.section("dataloader").get("batch_size"),
                "scheduler": self.cfg.section("scheduler").get("name", "none"),
                "augmentation": self.cfg.section("augmentation"),
            }
        )
        return {
            "run_id": self.result.run_id,
            "model": self.result.model_name,
            "optimizer": self.result.optimizer_name,
            "seed": self.result.seed,
            "device": device,
            "epochs": [asdict(record) for record in self.result.epochs],
            "best_epoch": self.result.best_epoch,
            "best_val_acc": self.result.best_val_acc,
            "total_train_time_s": self.result.total_train_time_s,
            "hyperparameters": hyperparameters,
        }

    @staticmethod
    def _validate_logits(logits: Any, labels: Any) -> None:
        if logits.ndim != 2:
            raise ValueError(f"model must return 2-D logits, got shape {tuple(logits.shape)}")
        if logits.shape[0] != labels.shape[0]:
            raise ValueError("logit batch size does not match label batch size")
