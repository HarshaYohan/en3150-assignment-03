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

from typing import TYPE_CHECKING

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
        raise NotImplementedError("Member 3: implement Trainer.__init__")

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
        raise NotImplementedError("Member 3: implement Trainer.fit")

    def train_epoch(self, model: nn.Module, loader: object, epoch: int) -> tuple[float, float]:
        """One training pass. Returns ``(mean_loss, accuracy)``.

        ``model.train()`` first - forgetting it leaves BatchNorm in eval mode
        and the model appears not to learn for reasons that take hours to find.
        """
        raise NotImplementedError("Member 3: implement train_epoch")

    def validate(self, model: nn.Module, loader: object) -> tuple[float, float]:
        """One validation pass. Returns ``(mean_loss, accuracy)``.

        ``model.eval()`` and ``torch.no_grad()``. This is also the method used
        for the final test pass by Member 1's evaluator, which is why it takes
        a loader rather than reaching into the bundle itself.
        """
        raise NotImplementedError("Member 3: implement validate")

    def save_checkpoint(self, path: object, model: nn.Module, epoch: int, val_acc: float) -> None:
        """Write a checkpoint carrying every key in ``CHECKPOINT_KEYS``.

        ``class_names`` and ``input_shape`` are included so Member 1 can
        evaluate a checkpoint without re-reading the config that produced it -
        a checkpoint should be self-describing.
        ``tests/contracts/test_checkpoint_contract.py`` asserts the key set.
        """
        raise NotImplementedError("Member 3: implement save_checkpoint")
