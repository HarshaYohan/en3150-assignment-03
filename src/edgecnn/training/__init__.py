"""SEAM 3 - training loop, optimizers, and the Section 3 summary.

Owner: Member 3. Driven from notebooks 03 and 04 (and 05, Member 4).

    from edgecnn.training import Trainer
    result = Trainer(cfg).fit(model, data, cfg)   # returns in every MODE
"""

from edgecnn.training.callbacks import CheckpointManager, EarlyStopping
from edgecnn.training.optimizer_study import summarize_study
from edgecnn.training.optimizers import (
    SUPPORTED_OPTIMIZERS,
    SUPPORTED_SCHEDULERS,
    build_optimizer,
    build_scheduler,
    describe_optimizer,
)
from edgecnn.training.trainer import Trainer

__all__ = [
    "CheckpointManager",
    "EarlyStopping",
    "SUPPORTED_OPTIMIZERS",
    "SUPPORTED_SCHEDULERS",
    "Trainer",
    "build_optimizer",
    "build_scheduler",
    "describe_optimizer",
    "summarize_study",
]
