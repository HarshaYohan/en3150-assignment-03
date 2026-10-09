"""The Section 3 optimizer comparison.           Assignment Section 3 [15]

Owner: Member 3.

The study itself is *run* in notebooks/03_optimizer_study.ipynb, one visible
block per optimizer, so each run's curves can be inspected before the next
starts. The experiments it trains are listed in one place,
``configs/stages/training.yaml -> optimizer_study.experiments``, and their
hyperparameters live in those experiment files.

    +---------------------------------------------------------------------+
    |  IN   the three TrainResults from notebook 03                       |
    |  OUT  a summary for the Section 3 discussion; the committed table   |
    |       is built from the committed JSON by notebook 07               |
    +---------------------------------------------------------------------+

The comparison only isolates the optimizer if everything else is held fixed:
same seed, same initial weights, same data order, same scheduler, same epochs.
``build_model_from_config`` and ``Trainer.fit`` both seed from ``cfg.seed``,
so running the three blocks in any order gives the same three results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from edgecnn.contracts.types import TrainResult


def summarize_study(results: dict[str, TrainResult]) -> dict[str, Any]:
    """Reduce the three runs to the Section 3 comparison.

    Args:
        results: ``{optimizer_name: TrainResult}`` - from the notebook in any
            mode, so the summary can be inspected during debug runs too.

    Report convergence AND final performance - the assignment asks about both,
    and they can disagree. Suggested columns:

    * ``best_val_acc``          - final performance
    * ``epochs_to_90pct_best``  - convergence speed, defined as the first epoch
      reaching 90% of that run's own best validation accuracy. Self-relative,
      so a slow-but-strong optimizer is not penalised twice.
    * ``mean_epoch_time_s``     - wall-clock cost. Adam holds two extra state
      tensors per parameter, which is worth a sentence in a report about
      memory-constrained devices: on an MCU that can matter more than the
      convergence advantage.
    * ``final_train_val_gap``   - overfitting signal.

    The momentum discussion Section 3 asks for should separate the two effects:
    the velocity term damps oscillation across narrow ravines, AND accelerates
    travel along consistently descending directions. Show both in the curves
    rather than asserting them.
    """
    if not results:
        raise ValueError("optimizer study requires at least one result")

    summary: dict[str, Any] = {}
    for optimizer_name, result in results.items():
        if not result.epochs:
            raise ValueError(f"optimizer {optimizer_name!r} has no completed epochs")

        best_val_acc = max(record.val_acc for record in result.epochs)
        threshold = 0.9 * best_val_acc
        convergence_epoch = next(
            record.epoch for record in result.epochs if record.val_acc >= threshold
        )
        final = result.epochs[-1]
        summary[optimizer_name] = {
            "best_val_acc": best_val_acc,
            "best_epoch": max(result.epochs, key=lambda record: record.val_acc).epoch,
            "epochs_to_90pct_best": convergence_epoch,
            "mean_epoch_time_s": result.mean_epoch_time_s,
            "final_train_val_gap": final.train_acc - final.val_acc,
            "final_train_acc": final.train_acc,
            "final_val_acc": final.val_acc,
            "epochs_completed": len(result.epochs),
        }
    return summary
