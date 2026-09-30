"""Training and validation loss curves.       Assignment Section 4 [25]

Owner: Member 3.  (Metrics and tables belong to Members 1 and 4; these curves
belong to the section Member 3 is marked on.) Called in notebooks 03, 04, 05.

    +---------------------------------------------------------------------+
    |  IN   a run_id (reads the committed history.json)                    |
    |       OR a TrainResult / history dict still in memory (debug runs)   |
    |  OUT  matplotlib Figure - shown inline by the notebook               |
    |       + results/figures/<run_id>/curves.png   (write=True only)      |
    |       + results/figures/optimizer_overlay.png (write=True only)      |
    +---------------------------------------------------------------------+

Every function returns the Figure, so a notebook cell ending in
``plot_loss_curves(result, write=cfg.is_official)`` displays it. The PNG is
saved only when ``write`` is True - which notebooks set from
``cfg.is_official``, so a debug run can never overwrite an official figure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from edgecnn.contracts.types import TrainResult

#: What a plotting function accepts: a committed run, or a result in memory.
HistorySource = Union[str, "TrainResult", dict[str, Any]]


def plot_loss_curves(source: HistorySource, write: bool = False) -> Figure:
    """Plot train and validation loss (and accuracy) for one run.

    The assignment asks specifically for training/validation loss curves, so
    loss is the required panel; a second accuracy panel is worth adding because
    the two together are what let the report argue about overfitting.

    Plot both on the same axis, not on separate figures - the gap between them
    IS the overfitting signal, and it is invisible across two plots. Mark the
    best epoch (the checkpoint ``evaluate_run`` uses) with a vertical line, so
    the reader can see which point the reported test accuracy came from.

    Args:
        source: ``run_id`` -> read the committed ``history.json`` via
            ``schema.read_json``; ``TrainResult`` or dict -> use it directly.
        write: Also save to ``paths.curves_png(run_id)``. Pass ``cfg.is_official``.
    """
    raise NotImplementedError("Member 3: implement plot_loss_curves")


def plot_optimizer_overlay(sources: list[HistorySource], write: bool = False) -> Figure:
    """Overlay the three Section 3 optimizer runs on shared axes.

    This single figure carries most of the Section 3 argument, so it has to
    support claims about both convergence speed and final performance:

    * validation loss for all three optimizers on one axis, coloured by
      ``plotting.OPTIMIZER_COLORS``;
    * a log-scaled x-axis option - early-epoch differences are where the
      momentum effect shows, and they are compressed to nothing on a linear
      axis over 30 epochs;
    * mark each run's best epoch.

    Expect the shape of the story to be: plain SGD converges slowly and noisily,
    momentum closes most of the gap, and Adam converges fastest early. Whether
    Adam also *ends* highest is an empirical question - report what the curves
    show rather than what is expected.

    With ``write=True``, also saves ``paths.OPTIMIZER_OVERLAY_PNG``. Notebook 03
    passes True only in official mode.
    """
    raise NotImplementedError("Member 3: implement plot_optimizer_overlay")
