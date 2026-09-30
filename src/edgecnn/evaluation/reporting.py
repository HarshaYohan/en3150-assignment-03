"""SEAM 5 - aggregation into report tables and figures.

Owner: Member 4.                  Assignment Sections 4 [25], 5 [20], 6 [20]
Driven from notebooks/07_final_comparison.ipynb; the confusion-matrix plot is
also used inline in notebooks 04 and 05.

    +---------------------------------------------------------------------+
    |  IN   results/metrics/*/history.json       (Member 3, 03/04/05)      |
    |       results/metrics/*/test_metrics.json  (Member 1, 03/04/05)      |
    |       results/metrics/*/resources.json     (Member 1, 06)            |
    |                                                                      |
    |  OUT  tables as Markdown text + Figures - always returned            |
    |       with write=True (official), also saved:                        |
    |       results/tables/custom_model_comparison.md    Section 4         |
    |       results/tables/optimizer_comparison.md       Section 3         |
    |       results/tables/final_comparison.md           Section 6         |
    |       results/figures/*.png  ->  report/figures/                     |
    +---------------------------------------------------------------------+

This is the last stage, and it consumes only committed JSON. That is why
``results/metrics/**/*.json`` is committed while checkpoints are not: Member 4
can rebuild every table without a GPU and without re-running anyone's training.

Every function here takes an explicit ``write`` flag. Per-run plots get
``write=cfg.is_official`` from the notebook; aggregation spans several runs, so
notebook 07 passes ``write = (MODE == "official")``. Results are returned
either way, so a debug pass still shows everything inline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure

    from edgecnn.contracts.types import EvalResult

#: A committed run, or an evaluation still in memory (debug runs).
EvalSource = Union[str, "EvalResult"]


def load_all_runs() -> dict[str, dict[str, Any]]:
    """Load every discoverable run's three JSON artifacts.

    Uses ``paths.discover_runs()`` rather than a hard-coded list, so a run
    joins the tables as soon as its JSON is committed.

    Returns:
        ``{run_id: {"history": ..., "test_metrics": ..., "resources": ...}}``.

    A run missing ``test_metrics.json`` has been trained but not evaluated -
    skip it with a warning naming the run and the missing file, rather than
    crashing. Half the team's runs landing is the normal state of the repo for
    most of the project.
    """
    raise NotImplementedError("Member 4: implement load_all_runs")


def check_same_hardware(run_ids: list[str]) -> dict[str, list[str]]:
    """Group runs by the device string recorded in their artifacts.

    Epoch times are compared across the runs of one table, so those runs must
    have trained on the same runtime - e.g. notebooks 03 and 04 both on one
    laptop, or both on a Colab T4. Warn, naming the runs, when a table mixes
    devices; the report must then either re-run or say so.

    Returns:
        ``{device_string: [run_ids]}``.
    """
    raise NotImplementedError("Member 4: implement check_same_hardware")


def build_custom_comparison_table(write: bool = True) -> str:
    """Section 4: Model A vs Model B. Returns Markdown; saves it when ``write``.

    Required columns, from the assignment text: total parameter count,
    estimated model size on disk (KB), training time per epoch, test accuracy.
    Add MACs - it is the number that actually demonstrates the depthwise
    separable saving, and accuracy per parameter is a weak proxy for it.

    The accompanying discussion has to state the trade-off in both directions:
    what accuracy was given up, and what was bought with it.
    """
    raise NotImplementedError("Member 4: implement build_custom_comparison_table")


def build_optimizer_comparison_table(write: bool = True) -> str:
    """Section 3: SGD vs SGD+momentum vs the chosen optimizer.

    Columns: optimizer, learning rate, best epoch, best validation accuracy,
    test accuracy, epochs to reach 90% of that run's own best validation
    accuracy, and mean epoch time.

    Include a memory column or note: Adam holds two extra state tensors per
    parameter, roughly tripling optimizer memory versus plain SGD. In a report
    about memory-constrained devices that is a relevant cost, not a footnote -
    though note it applies at training time, not at inference.
    """
    raise NotImplementedError("Member 4: implement build_optimizer_comparison_table")


def build_final_comparison_table(write: bool = True) -> str:
    """Section 6: Model B vs the fine-tuned SOTA backbones.

    The central table of the report. Columns: model, trainable params, total
    params, size (KB and MB), MACs, CPU inference latency, peak memory, test
    accuracy, macro precision, macro recall.

    Points the discussion should make, each of which the table should evidence:

    * Model B is likely to lose on accuracy and win by one to two orders of
      magnitude on parameters and size. State the ratio, not just both numbers.
    * The SOTA models were designed for 224x224. At 64x64 MobileNetV2's 32x
      stride leaves a 2x2 feature map, so they are handicapped here - a fair
      report says so rather than claiming a clean win for the custom model.
    * Pretrained weights bring ImageNet features for free, which matters most
      when training data is scarce. EuroSAT has 27,000 images, which is enough
      to train from scratch; the advantage would be far larger at 2,700.
    * Peak activation memory, not parameter count, is often what actually
      blocks MCU deployment. A model can fit in flash and still not run.
    * The honest conclusion may well be "it depends on the memory budget" -
      identify the threshold at which the answer flips, rather than declaring a
      winner. That is a stronger answer than either extreme.
    """
    raise NotImplementedError("Member 4: implement build_final_comparison_table")


def plot_confusion_matrix(
    source: EvalSource,
    normalize: bool = True,
    write: bool = False,
) -> Figure:
    """Render one run's confusion matrix; returns the Figure.

    ``source`` is a ``run_id`` (reads the committed ``test_metrics.json``) or an
    ``EvalResult`` still in memory. With ``write=True`` - which notebooks pass as
    ``cfg.is_official`` - also saves ``paths.confusion_matrix_png(run_id)``.

    Stored counts are raw; normalise at plot time only. Normalise by TRUE class
    (rows sum to 1) so the diagonal reads as per-class recall - normalising by
    column gives precision instead, and an unlabelled plot leaves the reader
    unable to tell which.

    Label both axes with ``class_names`` and rotate the x labels; EuroSAT class
    names are long enough to overlap otherwise.
    """
    raise NotImplementedError("Member 4: implement plot_confusion_matrix")


def plot_tradeoff_scatter(x_metric: str = "trainable_params", write: bool = True) -> Figure:
    """The Section 6 argument in one figure; returns the Figure.

    Test accuracy against cost, one point per model, log-scaled x. Produce it
    for both ``trainable_params`` and ``macs`` - they tell different stories,
    because parameter count tracks memory while MACs track compute, and a model
    can be cheap in one and expensive in the other.

    Label each point with its model name and draw the Pareto frontier. A reader
    should be able to see at a glance which models are dominated and which
    represent a real choice.
    """
    raise NotImplementedError("Member 4: implement plot_tradeoff_scatter")


def export_report_figures(write: bool = True) -> list[Path]:
    """Copy every committed figure into ``report/figures/``.

    Keeps one source of truth: figures are generated into ``results/figures/``
    and mirrored, never hand-copied or regenerated separately for the report.
    With ``write=False``, list what would be copied without copying.
    """
    raise NotImplementedError("Member 4: implement export_report_figures")
