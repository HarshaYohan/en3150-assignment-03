# `results/figures/` — generated figures

**COMMITTED.** Written only by official runs, when a notebook passes `write=True` (i.e.
`write=cfg.is_official`).

```
results/figures/dataset/class_distribution.png   notebook 01 - §1
results/figures/dataset/sample_grid.png          notebook 01 - §1
results/figures/<run_id>/curves.png              notebooks 03-05 - §3, §4, §5 loss curves
results/figures/<run_id>/confusion_matrix.png    notebooks 04-05 - §4, §5
results/figures/optimizer_overlay.png            notebook 03 - §3, the three optimizers on one axis
results/figures/accuracy_vs_*.png                notebook 07 - §6, the trade-off
```

Figures are **generated, never hand-made**. Notebook `07` copies this folder into `report/figures/`,
so there is one source of truth, and a figure can never drift out of sync with the numbers in the
tables.

Every plot is also **returned** to the notebook and shown inline at a small resolution, which keeps
committed notebooks light. The PNG saved here is at print resolution. Style comes from
`edgecnn.evaluation.plotting`, which gives each model and each optimizer a fixed colour: a reader who
learns "orange is Model B" on the loss curves does not have to relearn it on the scatter plot.

Confusion matrices are stored as raw counts in JSON and normalised only when plotted, by **true
class**, so the diagonal reads as per-class recall.
