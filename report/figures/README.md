# `report/figures/`

**Do not put anything here by hand.**

Notebook [`07_final_comparison`](../../notebooks/07_final_comparison.ipynb) copies `results/figures/`
into this folder with `export_report_figures(write=True)` during its official run. With one source of
truth, a figure in the report can never drift out of sync with the numbers in the tables beside it.

If a figure is missing, the run that produces it has not been committed yet. Check
`results/metrics/` for the corresponding `<run_id>`, and see which notebook produces it in
[`results/figures/README.md`](../../results/figures/README.md).
