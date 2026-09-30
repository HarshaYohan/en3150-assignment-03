# `tests/contracts/` — the cross-member seam tests

**Run after every `git pull`:**

```powershell
pytest tests/contracts
```

| File | Checks | Needs torch? |
|---|---|---|
| `test_schemas.py` | schemas are valid, accept good artifacts, reject bad ones | no |
| `test_paths_and_configs.py` | `run_id` round-trips; artifact paths agree; debug checkpoints are kept apart; all six experiments load | no |
| `test_config_layout.py` | one home per setting; `stages.<name>` I/O blocks; no stage file defines an optimizer; `MODE` presets and `is_official` | no |
| `test_seams.py` | Seam 1 batch shapes; Seam 2 logits for all four models; same seed gives the same weights; every experiment builds from its config | yes |

The first three need neither torch nor a dataset, so they run in about a second. `00_setup.ipynb`
runs this folder as its last check.

## Green by design

Tests whose subject is still a stub **skip** rather than fail, through the `pending()` helper in
[`../conftest.py`](../conftest.py). They turn into real assertions the moment that stub is
implemented. A failure therefore really means something broke — and that is only useful if the suite
is normally green. See [../README.md](../README.md).

## What belongs here

Tests of the **contract**, not of an implementation. "Model B returns logits of the right shape"
belongs here. "Model B's third conv layer has 64 filters" does not: Member 2 must stay free to retune
the architecture without breaking someone else's test.
