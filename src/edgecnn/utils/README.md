# `utils/` — shared helpers

**Owner: Member 1.** Used by all four members. Few dependencies, and no side effects on import.

| File | Purpose |
|---|---|
| `seed.py` | `seed_everything`, `worker_init_fn` |
| `device.py` | `resolve_device`, `describe_device` |
| `logging.py` | `setup_logging`, `get_logger` |
| `io.py` | `sha256_file`, `ensure_dir`, `write_markdown_table` |
| `colab.py` | `in_colab`, `push_results` — the Colab fallback. **Implemented**; the others are stubs |

## Inputs / outputs

**Inputs:** plain arguments — no dataset, no model.
**Outputs:** helpers imported by every other package and by the notebooks.

## Why seeding matters here

The assignment asks for random seeds to be recorded. The four-way split makes seeding matter for a
second reason. If Member 3's Adam run and SGD run start from different initial weights, the §3
comparison measures initialisation noise as well as the optimizer, and the conclusion about momentum
is not supported.

Seeding happens in two places, so results never depend on which notebook cells ran earlier:

- `build_model_from_config` seeds torch immediately before building the model (initial weights);
- `Trainer.fit` calls `seed_everything(cfg.seed)` at the start (dropout, augmentation, batch order).

`worker_init_fn` matters too. Without it, DataLoader workers get unseeded random states and
augmentation differs between runs, even when `seed_everything` was called.

## Why `describe_device` exists

Section 4 requires the evaluation hardware to be reported, and a timing without a device string is
meaningless. This produces the string that goes into `resources.json` and the report table, e.g.
`cpu (11th Gen Intel Core i7-11800H)`.

## `colab.py` and the token rule

A notebook's last cell calls `push_results(...)` after an official run on Colab, to commit and push
that run's result files before the session is wiped. The repository is **public** and notebook
outputs are committed, so the GitHub token:

- is read from Colab Secrets, never typed into a cell;
- reaches git only through an environment variable read by a credential helper — never in a URL or
  a command-line argument;
- is scrubbed from git's output before anything is printed.

`tests/test_colab.py` checks all three with git mocked out. The full procedure is in
[`notebooks/README.md` → Colab](../../../notebooks/README.md#colab-fallback).

## For JSON that crosses a member boundary

Use `edgecnn.contracts.schema.read_json` / `write_json` instead of anything here — those validate
against the contract.
