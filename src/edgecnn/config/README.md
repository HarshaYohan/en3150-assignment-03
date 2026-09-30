# `config/` — layered YAML loading

**Owner: Member 1.** Used by every notebook and every member.

## Inputs / outputs

**Input:** a path to `configs/experiments/<run>.yaml`, plus the notebook's `MODE`.
**Output:** a `ResolvedConfig`. It holds the merged mapping, the handful of fields every signature
needs (`run_id`, `model_name`, `optimizer_name`, `seed`, `device`, `mode`) and `is_official`.

```python
from edgecnn.config import load_config

cfg = load_config("configs/experiments/model_b__adam.yaml", mode=MODE)
cfg.run_id                              # "model_b__adam__seed42"
cfg.section("training")["epochs"]       # 30, or 2 in debug / synthetic
cfg.section("optimizer")["name"]        # "adam" - defined only in the experiment file
cfg.is_official                         # the write switch every library function obeys
```

## The merge

```
base.yaml  <-  stages/*.yaml (via `extends:`)  <-  experiments/<run>.yaml  <-  modes[MODE]  <-  overrides
```

Mappings merge recursively. **Lists are replaced wholesale.** Every setting lives in one top-level
section with one home, and stage I/O declarations sit under `stages.<name>`. The rules and the bug
that motivated them are in [`configs/README.md`](../../../configs/README.md).

## Guarantees

- **Paths are repo-relative.** `resolve_path` anchors them to `paths.REPO_ROOT`, which comes from
  `__file__`, so the same YAML works from `notebooks/`, from `tests/` and in a Colab clone.
- **An unknown mode is rejected** with the list of valid ones.
- **A `run_id` cannot disagree with itself.** If a config states `run_id:` and it does not match what
  `model`, `optimizer` and `seed` imply, loading raises `ConfigError`. Without that check, artifacts
  would be written to one directory and read from another — a failure that is silent, because a glob
  matching nothing raises nothing.
