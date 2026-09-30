# `models/` — SEAM 2 producer

**Shared between Member 2 (`custom/`) and Member 4 (`pretrained/`).**
`registry.py` is a shared file: PR + 1 review.

## Inputs

| From | What |
|---|---|
| `DataBundle` | `num_classes` |
| `configs/stages/models.yaml` | the `model_a` and `model_b` sections (Member 2) |
| `configs/stages/pretrained.yaml` | the `pretrained` section: resolution, backbones, freezing, LR scale (Member 4) |

## Outputs

```python
from edgecnn.models import build_model_from_config
model = build_model_from_config(cfg, num_classes=data.num_classes)
```

An `nn.Module` satisfying `contracts.protocols.ClassifierModel`:

```
in  : float32 (B, 3, 64, 64)
out : float32 (B, num_classes)   RAW LOGITS
```

Registry keys: `model_a`, `model_b` (Member 2) · `mobilenet_v2`, `squeezenet1_1` (Member 4).

## One way to build a model

`build_model_from_config` is the only construction path notebooks use. It:

1. finds the model's settings — `cfg.section("model_b")` for a custom model, or the matching
   `pretrained.backbones` entry for a backbone — and passes them to the builder as `**overrides`;
2. works out the input shape — 64×64, or `pretrained.input_resolution` for the 128px ablation;
3. seeds torch with `cfg.seed` immediately before construction, so initial weights are identical
   whichever notebook cells ran first. The three §3 optimizer runs depend on this.

Building models this way everywhere is what guarantees that the model trained in `03`–`05` is the
same architecture profiled in `06`. Never assemble the overrides by hand in a cell.

The lower-level `build_model(name, num_classes, input_shape, seed=..., **overrides)` is there for
tests and one-off checks.

## Why the registry exists

Member 3's trainer must be identical for all four models. If it imported model modules directly, it
would grow a branch per architecture. The four models would then drift apart — different loss
reductions, different eval-mode handling, different timing points — until the §6 comparison table
could not be defended. So the trainer only ever talks to the registry. **If the trainer needs a branch
on model name, the contract is wrong: fix the contract, not the trainer.**

## No softmax inside a model

`CrossEntropyLoss` applies log-softmax itself. A model that also softmaxes trains on a
double-softmaxed signal: it does not crash, it quietly underperforms. `tests/contracts/` checks for
it by asserting output rows do not sum to 1. It also matters for §2's hardware-aware argument:
softmax is an exponential per class per inference, which is wasted work when all you need is the
predicted label.

## Subfolders

- [`custom/`](custom/README.md) — Member 2, §2 [20 marks], notebook `02`
- [`pretrained/`](pretrained/README.md) — Member 4, §5 [20 marks], notebook `05`
