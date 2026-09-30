# `models/custom/` — Model A and Model B

**Owner: Member 2.** Assignment §2 — **20 marks**.
**Notebook:** [`02_custom_architectures`](../../../../notebooks/02_custom_architectures.ipynb).
The models are trained by Member 3 in `03` (Model B) and `04` (Model A).

## Inputs

`num_classes` from the `DataBundle`, and — as the builder's `**overrides` — the `model_a` / `model_b`
sections of [`configs/stages/models.yaml`](../../../../configs/stages/models.yaml): `blocks`, `head`,
`activation`, `batch_norm` (plus `stem` for Model B). `build_model_from_config` passes them in, so
widths and depths come from the config, never from constants in the code.

## Outputs

- two registry entries, `model_a` and `model_b`;
- two tables for the report, written by `architecture_tables(..., write=cfg.is_official)` in notebook
  `02`:
  - `results/tables/custom_architectures.md` — layer-by-layer topology
  - `results/tables/param_breakdown.md` — per-layer parameter derivation

Both models return **raw logits**, expose `.num_classes`, and work at batch size 1, which is where
Member 1 measures latency.

## Files

| File | Purpose |
|---|---|
| `model_a.py` | the standard CNN |
| `model_b.py` | the depthwise-separable CNN, ≤100k parameters |
| `blocks.py` | `conv_bn_act`, `depthwise_separable_conv`, `get_activation` |
| `summary.py` | `layer_table`, `architecture_tables` — the §2 tables |

## How to work on it

Prototype in `notebooks/scratch/m2_<topic>.ipynb`. Move the class into `model_a.py` / `model_b.py`
once it works, then re-run notebook `02`. `%autoreload` picks up each edit. The full walkthrough is
[CONTRIBUTING.md §4 — Member 2 builds Model B](../../../../CONTRIBUTING.md#4-worked-example--member-2-builds-model-b).

## Model A — standard CNN

Interleaved `Conv2d` + `MaxPool2d` with a fully connected head. It is deliberately **not**
parameter-constrained: it is the control that shows what Model B gives up.

Worth stating explicitly in the report: flattening an 8×8×128 map into `FC(128)` costs
`8*8*128*128 = 1,048,576` weights in a single layer — about ten times Model B's entire budget. That
one line is the clearest illustration of where the parameters in a small CNN actually go.

## Model B — depthwise-separable CNN

**Hard cap: 100,000 trainable parameters.** `tests/test_param_budget.py` fails if it is exceeded, and
notebook `02` asserts it too.

Two distinct mechanisms carry the saving. Keeping them separate makes a stronger §2 answer than
crediting everything to the convolutions:

**1. Depthwise separable convolution.**

```
standard  : k^2 * C_in * C_out
separable : k^2 * C_in  +  C_in * C_out
ratio     : 1/C_out + 1/k^2         ~= 1/8 at k=3, C_out=64
```

**2. Global average pooling instead of flatten.** This replaces Model A's ~1.05M-weight FC layer with
`96 -> num_classes`, under a thousand weights. Within a 100k budget it saves *more* than the
depthwise convolutions do.

## What §2 asks for beyond the code

- **Explicit network parameters:** kernel sizes, filter counts and FC widths, stated in the report.
- **Total trainable parameters, calculated:** derive them by hand, layer by layer. `layer_table`
  checks your arithmetic; it does not replace it.
- **A hardware-aware justification of the activation.** ReLU and ReLU6 are a single `max()`: no
  exponential, no lookup table, no floating-point unit required. sigmoid, tanh, ELU, GELU and Swish
  each need a transcendental function per activation, which dominates runtime on a microcontroller.
  ReLU6's bounded range also keeps int8 quantisation well-scaled, which is what actually gets a model
  onto an MCU.

## The bug that will cost you the budget

A depthwise convolution built with `groups=1` instead of `groups=in_channels` still trains perfectly
well. It is just a normal convolution with several times the parameters, and nothing else in the
pipeline complains. `tests/test_param_budget.py` is what catches it.
