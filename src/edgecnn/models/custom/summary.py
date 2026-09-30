"""Architecture tables - the Section 2 derivation.

Owner: Member 2.  Called from notebooks/02_custom_architectures.ipynb.

    +---------------------------------------------------------------------+
    |  IN   model_a, model_b (built with build_model_from_config)         |
    |  OUT  per-layer tables - always returned                            |
    |       + results/tables/custom_architectures.md  } write=True only   |
    |       + results/tables/param_breakdown.md       }                   |
    +---------------------------------------------------------------------+

Section 2 asks for the network parameters to be detailed explicitly and the
total trainable parameters calculated. These tables are the *check* on the
hand derivation in the report, not a replacement for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from torch import nn


def layer_table(model: nn.Module, input_shape: tuple[int, int, int]) -> pd.DataFrame:
    """One row per layer: type, kernel, stride, output shape, parameters.

    Walk the modules with a forward hook on a single ``(1, *input_shape)``
    input to record output shapes. Put the parameter formula next to the count
    (e.g. ``3*3*32*64 = 18432``) so the table doubles as the Section 2
    derivation, and end with a total that must equal the trainable count.
    Call out the largest row - for Model A it is the flatten into the first FC
    layer, which is the clearest illustration of where parameters go.
    """
    raise NotImplementedError("Member 2: implement layer_table")


def architecture_tables(
    model_a: nn.Module,
    model_b: nn.Module,
    input_shape: tuple[int, int, int],
    write: bool = False,
) -> dict[str, str]:
    """Both Section 2 tables as Markdown; with ``write=True``, also saved.

    Returns ``{"custom_architectures": md, "param_breakdown": md}``, written to
    ``paths.CUSTOM_ARCHITECTURES_TABLE`` and ``paths.PARAM_BREAKDOWN_TABLE``.
    Notebook 02 passes ``write=cfg.is_official``.
    """
    raise NotImplementedError("Member 2: implement architecture_tables")
