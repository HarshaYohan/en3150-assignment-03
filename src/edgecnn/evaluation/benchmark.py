"""SEAM 4 - resource profiling.          Assignment Sections 4 [25] + 6 [20]

Owner: Member 1.  Used for EVERY model, from notebooks/06_resource_benchmark.ipynb.

    +---------------------------------------------------------------------+
    |  IN   run_ids + their experiment configs (architecture only)        |
    |       history.json  (for mean_epoch_time_s)                          |
    |  OUT  ResourceProfile per run - always returned                      |
    |       -> results/metrics/<run_id>/resources.json   (official only)   |
    +---------------------------------------------------------------------+

This module produces the cost side of the accuracy/memory/compute trade-off
that Section 6 is entirely about. Every number here must come from one code
path, run on one device, or the comparison is not a comparison.

Why a separate notebook, in one session: latency and peak memory depend on
the machine, but NOT on trained weights - an architecture costs the same with
random weights as with trained ones. So notebook 06 builds all four
architectures fresh and profiles them back to back on one CPU. No checkpoints
are needed, and every row of the Section 6 table was measured on the same
hardware, whoever trained which model where.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn

    from edgecnn.contracts.types import ResourceProfile


def profile_model(
    model: nn.Module,
    input_shape: tuple[int, int, int],
    run_id: str,
    model_name: str,
    mean_epoch_time_s: float = 0.0,
    latency_device: str = "cpu",
    latency_repeats: int = 100,
    latency_warmup: int = 10,
) -> ResourceProfile:
    """Measure parameters, size, MACs, latency and peak memory.

    What each number means, and the trap in each:

    * **trainable_params** - ``requires_grad=True`` only. Equals total for the
      custom models; smaller for a partially frozen backbone.
    * **total_params** - everything. *This*, not the trainable count,
      determines the memory an edge device needs at inference. Section 5 asks
      for both, and conflating them flatters the fine-tuned models.
    * **model_size_kb** - serialise the ``state_dict`` to a temp file and
      measure it, rather than computing ``numel * 4``. The computed figure
      misses buffers such as BatchNorm running statistics, which are real
      bytes on the device.
    * **macs** - from :func:`count_macs`: convolution and fully-connected
      multiply-accumulates for one image. Papers that count FLOPs report roughly
      ``2 x MACs``; state which is used in the report, or the comparison against
      published MobileNet figures will look wrong by a factor of two.
    * **inference_latency_ms** - batch size 1, ``model.eval()``,
      ``torch.no_grad()``, warm up first. Measure on **CPU for every model**
      even on a CUDA machine: the Section 6 argument is about edge deployment,
      and edge devices have no GPU. On GPU you would also need
      ``torch.cuda.synchronize()`` or the timing measures kernel-queueing
      rather than compute.
    * **peak_mem_mb** - peak activation plus parameter memory in a forward
      pass. Often the binding constraint on a microcontroller, and frequently
      the real reason a model cannot be deployed even when its parameter count
      looks acceptable.

    ``mean_epoch_time_s`` is copied from ``history.json``, never re-measured -
    Member 3's trainer is the only thing that times epochs.
    """
    raise NotImplementedError("Member 1: implement profile_model")


def count_macs(model: nn.Module, input_shape: tuple[int, int, int]) -> int:
    """Multiply-accumulate operations (MACs) for one image's forward pass.

    Counts the convolutions and fully-connected layers - the convention of the
    MobileNet and SqueezeNet papers and of torchvision's published model
    figures, which this reproduces exactly (MobileNetV2 0.301 G, SqueezeNet 1.1
    0.349 G, at 224x224). Batch-norm, activations, pooling and bias additions
    are left out: they are cheap next to the convolutions, and counting them
    would make the numbers incomparable with published ones.

    * a convolution costs output values x (input channels / groups) x kernel
      area - so a depthwise convolution, with one group per channel, costs a
      fraction of a standard one;
    * a fully-connected layer costs output values x input features.

    One MAC is one multiply and one add; papers that count FLOPs report
    roughly twice as many. The model is measured on a copy in evaluation mode,
    so the caller's model - its weights, batch-norm statistics and train /
    eval mode - is untouched.

    Args:
        model: Any model that takes a ``(1, *input_shape)`` batch.
        input_shape: ``(channels, height, width)`` of one image.

    Returns:
        The number of MACs at batch size 1.
    """
    import copy
    import math

    import torch
    from torch import nn

    probe = copy.deepcopy(model).cpu().eval()
    total = 0

    def conv(module: Any, inputs: Any, output: torch.Tensor) -> None:
        nonlocal total
        per_value = (module.in_channels // module.groups) * math.prod(module.kernel_size)
        total += output[0].numel() * per_value

    def linear(module: Any, inputs: Any, output: torch.Tensor) -> None:
        nonlocal total
        total += output[0].numel() * module.in_features

    hooks = []
    for module in probe.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            hooks.append(module.register_forward_hook(conv))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear))
    try:
        with torch.no_grad():
            probe(torch.zeros(1, *input_shape))
    finally:
        for hook in hooks:
            hook.remove()
    return int(total)


def measure_latency(
    model: nn.Module,
    input_shape: tuple[int, int, int],
    device: str = "cpu",
    batch_size: int = 1,
    warmup: int = 10,
    repeats: int = 100,
) -> float:
    """Mean single-sample inference latency in milliseconds.

    Warm-up matters: the first few forward passes pay lazy CUDA context setup,
    cuDNN algorithm selection and allocator warm-up, and including them can
    inflate the mean several-fold.

    Report the median alongside the mean if the variance is high - a laptop
    under thermal throttling produces a long tail that makes the mean
    unrepresentative.
    """
    raise NotImplementedError("Member 1: implement measure_latency")


def measure_model_size_kb(model: nn.Module) -> float:
    """Serialised ``state_dict`` size in kilobytes.

    Save to a temporary file with ``torch.save`` and measure it. Always return
    KB; ``resources.schema.json`` fixes the unit so a KB/MB mix-up cannot
    reach the Section 6 table, where it would make the argument wrong rather
    than merely imprecise.
    """
    raise NotImplementedError("Member 1: implement measure_model_size_kb")


def profile_all(
    run_ids: list[str],
    *,
    mode: str = "official",
    latency_device: str = "cpu",
) -> dict[str, ResourceProfile]:
    """Profile every run's architecture in ONE session - notebook 06.

    For each ``run_id``:

    1. ``cfg = load_config(paths.experiment_config(run_id), mode=mode)``
    2. ``model = build_model_from_config(cfg, num_classes)`` - the same builder
       and settings the training notebook used, so the profiled model is the
       trained architecture. ``num_classes`` comes from ``split_meta.json``.
    3. :func:`profile_model` on ``latency_device``; ``mean_epoch_time_s`` is
       copied from that run's ``history.json`` when it exists.
    4. Only if ``cfg.is_official``: write ``resources.json`` through
       ``schema.write_json``.

    Runs that share an architecture (the three ``model_b`` optimizer runs)
    may reuse one measurement - only their epoch times differ.

    Returns:
        ``{run_id: ResourceProfile}`` in every mode, for display in the notebook.
    """
    raise NotImplementedError("Member 1: implement profile_all")
