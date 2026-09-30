"""Reproducibility: one call seeds every random number generator the project uses.

Two runs of the same configuration should produce the same numbers. That matters
twice over here: the assignment asks for seeds to be recorded, and the optimizer
comparison is only meaningful if every optimizer starts from the same initial
weights and sees the same batches.

Team notes:
Owner: Member 1. Called at the start of Trainer.fit, and by build_model_from_config
(through torch.manual_seed) before a model is constructed.
"""

from __future__ import annotations

import os
import random

#: cuBLAS needs this workspace setting for deterministic GPU matrix products.
#: It only takes effect if set before CUDA is first used in the process.
CUBLAS_WORKSPACE = ":4096:8"


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU and every GPU).

    Args:
        seed: The seed. Every reported run uses 42.
        deterministic: Also make GPU computation repeatable: cuDNN picks
            deterministic kernels and stops auto-tuning, and PyTorch prefers
            deterministic algorithms where it has them. This costs some speed.
            Without it, two runs of the same configuration can differ by a few
            tenths of a percent - the same size as the differences between
            models that the report compares.

    A few GPU operations have no deterministic implementation. PyTorch then
    emits a warning instead of stopping the run, and the report should note
    that such runs are reproducible only up to that operation.
    """
    import numpy as np
    import torch

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds the CPU and every CUDA device

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def worker_init_fn(worker_id: int) -> None:
    """Seed NumPy and Python's ``random`` inside a DataLoader worker process.

    PyTorch already gives each worker its own torch seed (the loader's base seed
    plus the worker id), but NumPy and ``random`` would otherwise start every
    worker from the same state. Pass this as ``DataLoader(worker_init_fn=...)``.

    Args:
        worker_id: Supplied by the DataLoader; unused, because
            ``torch.initial_seed()`` is already unique per worker.
    """
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
