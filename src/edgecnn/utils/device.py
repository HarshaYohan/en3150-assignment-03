"""Device selection and hardware identification.

Timings are meaningless without the hardware they were measured on, and the
assignment requires the evaluation hardware to be reported.
:func:`describe_device` produces the string recorded next to every timing.

Team notes:
Owner: Member 1. The device string goes into resources.json and history.json.
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def resolve_device(spec: str = "auto") -> torch.device:
    """Turn a device setting from the config into a concrete ``torch.device``.

    Args:
        spec: ``"auto"`` (the GPU if one is available, else the CPU), ``"cpu"``,
            ``"cuda"`` or ``"cuda:N"``.

    Raises:
        RuntimeError: if a GPU is requested explicitly but is not available.
            Failing loudly is deliberate: silently falling back to the CPU
            would produce epoch times many times slower, with nothing to show
            why.
        ValueError: for any other device name.
    """
    import torch

    spec = (spec or "auto").strip().lower()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        device = torch.device(spec)
    except RuntimeError as exc:
        raise ValueError(f"unknown device {spec!r}; use 'auto', 'cpu', 'cuda' or 'cuda:N'") from exc

    if device.type == "cpu":
        return device
    if device.type != "cuda":
        raise ValueError(f"unsupported device {spec!r}; use 'auto', 'cpu', 'cuda' or 'cuda:N'")
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"device {spec!r} was requested but CUDA is not available here. "
            f"Use 'auto' or 'cpu', or install a CUDA build of PyTorch."
        )
    index = 0 if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"device {spec!r} was requested but only {torch.cuda.device_count()} GPU(s) exist."
        )
    return device


def describe_device(device: torch.device | str) -> str:
    """A readable hardware string, e.g. ``"cpu (11th Gen Intel Core i5-11400H @ 2.70GHz)"``.

    GPUs are named through CUDA, CPUs through the operating system. If a name
    cannot be found, the plain device type is returned instead of raising - a
    missing model name should never abort a measurement.
    """
    import torch

    device = torch.device(device)
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        try:
            return f"cuda:{index} ({torch.cuda.get_device_name(index)})"
        except Exception:
            return f"cuda:{index}"
    name = cpu_name()
    return f"cpu ({name})" if name else "cpu"


def cpu_name() -> str:
    """The CPU's marketing name, or an empty string if the OS won't say.

    ``platform.processor()`` is not enough on Windows, where it returns a family
    code such as "Intel64 Family 6 Model 141"; the registry holds the real name.
    """
    name = ""
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0])
        elif sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        name = line.split(":", 1)[1]
                        break
        elif sys.platform == "darwin":
            name = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout
    except Exception:
        name = ""
    name = name or platform.processor() or ""
    name = re.sub(r"\((R|TM|C)\)", "", name, flags=re.IGNORECASE)  # trademark marks
    return " ".join(name.split())
