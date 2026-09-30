"""Console logging with one consistent format.

One setup for every notebook means runs produce comparable output, which makes a
pasted log easy to read when comparing timings between machines.

Team notes:
Owner: Member 1.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"
_MARK = "_edgecnn_handler"  # tags the handlers this module owns


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure the root logger: timestamp, level and logger name on each line.

    Safe to call more than once - re-running a notebook cell replaces the
    handlers this function added rather than stacking duplicates. Output goes to
    standard output, which notebooks display as ordinary text.

    Args:
        level: A logging level name, e.g. ``"INFO"`` or ``"DEBUG"``.
        log_file: Also write every line to this file, so a long run leaves a
            record that survives closing the notebook.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _MARK, False):
            root.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        setattr(handler, _MARK, True)
        root.addHandler(handler)
    root.setLevel(level.upper() if isinstance(level, str) else level)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger. Use ``get_logger(__name__)``."""
    return logging.getLogger(name)
