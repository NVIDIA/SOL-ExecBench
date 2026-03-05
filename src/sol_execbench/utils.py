"""Utility functions for FlashInfer-Bench."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from .data import SupportedHardware, Environment


def env_snapshot(hardware: SupportedHardware) -> Environment:
    import torch
    import triton

    libs: dict[str, str] = {"torch": torch.__version__}
    try:
        libs["triton"] = getattr(triton, "__version__", "unknown")
    except Exception:
        pass

    if torch.cuda.is_available():
        libs["cuda"] = torch.version.cuda
    return Environment(hardware=hardware.value, libs=libs)


def redirect_stdio_to_file(log_path: str) -> tuple[int, int]:
    """Redirect stdout/stderr to log file.

    Returns original stdout and stderr file descriptors for printing to terminal.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    original_stdout_fd = os.dup(1)
    original_stderr_fd = os.dup(2)
    fd = os.open(log_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
    # Redirect stdout/stderr to log file
    os.dup2(fd, 1)  # stdout -> fd
    os.dup2(fd, 2)  # stderr -> fd
    os.close(fd)
    sys.stdout = open(1, "w", encoding="utf-8", buffering=1, closefd=False)
    sys.stderr = open(2, "w", encoding="utf-8", buffering=1, closefd=False)
    return original_stdout_fd, original_stderr_fd


def flush_stdio_streams() -> None:
    """Best-effort flush of redirected stdout/stderr streams."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
