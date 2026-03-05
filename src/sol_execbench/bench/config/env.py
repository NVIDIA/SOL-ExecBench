"""Defines the environment variables used in sol_execbench evaluate."""

import os
from pathlib import Path


def get_sol_execbench_dataset_path() -> Path:
    """Get the value of the SOLEXECBENCH_DATASET_PATH environment variable. It controls the path to the
    dataset to dump or to load.

    Returns
    -------
    Path
        The value of the SOLEXECBENCH_DATASET_PATH environment variable.
    """
    value = os.environ.get("SOLEXECBENCH_DATASET_PATH")
    if value:
        return Path(value).expanduser()
    return Path(Path.home() / ".cache" / "sol_execbench" / "dataset")


def get_sol_execbench_cache_path() -> Path:
    """Get the value of the SOLEXECBENCH_CACHE_PATH environment variable. It controls the path to the cache.

    Returns
    -------
    Path
        The value of the SOLEXECBENCH_CACHE_PATH environment variable.
    """
    value = os.environ.get("SOLEXECBENCH_CACHE_PATH")
    if value:
        return Path(value).expanduser()
    return Path.home() / ".cache" / "sol_execbench" / "cache"
