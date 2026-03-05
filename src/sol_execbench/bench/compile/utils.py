"""Utility functions for building solutions."""

from __future__ import annotations

import re
from pathlib import Path

from ...data import Solution, SourceFile


def write_sources_to_path(path: Path, sources: list[SourceFile]) -> list[Path]:
    """Write source files to a directory and return their paths. Create path if not exists.

    This function writes all source files from a solution to a specified directory,
    creating subdirectories as needed. It performs security checks to prevent path
    traversal attacks and absolute path injection.

    Parameters
    ----------
    path : Path
        The root directory where source files will be written.
    sources : list[SourceFile]
        The list of source files to write. Each file's path must be relative and
        not contain parent directory references ("..").

    Returns
    -------
    list[Path]
        List of absolute paths to the written files.

    Raises
    ------
    AssertionError
        If any source file has an absolute path or contains path traversal.
    """
    path.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for src in sources:
        # Defensive assertion: path should be validated at Solution creation time
        src_path_obj = Path(src.path)

        assert not src_path_obj.is_absolute(), f"Absolute path detected: {src.path}"
        assert ".." not in src_path_obj.parts, f"Path traversal detected: {src.path}"

        src_path = path / src.path

        # Ensure parent directories exist
        src_path.parent.mkdir(parents=True, exist_ok=True)

        # Write source file
        src_path.write_text(src.content)
        paths.append(src_path)

    return paths


def create_package_name(solution: Solution, package_prefix: str = "") -> str:
    """Generate a unique package name for a solution.

    The package name is constructed from three parts:
    1. A prefix (typically identifying the builder)
    2. The normalized solution name (alphanumeric and underscores only)
    3. A 6-character hash of the solution content

    This ensures the package name is both human-readable and uniquely identifies
    the solution's content.

    Parameters
    ----------
    solution : Solution
        The solution to create a package name for.
    prefix : str, optional
        The prefix to prepend to the package name. Default is empty string.

    Returns
    -------
    str
        A unique package name in the format: {prefix}{normalized_name}_{hash}.

    Examples
    --------
    >>> create_package_name(solution, "sol_execbench_python_")
    'sol_execbench_python_rmsnorm_v1_a3f2b1'
    """
    # Normalize the solution name
    s = re.sub(r"[^0-9a-zA-Z_]", "_", solution.name)
    if not s or s[0].isdigit():
        s = "_" + s

    return package_prefix + s + "_" + solution.hash()[:6]
