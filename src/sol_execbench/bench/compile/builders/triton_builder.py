"""Builder for Triton GPU kernels."""

from __future__ import annotations

import importlib.util
from typing import ClassVar

from ..builder import Builder
from ..runnable import Runnable
from ....data import Definition, Solution, SupportedLanguages

from .python_builder import PythonBuilder


class TritonBuilder(PythonBuilder):
    """Builder for Triton solutions.

    This builder extends PythonBuilder to handle Triton GPU kernels. Triton code
    is Python-based, so the build process is similar to PythonBuilder, with the
    main difference being the language tag in metadata.
    """

    _PACKAGE_PREFIX: ClassVar[str] = "sol_execbench_triton_"
    """Prefix for cache keys to distinguish Triton solutions from pure Python ones."""

    _BUILD_DIR_NAME: ClassVar[str] = "triton"
    """Subdirectory under SOLEXECBENCH_CACHE_PATH where build results are stored"""

    def __init__(self) -> None:
        Builder.__init__(self, self._PACKAGE_PREFIX, self._BUILD_DIR_NAME)

    @staticmethod
    def is_available() -> bool:
        """Check if Triton is available in the current environment.

        Returns
        -------
        bool
            True if Triton is installed, False otherwise.
        """
        return importlib.util.find_spec("triton") is not None

    def can_build(self, solution: Solution) -> bool:
        """Check if this builder can build the given solution.
        The solution should be Triton source code.

        Parameters
        ----------
        solution : Solution
            Solution to check

        Returns
        -------
        bool
            True if solution language is Triton
        """
        return solution.spec.language == SupportedLanguages.TRITON

    def build(self, definition: Definition, solution: Solution) -> Runnable:
        """Build a Triton solution into a runnable.

        This method delegates to PythonBuilder.build() and updates the build_type
        in metadata to 'triton'.

        Parameters
        ----------
        definition : Definition
            The problem definition.
        solution : Solution
            The Triton solution to build.

        Returns
        -------
        Runnable
            An executable wrapper around the Triton kernel.
        """
        result = super().build(definition, solution)
        result.metadata.build_type = "triton"
        return result
