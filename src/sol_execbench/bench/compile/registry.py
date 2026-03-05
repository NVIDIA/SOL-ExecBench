"""Builder registry for dispatching and caching builds."""

from __future__ import annotations

from typing import ClassVar, Type

import filelock

from ...data import (
    BuildSpec,
    Definition,
    Solution,
    SourceFile,
    SupportedHardware,
    SupportedLanguages,
)
from ..config import get_sol_execbench_cache_path

from .builder import Builder, BuildError
from .builders import PythonBuilder, TorchBuilder, TritonBuilder
from .runnable import Runnable

_BUILDER_PRIORITY: list[Type[Builder]] = [
    TritonBuilder,
    PythonBuilder,
    TorchBuilder,
]
"""Builder types in priority order for automatic selection."""


class BuilderRegistry:
    """Central registry for managing and dispatching builders.

    The BuilderRegistry maintains a list of available builders and automatically selects
    the appropriate one for each solution. It also provides caching to avoid redundant
    builds of the same solution.

    This class follows the singleton pattern - use get_instance() to obtain the shared
    registry instance.
    """

    _instance: ClassVar["BuilderRegistry" | None] = None
    """Singleton instance of the BuilderRegistry."""

    _builders: list[Builder]
    """List of available builders in priority order."""

    _cache: dict[str, Runnable | BuildError]
    """Cache mapping solution hashes to built runnables or build errors."""

    def __init__(self, builders: list[Builder]) -> None:
        """Initialize the registry with a list of builders.

        Parameters
        ----------
        builders : list[Builder]
            List of builder instances to use. Must contain at least one builder.

        Raises
        ------
        ValueError
            If the builders list is empty.
        """
        if len(builders) == 0:
            raise ValueError("BuilderRegistry requires at least one builder")
        self._builders = list(builders)
        self._cache: dict[str, Runnable | BuildError] = {}

    @classmethod
    def get_instance(cls) -> "BuilderRegistry":
        """Get the singleton registry instance.

        On first call, this method initializes the registry by instantiating all
        available builders (those whose is_available() returns True) in priority order.
        Subsequent calls return the same instance.

        The following builders are available (high to low priority):

        - TritonBuilder: Build Triton solutions.
        - PythonBuilder: Build Python solutions.
        - TorchBuilder: Build CUDA/C++ solutions using PyTorch extension system.

        Returns
        -------
        BuilderRegistry
            The shared registry instance.
        """
        if cls._instance is None:
            builders = []
            for builder_type in _BUILDER_PRIORITY:
                if builder_type.is_available():
                    builders.append(builder_type())
            cls._instance = BuilderRegistry(builders)
        return cls._instance

    def build(self, definition: Definition, solution: Solution) -> Runnable:
        """Build a solution into a runnable, using cache if available.

        This method first checks if the solution has already been built (by comparing
        its hash). If not, it tries each registered builder in priority order until
        one reports it can build the solution. The resulting runnable is cached for
        future use.

        This method is process-safe: concurrent builds of the same solution are
        serialized using file locks.

        Parameters
        ----------
        definition : Definition
            The problem definition specifying the expected interface.
        solution : Solution
            The solution to build.

        Returns
        -------
        Runnable
            An executable wrapper around the built solution.

        Raises
        ------
        BuildError
            If no registered builder can build this solution, or if the build fails.
        """
        hash = solution.hash()
        if hash in self._cache:
            cached = self._cache[hash]
            if isinstance(cached, BuildError):
                raise cached
            return cached

        # Use file lock to ensure process-safe build
        lock_dir = get_sol_execbench_cache_path() / "build_locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_file = lock_dir / f"{hash}.lock"

        with filelock.FileLock(lock_file):
            # Double-check after acquiring lock
            if hash in self._cache:
                cached = self._cache[hash]
                if isinstance(cached, BuildError):
                    raise cached
                return cached

            builder = next((b for b in self._builders if b.can_build(solution)), None)
            if builder is None:
                error = BuildError(
                    f"No registered builder can build solution '{solution.name}'"
                )
                self._cache[hash] = error
                raise error

            try:
                runnable = builder.build(definition, solution)
            except BuildError as e:
                self._cache[hash] = e
                raise
            else:
                self._cache[hash] = runnable

        return runnable

    def build_reference(self, definition: Definition) -> Runnable:
        """Build the reference implementation for a definition.

        This is a convenience method that creates a pseudo-solution from the definition's
        reference code and builds it using the standard build() method.

        Parameters
        ----------
        definition : Definition
            The definition containing the reference implementation.

        Returns
        -------
        Runnable
            An executable wrapper around the reference implementation.

        Raises
        ------
        BuildError
            If the reference implementation cannot be built.
        """
        pseudo = Solution(
            name=f"{definition.name}__reference",
            definition=definition.name,
            author="__builtin__",
            spec=BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=[
                    SupportedHardware.B200,
                    SupportedHardware.LOCAL,
                ],
                entry_point="main.py::run",
                destination_passing_style=False,
            ),
            sources=[
                SourceFile(path="main.py", content=definition.reference),
            ],
            description="reference",
        )
        return self.build(definition, pseudo)

    def cleanup(self) -> None:
        """Cleanup the cache and cleanup all built runnables.

        This method calls cleanup() on all cached runnables to release resources,
        then clears the cache. Cleanup errors are caught and ignored to ensure
        all runnables are cleaned up.
        """
        for cached in self._cache.values():
            if isinstance(cached, Runnable):
                try:
                    cached.cleanup()
                except Exception:
                    pass
        self._cache.clear()
