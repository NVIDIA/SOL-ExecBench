"""Abstract base class for solution builders."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from ...data import Definition, Solution
from ..config import get_sol_execbench_cache_path

from .runnable import Runnable
from .utils import create_package_name


class BuildError(RuntimeError):
    """Raised when a builder fails to construct a runnable implementation."""


class Builder(ABC):
    """Abstract base class for building solutions into runnable implementations.

    A Builder transforms a (Definition, Solution) pair into a Runnable object, which is an
    executable implementation of the solution. Different builders handle different programming
    languages (e.g., Python, CUDA, Triton) and build systems.

    Subclasses must implement all its abstract methods. Expectedly, the concrete builder should
    operate in the folder `FIB_CACHE_PATH / builder_specific_subfolder / package_name`, where
    `package_name` is a unique name created from the solution.
    """

    _package_prefix: str
    """The prefix to prepend to the package name."""

    _build_dir_name: str
    """The name of the build subdirectory of the concrete builder."""

    def __init__(self, package_prefix: str, build_dir_name: str) -> None:
        """Initialize the builder.

        Parameters
        ----------
        package_prefix : str
            The prefix to prepend to the package name. This should be unique for each builder type.
        build_dir_name : str
            The name of the build subdirectory of the concrete builder. This should be
            unique for each builder type.
        """
        self._package_prefix = package_prefix
        self._build_dir_name = build_dir_name

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """Check if this builder is available in the current environment.

        Override this method in subclasses to check for specific dependencies
        (e.g., CUDA, Triton, TVM). The default implementation returns True.

        Returns
        -------
        bool
            True if the builder can be used, False otherwise.
        """
        ...

    @abstractmethod
    def can_build(self, solution: Solution) -> bool:
        """Check if this builder can handle the given solution.

        Parameters
        ----------
        solution : Solution
            The solution to check.

        Returns
        -------
        bool
            True if this builder can build the solution, False otherwise.
        """
        ...

    @abstractmethod
    def build(self, definition: Definition, solution: Solution) -> Runnable:
        """Build a solution into a runnable implementation.

        This method compiles/loads the solution's source code and returns a Runnable
        object that can be executed with the interface specified by the definition.

        Parameters
        ----------
        definition : Definition
            The problem definition that specifies the expected interface.
        solution : Solution
            The solution implementation to build.

        Returns
        -------
        Runnable
            An executable wrapper around the built implementation.

        Raises
        ------
        BuildError
            If the build fails for any reason (compilation errors, missing dependencies, etc.).
        """
        ...

    def _get_package_name_and_build_path(self, solution: Solution) -> tuple[str, Path]:
        """Get the package name and build path for the solution. The package name is a unique
        identifier for the solution with only alphanumeric characters and underscores. The
        build path is FIB_CACHE_PATH / build_dir_name / package_name.

        Parameters
        ----------
        solution : Solution
            The solution to get the package name and build path for.

        Returns
        -------
        tuple[str, Path]
            The package name and build path for the solution.
        """
        package_name = create_package_name(solution, self._package_prefix)
        build_path = (
            get_sol_execbench_cache_path() / self._build_dir_name / package_name
        )
        return package_name, build_path

    def _try_validate_signature(
        self, callable: Callable, definition: Definition, solution: Solution
    ) -> None:
        """Try to validate the signature of the callable. It only checks the number of
        parameters and the return type. If the signature is unavailable, it will skip all checks.

        Parameters
        ----------
        callable : Callable
            The callable to validate.
        definition : Definition
            The definition to validate against.
        solution : Solution
            The solution to validate against.

        Raises
        ------
        BuildError
            If the signature can be checked and is invalid.
        """

        # Analyze the signature of the callable (skip all checks if signature unavailable)
        try:
            signature = inspect.signature(callable)
        except (ValueError, TypeError):
            return

        # Count parameters, handling variadic arguments:
        # - **kwargs (VAR_KEYWORD): ignored, allows extra keyword arguments
        # - *args (VAR_POSITIONAL): enables range matching (num_params <= expected)
        params = signature.parameters.values()
        has_var_positional = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL for p in params
        )
        num_params = sum(
            p.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for p in params
        )

        dps = solution.spec.destination_passing_style
        expected_nparam = (
            len(definition.inputs) + len(definition.outputs)
            if dps
            else len(definition.inputs)
        )

        if has_var_positional:
            if num_params > expected_nparam:
                style = "Destination-passing" if dps else "Value-returning"
                raise BuildError(
                    f"{style} style callable: expected {expected_nparam} parameters, "
                    f"but got at least {num_params}"
                )
        elif num_params != expected_nparam:
            style = "Destination-passing" if dps else "Value-returning"
            raise BuildError(
                f"{style} style callable: expected {expected_nparam} parameters, but got "
                f"{num_params}"
            )

        # Check return annotation (only for value-returning style)
        if not solution.spec.destination_passing_style:
            num_outputs = len(definition.outputs)
            ret_ann = signature.return_annotation
            if ret_ann is not inspect.Signature.empty:
                origin = getattr(ret_ann, "__origin__", None)
                if origin is tuple:
                    # If returning tuple, length must match num_outputs
                    args = getattr(ret_ann, "__args__", None)
                    if args is not None and len(args) != num_outputs:
                        raise BuildError(
                            f"Value-returning style callable with {num_outputs} outputs must "
                            f"return a {num_outputs}-element tuple, got {ret_ann}"
                        )
                elif ret_ann is None:
                    # If return None, num_outputs must be 0
                    if num_outputs != 0:
                        raise BuildError(
                            f"Value-returning style callable with {num_outputs} outputs must "
                            f"not return None"
                        )
                elif num_outputs != 1:
                    # If returning non-tuple, num_outputs must be 1
                    raise BuildError(
                        f"Value-returning style callable returning non-tuple must "
                        f"only have one output, got {num_outputs}"
                    )
