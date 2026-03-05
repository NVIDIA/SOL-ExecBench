"""Builder for CUDA solutions using PyTorch's C++/CUDA extension system."""

from __future__ import annotations

import logging
import shutil
import sys
from importlib import resources
from pathlib import Path
from typing import Callable, ClassVar, Optional

from ..builder import Builder, BuildError
from ..runnable import Runnable, RunnableMetadata
from ..utils import write_sources_to_path
from ....data import Definition, Solution, SupportedBindings, SupportedLanguages

logger = logging.getLogger(__name__)


class TorchBuilder(Builder):
    """Builder for CUDA solutions using PyTorch's C++/CUDA extension loader.

    This builder compiles C++/CUDA source files into a Python extension module using
    torch.utils.cpp_extension.load(). It supports common CUDA dependencies like cuBLAS,
    cuDNN, and CUTLASS.
    """

    _PACKAGE_PREFIX: ClassVar[str] = "torch_"
    """A unique prefix to prepend to the package name."""

    _BUILD_DIR_NAME: ClassVar[str] = "torch"
    """A unique subdirectory under SOLBENCH_CACHE_PATH where build results are stored."""

    _CPP_CUDA_EXTENSIONS: ClassVar[list[str]] = [".cu", ".cpp", ".cc", ".cxx", ".c"]
    """C/C++ and CUDA source file extensions."""

    def __init__(self) -> None:
        """Initialize the TorchBuilder and discover available CUDA dependencies."""
        super().__init__(self._PACKAGE_PREFIX, self._BUILD_DIR_NAME)

    @staticmethod
    def is_available() -> bool:
        """Check if CUDA is available in the current environment.

        Returns
        -------
        bool
            True if PyTorch is installed and CUDA is available, False otherwise.
        """
        try:
            import torch
        except ImportError:
            return False
        return torch.cuda.is_available()

    def can_build(self, solution: Solution) -> bool:
        """Check if this builder can handle the given solution. The solution should be CUDA or
        C++ source code with torch binding.

        Parameters
        ----------
        solution : Solution
            Solution to check

        Returns
        -------
        bool
            True if solution language is CUDA or C++ and binding is torch
        """
        is_cpp_or_cuda = (
            solution.spec.language == SupportedLanguages.CUDA
            or solution.spec.language == SupportedLanguages.CPP
        )
        return is_cpp_or_cuda and solution.spec.binding == SupportedBindings.TORCH

    def _filter_sources(self, source_paths: list[Path]) -> list[str]:
        """Filter source files to include only C/C++/CUDA files.

        Parameters
        ----------
        source_paths : list[Path]
            List of all source file paths.

        Returns
        -------
        list[str]
            List of source file paths with C/C++/CUDA extensions.
        """
        return [
            str(path)
            for path in source_paths
            if path.suffix in self._CPP_CUDA_EXTENSIONS
        ]

    def _get_cleaner(self, build_dir: Path) -> Callable[[], None]:
        """Create a cleaner function that removes the build directory.

        Parameters
        ----------
        build_dir : Path
            The directory to delete.

        Returns
        -------
        Callable[[], None]
            A function that performs the cleanup.
        """

        def cleaner() -> None:
            shutil.rmtree(build_dir, ignore_errors=True)

        return cleaner

    def build(self, definition: Definition, solution: Solution) -> Runnable:
        """Build a CUDA solution into a runnable.

        This method writes the solution sources to a build directory, compiles them
        using PyTorch's cpp_extension.load(), and returns a callable wrapper.

        Parameters
        ----------
        definition : Definition
            The problem definition.
        solution : Solution
            The CUDA solution to build.

        Returns
        -------
        Runnable
            An executable wrapper around the compiled extension.

        Raises
        ------
        BuildError
            If the entry file is not a C/C++/CUDA file, compilation fails, or the
            entry symbol is not found in the compiled extension.
        """
        from torch.utils.cpp_extension import load

        entry_file_extension = solution.get_entry_path().suffix
        if entry_file_extension not in self._CPP_CUDA_EXTENSIONS:
            raise BuildError(
                f"Entry file type not recognized. Must be one of {self._CPP_CUDA_EXTENSIONS}, "
                f"got {entry_file_extension}."
            )

        symbol = solution.get_entry_symbol()
        package_name, build_dir = self._get_package_name_and_build_path(solution)
        source_paths = write_sources_to_path(build_dir, solution.sources)

        cpp_cuda_paths = self._filter_sources(source_paths)

        compile_opts = solution.spec.compile_options
        extra_include_paths = [str(build_dir)]
        extra_cflags = compile_opts.cflags if compile_opts else []
        extra_cuda_cflags = compile_opts.cuda_cflags if compile_opts else []
        extra_ldflags = compile_opts.ld_flags if compile_opts else []
        with_cuda = solution.spec.language == SupportedLanguages.CUDA

        try:
            ext = load(
                name=package_name,
                sources=cpp_cuda_paths,
                extra_include_paths=extra_include_paths,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_ldflags=extra_ldflags,
                with_cuda=with_cuda,
                build_directory=build_dir,
                verbose=True,
            )
        except Exception as e:
            raise BuildError(
                f"Torch build failed for solution '{solution.name}': {e}"
            ) from e

        try:
            callable = getattr(ext, symbol)
        except AttributeError as e:
            raise BuildError(
                f"Exported symbol '{symbol}' not found in built extension"
            ) from e

        metadata = RunnableMetadata(
            build_type="torch",
            definition_name=definition.name,
            solution_name=solution.name,
            destination_passing_style=solution.spec.destination_passing_style,
            definition=definition,
            misc={"entry_symbol": symbol, "binary": getattr(ext, "__file__", None)},
        )

        cleaner = self._get_cleaner(build_dir)

        self._try_validate_signature(callable, definition, solution)

        return Runnable(callable=callable, metadata=metadata, cleaner=cleaner)


class DependencyManager:
    """Dependency manager for CUDA dependencies. Not used for now."""

    _CUDA_DEPS: ClassVar[dict[str, tuple[str, Optional[list[str]]]]] = {
        "cublas": ("nvidia.cublas", ["cublas", "cublasLt"]),
        "cudnn": ("nvidia.cudnn", ["cudnn"]),
        "cutlass": (
            "flashinfer_bench.third_party.cutlass",
            None,
        ),  # Header-only dependency
    }
    """CUDA dependencies and their package names and library names"""

    _extra_include_paths: dict[str, str]
    """Extra include paths for CUDA dependencies"""

    _extra_ldflags: dict[str, list[str]]
    """Extra link flags for CUDA dependencies"""

    def __init__(self):
        """Initialize the dependency manager."""
        self._discover_cuda_deps()

    def _discover_cuda_deps(self) -> None:
        """Discover available CUDA dependencies and their include/library paths.

        This method populates _extra_include_paths and _extra_ldflags with paths
        for dependencies like cuBLAS, cuDNN, and CUTLASS if they are installed.
        """
        self._extra_include_paths = {}
        self._extra_ldflags = {}
        for dep_name, (pkg_name, libs) in self._CUDA_DEPS.items():
            include_path, ldflags = self._get_package_paths(pkg_name, libs)
            if include_path:
                self._extra_include_paths[dep_name] = include_path
            if ldflags:
                self._extra_ldflags[dep_name] = ldflags

    def _get_package_paths(
        self, pkg_name: str, lib_names: Optional[list[str]] = None
    ) -> tuple[Optional[str], list[str]]:
        """Discover include and library paths for a given package.

        This function searches for the include and lib directories in a Python package
        and returns the paths and linker flags needed to use them.

        Parameters
        ----------
        pkg_name : str
            The Python package name to search for (e.g., 'nvidia.cublas').
        lib_names : Optional[list[str]]
            List of library names to link against (e.g., ['cublas', 'cublasLt']).
            If None, only include paths are returned.

        Returns
        -------
        tuple[Optional[str], list[str]]
            A tuple of (include_path, ldflags) where include_path is the path to the
            include directory (or None if not found) and ldflags is a list of linker
            flags to use the libraries.
        """
        include_path = None
        ldflags = []

        try:
            include_dir = resources.files(pkg_name) / "include"
            if include_dir.exists():
                include_path = str(include_dir)

            if lib_names:
                lib_dir = resources.files(pkg_name) / "lib"
                if lib_dir.exists():
                    lib_path = Path(lib_dir)

                    if sys.platform.startswith("linux"):
                        ldflags = [f"-L{lib_path}", f"-Wl,-rpath,{lib_path}"]

                        for lib_name in lib_names:
                            # Look for unversioned .so first
                            lib_file = lib_path / f"lib{lib_name}.so"
                            if lib_file.exists():
                                ldflags.append(f"-l{lib_name}")
                            else:
                                # Find versioned .so files
                                versioned = sorted(lib_path.glob(f"lib{lib_name}.so.*"))
                                if versioned:
                                    ldflags.append(f"-l:{versioned[-1].name}")
                                else:
                                    ldflags.append(f"-l{lib_name}")  # Fallback

                    elif sys.platform == "win32":
                        ldflags = [f"/LIBPATH:{lib_path}"] + lib_names

        except Exception:
            logger.warning(
                "Failed to discover resources for CUDA package '%s'; continuing without it.",
                pkg_name,
                exc_info=True,
            )

        return include_path, ldflags

    def get_dependency_flags(self, solution: Solution) -> tuple[list[str], list[str]]:
        """Extract include paths and linker flags for solution dependencies.

        Parameters
        ----------
        solution : Solution
            The solution whose dependencies to process.

        Returns
        -------
        tuple[list[str], list[str]]
            A tuple of (include_paths, ldflags) containing the include directories
            and linker flags needed for the solution's dependencies.

        Raises
        ------
        BuildError
            If a required dependency is not available in the environment.
        """
        extra_include_paths = []
        extra_ldflags = []

        for dep in solution.spec.dependencies or []:
            if dep not in self._CUDA_DEPS.keys():
                logger.warning(f"Dependency '{dep}' not found in CUDA_DEPS")
                continue
            inc_path = self._extra_include_paths.get(dep)
            if not inc_path:
                raise BuildError(
                    f"{dep} is not available in the current environment but referenced "
                    f"by {solution.name}"
                )
            extra_include_paths.append(inc_path)
            ldflags = self._extra_ldflags.get(dep)
            if ldflags:
                extra_ldflags.extend(ldflags)

        return extra_include_paths, extra_ldflags
