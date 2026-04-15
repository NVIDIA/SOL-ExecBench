# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Strong-typed data definitions for solution implementations."""

import hashlib
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from .base_model import BaseModelWithDocstrings, NonEmptyString


class SupportedLanguages(str, Enum):
    """Supported programming languages for solution implementations.

    Enumeration of programming languages that can be used to implement
    solutions for computational workloads.
    """

    # Python languages
    PYTORCH = "pytorch"
    """PyTorch programming language. Specified if the solution uses PyTorch operators."""
    TRITON = "triton"
    """Triton GPU programming language."""
    CUTE_DSL = "cute_dsl"
    """NVIDIA CuTe DSL programming language."""
    CUTILE = "cutile"
    """NVIDIA cuTile programming language."""
    CUDNN_FRONTEND = "cudnn_frontend"
    """NVIDIA cuDNN frontend programming language."""
    # C++ languages
    CUTLASS = "cutlass"
    """Pure C++ source code."""
    CUDNN = "cudnn"
    """NVIDIA cuDNN programming language."""
    CUBLAS = "cublas"
    """NVIDIA cuBLAS programming language."""
    CUDA_CPP = "cuda_cpp"
    """CUDA C++ programming language with inline PTX support."""


class SupportedHardware(str, Enum):
    """Supported hardware targets for solution implementations.

    Enumeration of hardware platforms that solutions can target.
    """

    B200 = "B200"
    """NVIDIA B200."""
    LOCAL = "LOCAL"
    """Local NVIDIA GPU."""


class SupportedBindings(str, Enum):
    """Supported bindings for C++/CUDA solution implementations.

    Enumeration of binding types that can be used to interface compiled
    C++/CUDA code with Python.
    """

    TORCH = "torch"
    """PyTorch C++/CUDA extension binding."""


class SourceFile(BaseModelWithDocstrings):
    """A single source code file in a solution implementation.

    Represents a source code file with its relative path and complete content.
    The file content is validated for syntax correctness based on the file extension.
    """

    path: NonEmptyString
    """The relative path of the file, including its name and extension (e.g., 'src/kernel.cu',
    'main.py'). When compiling the solution, a temporary solution source directory will be
    created, and the file will be placed according to this path. The path should not contain
    parent directory traversal ("..")."""
    content: NonEmptyString
    """The complete text content of the source file."""

    @model_validator(mode="after")
    def _validate_source_path(self) -> "SourceFile":
        """Validate source path for security.

        Raises
        ------
        ValueError
            If the path contains security issues (absolute paths or path traversal).
        """
        src_path = Path(self.path)
        if src_path.is_absolute():
            raise ValueError(
                f"Invalid source path (absolute path not allowed): {self.path}"
            )
        if ".." in src_path.parts:
            raise ValueError(
                f"Invalid source path (parent directory traversal not allowed): {self.path}"
            )
        return self



class CompileOptions(BaseModelWithDocstrings):
    """Compiler and linker flags for C++/CUDA solutions.

    Passed directly to the underlying build system (e.g. torch.utils.cpp_extension.load).
    """

    cflags: list[str] = Field(default_factory=list)
    """Extra flags passed to the C++ compiler (gcc/g++)."""
    cuda_cflags: list[str] = Field(default_factory=lambda: ["-O3", "--use_fast_math"])
    """Extra flags passed to the CUDA compiler (nvcc). Defaults to -O3 and --use_fast_math."""
    ld_flags: list[str] = Field(default_factory=lambda: ["-lcuda"])
    """Extra flags passed to the linker. Defaults to -lcuda."""


class BuildSpec(BaseModelWithDocstrings):
    """Build specification for a solution implementation.

    Contains all technical specifications required to build and execute a solution, including
    language, hardware targets, dependencies, entry point, and build commands.
    """

    languages: list[SupportedLanguages]
    """The list of programming languages used to implement the solution. C++ languages and Python languages cannot be mixed."""
    target_hardware: list[SupportedHardware] = Field(min_length=1)
    """List of hardware this solution is compatible with (e.g., B200, LOCAL)."""
    entry_point: NonEmptyString
    """The exact path to the function to be called. Format: '{file_path}::{function_name}'
    (e.g., 'main.py::run')."""
    dependencies: list[NonEmptyString] = Field(default_factory=list)
    """Optional list of required libraries or packages. E.g. for CUDA, we support 'cublas',
    'cudnn', 'cutlass'"""
    destination_passing_style: bool = True
    """Whether to use destination passing style for the solution. If True, the solution should
    accept the output tensors as the last arguments. If False, the solution should return the
    output tensors."""
    binding: Optional[SupportedBindings] = None
    """The binding type to use for C++/CUDA solutions. If None, defaults to 'torch' for
    C++/CUDA languages. Ignored for Python and Triton languages."""
    compile_options: Optional[CompileOptions] = None
    """Optional compiler and linker flags. Only used for C++/CUDA solutions with torch binding."""

    @model_validator(mode="after")
    def _validate_entry_point(self) -> "BuildSpec":
        """Validate entry_point format.

        Raises
        ------
        ValueError
            If entry_point doesn't follow the required format.
        """
        if self.entry_point.count("::") != 1:
            raise ValueError(
                f"Invalid entry point format: {self.entry_point}. Expected "
                '"<file_path>::<function_name>".'
            )
        return self

    @model_validator(mode="after")
    def _validate_languages(self) -> "BuildSpec":
        """Validate languages support matrix.

        Raises
        ------
        ValueError
            If the languages are not valid.
        """

        python_languages = [
            SupportedLanguages.PYTORCH,
            SupportedLanguages.TRITON,
            SupportedLanguages.CUTE_DSL,
            SupportedLanguages.CUTILE,
            SupportedLanguages.CUDNN_FRONTEND,
        ]
        cpp_languages = [
            SupportedLanguages.CUTLASS,
            SupportedLanguages.CUDNN,
            SupportedLanguages.CUBLAS,
            SupportedLanguages.CUDA_CPP,
        ]

        included_python_langs = [
            language for language in self.languages if language in python_languages
        ]
        included_cpp_langs = [
            language for language in self.languages if language in cpp_languages
        ]
        if len(included_cpp_langs) and len(included_python_langs):
            raise ValueError(
                f"C++ and Python cannot be mixed, but got {included_cpp_langs} and {included_python_langs}"
            )

        # Validate entry point file suffix matches the language category.
        entry_file = self.entry_point.split("::")[0]
        suffix = Path(entry_file).suffix
        if included_cpp_langs and suffix not in (
            ".cu",
            ".cpp",
            ".cc",
            ".cxx",
            ".c",
            ".h",
            ".hpp",
            ".cuh",
        ):
            raise ValueError(
                f"C++ languages require a C++/CUDA entry point file, "
                f"but got '{entry_file}' (suffix '{suffix}')"
            )
        if included_python_langs and suffix != ".py":
            raise ValueError(
                f"Python languages require a .py entry point file, "
                f"but got '{entry_file}' (suffix '{suffix}')"
            )
        return self


class Solution(BaseModelWithDocstrings):
    """A concrete implementation for a given Definition.

    Represents a complete solution that provides a high-performance implementation
    for a computational workload defined by a Definition. Contains all source code,
    build specifications, and metadata required for building, interfacing, and
    benchmarking the implementation.
    """

    model_config = ConfigDict(use_attribute_docstrings=True, frozen=True)
    """Treat Solution as immutable to safely memoize derived fields."""

    _hash_cache: str = PrivateAttr()
    """Memoized hash of the solution content."""

    name: NonEmptyString
    """A unique, human-readable name for this specific solution (e.g., 'rmsnorm_triton_v1_h100')."""
    definition: NonEmptyString
    """The name of the Definition this implementation solves."""
    author: NonEmptyString
    """The name of the author or agent system that created this solution."""
    spec: BuildSpec
    """Technical specifications for building and executing this solution."""
    sources: list[SourceFile] = Field(min_length=1)
    """Array of source code files representing the complete implementation."""
    description: Optional[str] = Field(default=None)
    """Optional human-readable description of the solution's technique or approach."""

    @model_validator(mode="after")
    def _validate_source_path_entry_point(self) -> "Solution":
        """Validate source file paths for uniqueness and entry file existence.

        Raises
        ------
        ValueError
            If duplicate source file paths are found or the entry file is not found in the sources.
        """
        seen_paths = set()
        for source in self.sources:
            # Check for duplicates
            if source.path in seen_paths:
                raise ValueError(f"Duplicate source path '{source.path}'")
            seen_paths.add(source.path)

        entry_file = self.spec.entry_point.split("::")[0]

        if entry_file not in seen_paths:
            raise ValueError(f"Entry source file '{entry_file}' not found in sources")

        return self

    def get_entry_path(self) -> Path:
        """Extract the file path from the entry point specification.

        The entry point format is '{file_path}::{function_name}', and this method
        returns the file path component as a Path object.

        Returns
        -------
        Path
            The relative path to the entry source file (e.g., 'main.py', 'src/kernel.cu').
        """
        return Path(self.spec.entry_point.split("::")[0])

    def get_entry_symbol(self) -> str:
        """Extract the function/symbol name from the entry point specification.

        The entry point format is '{file_path}::{function_name}', and this method
        returns the function name component. This is the symbol that builders will
        look up in the compiled module or imported Python module.

        Returns
        -------
        str
            The function or symbol name to be loaded (e.g., 'run', 'forward', 'kernel').
        """
        return self.spec.entry_point.split("::")[-1]

    def get_entry_source(self) -> Optional[SourceFile]:
        """Get the entry source file specified in the build spec.

        Returns
        -------
        Optional[SourceFile]
            The SourceFile object containing the entry point, or None if not found.
        """
        entry_path = self.spec.entry_point.split("::")[0]
        for source in self.sources:
            if source.path == entry_path:
                return source
        return None

    def model_post_init(self, __context: Any) -> None:
        # Precompute hash once since the model is frozen/immutable.
        object.__setattr__(self, "_hash_cache", self._compute_hash())

    def _compute_hash(self) -> str:
        """Compute a deterministic hash of the solution content."""
        h = hashlib.sha1()
        for s in (
            self.name,
            self.definition,
            *[lang.value for lang in self.spec.languages],
            self.spec.entry_point,
            self.spec.binding.value if self.spec.binding else "",
            *self.spec.dependencies,
            *(part for src in self.sources for part in (src.path, src.content)),
        ):
            h.update(s.encode())

        return h.hexdigest()

    def hash(self) -> str:
        """Return the memoized deterministic hash of the solution content.

        This hash is computed from all fields that affect the solution's behavior:
        name, definition, language, entry point, dependencies, and all source file
        paths and contents. This ensures that any meaningful change to the solution
        results in a different hash.

        The hash is used for caching build artifacts, allowing solutions with the same
        hash to reuse the same cached build result.

        Returns
        -------
        str
            A SHA1 hash (40 hex characters) uniquely identifying this solution's content.
        """
        return self._hash_cache

    def __hash__(self) -> int:  # pragma: no cover - trivial wrapper
        # Use the memoized content hash for fast hashing in dict/set keys.
        return hash(self._hash_cache)

    def __eq__(self, other: object) -> bool:  # pragma: no cover - trivial wrapper
        if not isinstance(other, Solution):
            return NotImplemented
        return self._hash_cache == other._hash_cache
