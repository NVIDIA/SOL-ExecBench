"""Solution file parser for creating Solution objects from source files."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Optional

from .data import (
    BuildSpec,
    Solution,
    SourceFile,
    SupportedBindings,
    SupportedHardware,
    SupportedLanguages,
)

# Valid source file extensions for filtering
VALID_SOURCE_EXTENSIONS = {
    ".py",
    ".cu",
    ".cuh",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".cc",
    ".cxx",
}

# Extension to language mappings
CUDA_EXTENSIONS = {".cu", ".cuh"}
CPP_EXTENSIONS = {".cpp", ".cc", ".cxx", ".c", ".h", ".hpp"}
PYTHON_EXTENSIONS = {".py"}


class SolutionParseError(Exception):
    """Exception raised when solution parsing fails."""

    pass


def detect_language_from_file(file_path: Path) -> SupportedLanguages:
    """Detect programming language from file extension and content.

    Parameters
    ----------
    file_path : Path
        Path to the source file.

    Returns
    -------
    SupportedLanguages
        Detected language enum value.

    Raises
    ------
    SolutionParseError
        If the file extension is not supported or language cannot be detected.

    Examples
    --------
    >>> detect_language_from_file(Path("kernel.cu"))
    <SupportedLanguages.CUDA: 'cuda'>
    >>> detect_language_from_file(Path("kernel.py"))  # Contains @triton.jit
    <SupportedLanguages.TRITON: 'triton'>
    """
    ext = file_path.suffix.lower()

    # CUDA files
    if ext in CUDA_EXTENSIONS:
        return SupportedLanguages.CUDA

    # C++ files
    if ext in CPP_EXTENSIONS:
        return SupportedLanguages.CPP

    # Python files (check for Triton decorator)
    if ext in PYTHON_EXTENSIONS:
        try:
            content = file_path.read_text(encoding="utf-8")

            # Check for Triton decorator
            if "@triton.jit" in content or "import triton" in content:
                return SupportedLanguages.TRITON

            return SupportedLanguages.PYTHON

        except Exception as e:
            raise SolutionParseError(f"Failed to read file {file_path}: {e}") from e

    # Unsupported extension
    raise SolutionParseError(
        f"Unsupported file extension: {ext}. "
        f"Supported extensions: {', '.join(sorted(VALID_SOURCE_EXTENSIONS))}"
    )


def extract_entry_point_from_python(file_path: Path, content: str) -> str:
    """Extract entry point function from Python/Triton code.

    Looks for the first public function (not starting with _) or a function
    with specific names like "run", "forward", "kernel", etc.

    Parameters
    ----------
    file_path : Path
        Path to the source file.
    content : str
        File content.

    Returns
    -------
    str
        Entry point in format "filename.py::function_name".

    Raises
    ------
    SolutionParseError
        If no suitable entry point function is found.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        raise SolutionParseError(f"Failed to parse Python file {file_path}: {e}") from e

    # Priority function names
    priority_names = {"run", "forward", "kernel", "main", "execute"}

    # Find all function definitions
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Skip private functions
            if not node.name.startswith("_"):
                functions.append(node.name)

                # Return immediately if priority name found
                if node.name in priority_names:
                    return f"{file_path.name}::{node.name}"

    # Return first public function if no priority name found
    if functions:
        return f"{file_path.name}::{functions[0]}"

    raise SolutionParseError(f"No public function found in {file_path}")


def extract_entry_point_from_cuda(file_path: Path, content: str) -> str:
    """Extract entry point function from CUDA/C++ code.

    Looks for functions with specific patterns like TORCH_EXTENSION_NAME,
    or common kernel launch function names.

    Parameters
    ----------
    file_path : Path
        Path to the source file.
    content : str
        File content.

    Returns
    -------
    str
        Entry point in format "filename.cu::function_name".

    Raises
    ------
    SolutionParseError
        If no suitable entry point function is found.
    """
    # Look for common patterns
    patterns = [
        # PyTorch extension binding
        r"TORCH_LIBRARY\([^,]+,\s*m\)\s*\{\s*m\.def\(['\"](\w+)['\"]",
        # Regular C++ function definitions
        r"(?:void|int|float|double|torch::Tensor)\s+(\w+)\s*\(",
    ]

    # Priority function names
    priority_names = {"forward", "kernel", "run", "execute", "launch"}

    functions = []
    for pattern in patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            func_name = match.group(1)
            functions.append(func_name)

            # Return immediately if priority name found
            if func_name in priority_names:
                return f"{file_path.name}::{func_name}"

    # Return first function if no priority name found
    if functions:
        return f"{file_path.name}::{functions[0]}"

    # Fallback: use filename without extension
    fallback_name = file_path.stem
    return f"{file_path.name}::{fallback_name}"


def extract_dependencies_from_python(content: str) -> list[str]:
    """Extract dependencies from Python imports.

    Parameters
    ----------
    content : str
        Python source code.

    Returns
    -------
    list[str]
        List of dependency names (e.g., ["torch", "triton"]).
    """
    dependencies = []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return dependencies

    # Extract import statements
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Get top-level package name
                pkg = alias.name.split(".")[0]
                if pkg not in dependencies and pkg not in {
                    "os",
                    "sys",
                    "math",
                    "typing",
                }:
                    dependencies.append(pkg)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                # Get top-level package name
                pkg = node.module.split(".")[0]
                if pkg not in dependencies and pkg not in {
                    "os",
                    "sys",
                    "math",
                    "typing",
                }:
                    dependencies.append(pkg)

    return sorted(dependencies)


def extract_dependencies_from_cuda(content: str) -> list[str]:
    """Extract dependencies from CUDA/C++ includes.

    Parameters
    ----------
    content : str
        CUDA/C++ source code.

    Returns
    -------
    list[str]
        List of dependency names (e.g., ["cublas", "cudnn"]).
    """
    dependencies = []

    # Pattern for #include statements
    include_pattern = r'#include\s+[<"]([^>"]+)[>"]'

    matches = re.finditer(include_pattern, content)
    for match in matches:
        include = match.group(1)

        # Extract library name from include path
        # e.g., "cublas_v2.h" -> "cublas", "torch/extension.h" -> "torch"
        if "/" in include:
            lib = include.split("/")[0]
        else:
            lib = include.split(".")[0]

        # Common CUDA/C++ library mappings
        lib_mapping = {
            "cublas": "cublas",
            "cudnn": "cudnn",
            "cufft": "cufft",
            "curand": "curand",
            "cusparse": "cusparse",
            "torch": "torch",
            "ATen": "torch",
        }

        for pattern, dep in lib_mapping.items():
            if pattern.lower() in lib.lower():
                if dep not in dependencies:
                    dependencies.append(dep)
                break

    return sorted(dependencies)


def parse_solution_from_file(
    file_path: Path,
    definition_name: str,
    author: str = "user",
    solution_name: Optional[str] = None,
    target_hardware: Optional[list[SupportedHardware]] = None,
) -> Solution:
    """Parse a single solution file and create a Solution object.

    Parameters
    ----------
    file_path : Path
        Path to the solution source file (.py, .cu, .cpp, etc.).
    definition_name : str
        Name of the problem definition this solution solves.
    author : str, optional
        Author name. Defaults to "user".
    solution_name : str, optional
        Solution name. If None, uses filename stem.
    target_hardware : list[SupportedHardware], optional
        Target hardware list. If None, defaults to [SupportedHardware.B200].

    Returns
    -------
    Solution
        Parsed solution object ready for evaluation.

    Raises
    ------
    SolutionParseError
        If the file cannot be parsed or required information cannot be extracted.

    Examples
    --------
    >>> solution = parse_solution_from_file(
    ...     Path("my_kernel.py"),
    ...     definition_name="rmsnorm_h4096",
    ...     author="alice",
    ... )
    >>> print(solution.name)
    'my_kernel'
    >>> print(solution.spec.language)
    <SupportedLanguages.PYTHON: 'python'>
    """
    if not file_path.exists():
        raise SolutionParseError(f"Solution file not found: {file_path}")

    if not file_path.is_file():
        raise SolutionParseError(f"Path is not a file: {file_path}")

    # Detect language
    language = detect_language_from_file(file_path)

    # Read file content
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        raise SolutionParseError(f"Failed to read file {file_path}: {e}") from e

    # Extract entry point
    if language in (SupportedLanguages.PYTHON, SupportedLanguages.TRITON):
        entry_point = extract_entry_point_from_python(file_path, content)
    else:
        entry_point = extract_entry_point_from_cuda(file_path, content)

    # Extract dependencies
    if language in (SupportedLanguages.PYTHON, SupportedLanguages.TRITON):
        dependencies = extract_dependencies_from_python(content)
    else:
        dependencies = extract_dependencies_from_cuda(content)

    # Determine binding type
    binding: Optional[SupportedBindings] = None
    if language in (SupportedLanguages.CUDA, SupportedLanguages.CPP):
        # Check for torch extension patterns
        if (
            "torch" in dependencies
            or "TORCH_LIBRARY" in content
            or "torch::Tensor" in content
        ):
            binding = SupportedBindings.TORCH

    # Create build spec
    spec = BuildSpec(
        language=language,
        target_hardware=target_hardware or [SupportedHardware.B200],
        entry_point=entry_point,
        dependencies=dependencies,
        binding=binding,
    )

    # Create source file
    source = SourceFile(path=file_path.name, content=content)

    # Create solution
    solution = Solution(
        name=solution_name or file_path.stem,
        definition=definition_name,
        author=author,
        spec=spec,
        sources=[source],
    )

    return solution


def parse_solution_from_directory(
    directory_path: Path,
    definition_name: str,
    author: str = "user",
    solution_name: Optional[str] = None,
    target_hardware: Optional[list[SupportedHardware]] = None,
    entry_point: Optional[str] = None,
) -> Solution:
    """Parse a directory of source files and create a Solution object.

    This is useful for multi-file solutions (e.g., .cu + .h files).

    Parameters
    ----------
    directory_path : Path
        Path to the directory containing solution source files.
    definition_name : str
        Name of the problem definition this solution solves.
    author : str, optional
        Author name. Defaults to "user".
    solution_name : str, optional
        Solution name. If None, uses directory name.
    target_hardware : list[SupportedHardware], optional
        Target hardware list. If None, defaults to [SupportedHardware.B200].
    entry_point : str, optional
        Override entry point in format "file::function". If None, auto-detects.

    Returns
    -------
    Solution
        Parsed solution object ready for evaluation.

    Raises
    ------
    SolutionParseError
        If the directory cannot be parsed or no valid source files are found.

    Examples
    --------
    >>> solution = parse_solution_from_directory(
    ...     Path("my_solution/"),
    ...     definition_name="rmsnorm_h4096",
    ...     author="alice",
    ... )
    """
    if not directory_path.exists():
        raise SolutionParseError(f"Solution directory not found: {directory_path}")

    if not directory_path.is_dir():
        raise SolutionParseError(f"Path is not a directory: {directory_path}")

    # Collect source files
    source_files = []
    main_file: Optional[Path] = None
    detected_language: Optional[SupportedLanguages] = None

    for file_path in sorted(directory_path.iterdir()):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        if ext not in VALID_SOURCE_EXTENSIONS:
            continue

        # Read content
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            raise SolutionParseError(f"Failed to read file {file_path}: {e}") from e

        # Create source file with relative path
        source = SourceFile(path=file_path.name, content=content)
        source_files.append(source)

        # Detect language from first file
        if detected_language is None:
            detected_language = detect_language_from_file(file_path)
            main_file = file_path

    if not source_files:
        raise SolutionParseError(f"No valid source files found in {directory_path}")

    if detected_language is None or main_file is None:
        raise SolutionParseError(
            f"Could not detect language from files in {directory_path}"
        )

    # Extract entry point if not provided
    if entry_point is None:
        content = main_file.read_text(encoding="utf-8")
        if detected_language in (SupportedLanguages.PYTHON, SupportedLanguages.TRITON):
            entry_point = extract_entry_point_from_python(main_file, content)
        else:
            entry_point = extract_entry_point_from_cuda(main_file, content)

    # Extract dependencies from main file
    main_content = main_file.read_text(encoding="utf-8")
    if detected_language in (SupportedLanguages.PYTHON, SupportedLanguages.TRITON):
        dependencies = extract_dependencies_from_python(main_content)
    else:
        dependencies = extract_dependencies_from_cuda(main_content)

    # Determine binding type
    binding: Optional[SupportedBindings] = None
    if detected_language in (SupportedLanguages.CUDA, SupportedLanguages.CPP):
        if "torch" in dependencies or any("torch" in s.content for s in source_files):
            binding = SupportedBindings.TORCH

    # Create build spec
    spec = BuildSpec(
        language=detected_language,
        target_hardware=target_hardware or [SupportedHardware.B200],
        entry_point=entry_point,
        dependencies=dependencies,
        binding=binding,
    )

    # Create solution
    solution = Solution(
        name=solution_name or directory_path.name,
        definition=definition_name,
        author=author,
        spec=spec,
        sources=source_files,
    )

    return solution


def save_solution_to_json(solution: Solution, output_path: Path) -> None:
    """Save a Solution object to a JSON file.

    This is useful for converting parsed solutions to the JSON format
    expected by the evaluate command.

    Parameters
    ----------
    solution : Solution
        Solution object to save.
    output_path : Path
        Output path for the JSON file.

    Examples
    --------
    >>> # Parse a solution from source
    >>> solution = parse_solution_from_file(Path("kernel.py"), "rmsnorm_h4096")
    >>> # Save to JSON for future use
    >>> save_solution_to_json(solution, Path("my_solution.json"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(solution.model_dump_json(indent=2, exclude_unset=True))


__all__ = [
    "SolutionParseError",
    "detect_language_from_file",
    "extract_entry_point_from_python",
    "extract_entry_point_from_cuda",
    "extract_dependencies_from_python",
    "extract_dependencies_from_cuda",
    "parse_solution_from_file",
    "parse_solution_from_directory",
    "save_solution_to_json",
]
