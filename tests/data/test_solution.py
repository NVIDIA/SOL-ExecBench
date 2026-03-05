import sys

import pytest

from sol_execbench.data import (
    BuildSpec,
    CompileOptions,
    Solution,
    SourceFile,
    SupportedBindings,
    SupportedHardware,
    SupportedLanguages,
)


def test_sourcefile_validation():
    # Valid file
    SourceFile(path="main.py", content="def run():\n    pass\n")
    # Empty path
    with pytest.raises(ValueError):
        SourceFile(path="", content="def run(): pass")
    # Non-string content
    with pytest.raises(ValueError):
        SourceFile(path="main.py", content=123)  # type: ignore[arg-type]


def test_sourcefile_path_security():
    # Absolute path is rejected
    with pytest.raises(ValueError, match="absolute path not allowed"):
        SourceFile(path="/tmp/kernel.py", content="def run(): pass")
    # Path traversal is rejected
    with pytest.raises(ValueError, match="parent directory traversal not allowed"):
        SourceFile(path="../outside/kernel.py", content="def run(): pass")
    # Subdirectory paths are fine
    SourceFile(path="src/kernel.cu", content="__global__ void k() {}")


def test_buildspec_entry_point_validation():
    # Valid
    BuildSpec(
        language=SupportedLanguages.PYTHON,
        target_hardware=[SupportedHardware.B200],
        entry_point="main.py::run",
    )
    # Missing ::
    with pytest.raises(ValueError):
        BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=[SupportedHardware.B200],
            entry_point="main.py",
        )
    # Too many ::
    with pytest.raises(ValueError):
        BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=[SupportedHardware.B200],
            entry_point="main.py::run::extra",
        )


def test_buildspec_target_hardware_validation():
    # Empty list is rejected
    with pytest.raises(ValueError):
        BuildSpec(
            language=SupportedLanguages.TRITON,
            target_hardware=[],
            entry_point="main.py::run",
        )
    # Multiple targets are valid
    BuildSpec(
        language=SupportedLanguages.TRITON,
        target_hardware=[SupportedHardware.B200, SupportedHardware.LOCAL],
        entry_point="main.py::run",
    )


def test_buildspec_dependencies_validation():
    # Valid dependencies
    BuildSpec(
        language=SupportedLanguages.CUDA,
        target_hardware=[SupportedHardware.B200],
        entry_point="main.cpp::run",
        dependencies=["cublas", "cutlass"],
    )
    # Empty string dependency is rejected
    with pytest.raises(ValueError):
        BuildSpec(
            language=SupportedLanguages.CUDA,
            target_hardware=[SupportedHardware.B200],
            entry_point="main.cpp::run",
            dependencies=[""],
        )


def test_buildspec_optional_fields():
    spec = BuildSpec(
        language=SupportedLanguages.TRITON,
        target_hardware=[SupportedHardware.B200],
        entry_point="main.py::run",
    )
    assert spec.binding is None
    assert spec.compile_options is None
    assert spec.destination_passing_style is True  # default
    assert spec.dependencies == []


def test_compile_options():
    # Default is all-empty lists
    opts = CompileOptions()
    assert opts.cflags == []
    assert opts.cuda_cflags == []
    assert opts.ld_flags == []

    # Custom flags
    opts2 = CompileOptions(
        cflags=["-O3"],
        cuda_cflags=["--use_fast_math", "-arch=sm_90"],
        ld_flags=["-lcublas"],
    )
    assert opts2.cflags == ["-O3"]
    assert opts2.cuda_cflags == ["--use_fast_math", "-arch=sm_90"]
    assert opts2.ld_flags == ["-lcublas"]


def test_buildspec_with_compile_options():
    opts = CompileOptions(cuda_cflags=["--use_fast_math"])
    spec = BuildSpec(
        language=SupportedLanguages.CUDA,
        target_hardware=[SupportedHardware.B200],
        entry_point="main.cpp::run",
        binding=SupportedBindings.TORCH,
        compile_options=opts,
    )
    assert spec.compile_options.cuda_cflags == ["--use_fast_math"]


def test_solution_validation_and_helpers():
    spec = BuildSpec(
        language=SupportedLanguages.TRITON,
        target_hardware=[SupportedHardware.B200],
        entry_point="main.py::run",
    )
    s1 = SourceFile(path="main.py", content="def run():\n    pass\n")
    s2 = SourceFile(path="util.py", content="x = 1\n")
    sol = Solution(
        name="sol1", definition="def1", author="me", spec=spec, sources=[s1, s2]
    )

    assert sol.get_entry_source() is s1
    assert sol.get_entry_path().name == "main.py"
    assert sol.get_entry_symbol() == "run"


def test_solution_duplicate_sources_rejected():
    spec = BuildSpec(
        language=SupportedLanguages.TRITON,
        target_hardware=[SupportedHardware.B200],
        entry_point="main.py::run",
    )
    s = SourceFile(path="main.py", content="def run(): pass")
    with pytest.raises(ValueError, match="Duplicate source path"):
        Solution(name="dup", definition="def1", author="x", spec=spec, sources=[s, s])


def test_solution_missing_entry_source_rejected():
    spec = BuildSpec(
        language=SupportedLanguages.TRITON,
        target_hardware=[SupportedHardware.B200],
        entry_point="missing.py::run",
    )
    s = SourceFile(path="main.py", content="def run(): pass")
    with pytest.raises(ValueError, match="not found in sources"):
        Solution(name="missing", definition="def1", author="x", spec=spec, sources=[s])


def test_solution_requires_at_least_one_source():
    spec = BuildSpec(
        language=SupportedLanguages.PYTHON,
        target_hardware=[SupportedHardware.LOCAL],
        entry_point="main.py::run",
    )
    with pytest.raises(ValueError):
        Solution(name="empty", definition="def1", author="x", spec=spec, sources=[])


def test_solution_hash_deterministic():
    spec = BuildSpec(
        language=SupportedLanguages.PYTHON,
        target_hardware=[SupportedHardware.B200],
        entry_point="main.py::run",
    )
    s = SourceFile(path="main.py", content="def run(): pass")
    sol1 = Solution(name="s", definition="d", author="a", spec=spec, sources=[s])
    sol2 = Solution(name="s", definition="d", author="a", spec=spec, sources=[s])
    assert sol1.hash() == sol2.hash()
    assert len(sol1.hash()) == 40  # SHA1 hex digest


def test_solution_hash_changes_on_content_change():
    spec = BuildSpec(
        language=SupportedLanguages.PYTHON,
        target_hardware=[SupportedHardware.B200],
        entry_point="main.py::run",
    )
    s1 = SourceFile(path="main.py", content="def run(): return 1")
    s2 = SourceFile(path="main.py", content="def run(): return 2")
    sol1 = Solution(name="s", definition="d", author="a", spec=spec, sources=[s1])
    sol2 = Solution(name="s", definition="d", author="a", spec=spec, sources=[s2])
    assert sol1.hash() != sol2.hash()


def test_solution_is_frozen():
    spec = BuildSpec(
        language=SupportedLanguages.PYTHON,
        target_hardware=[SupportedHardware.B200],
        entry_point="main.py::run",
    )
    s = SourceFile(path="main.py", content="def run(): pass")
    sol = Solution(name="sol", definition="def1", author="me", spec=spec, sources=[s])
    with pytest.raises(Exception):
        sol.name = "other"  # type: ignore[misc]


def test_path_traversal_attack():
    spec = BuildSpec(
        language=SupportedLanguages.CUDA,
        target_hardware=[SupportedHardware.B200],
        entry_point="../../kernel.cpp::run",
    )
    with pytest.raises(ValueError, match="parent directory traversal not allowed"):
        Solution(
            name="malicious",
            definition="def1",
            author="attacker",
            spec=spec,
            sources=[SourceFile(path="../../kernel.cpp", content="int main() {}")],
        )


def test_absolute_path_attack():
    spec = BuildSpec(
        language=SupportedLanguages.CUDA,
        target_hardware=[SupportedHardware.B200],
        entry_point="/tmp/kernel.cpp::run",
    )
    with pytest.raises(ValueError, match="absolute path not allowed"):
        Solution(
            name="malicious",
            definition="def1",
            author="attacker",
            spec=spec,
            sources=[SourceFile(path="/tmp/kernel.cpp", content="int main() {}")],
        )


if __name__ == "__main__":
    pytest.main(sys.argv)
