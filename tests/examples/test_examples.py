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

"""End-to-end tests for the examples/ directory.

Each test loads an example problem (definition.json + workload.jsonl +
solution_*.json), packages it via ProblemPackager, runs the compile and
execute phases locally via subprocess, and asserts that all workloads pass.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from sol_execbench.core import (
    BenchmarkConfig,
    Definition,
    Solution,
    Workload,
)
from sol_execbench.driver.problem_packager import ProblemPackager

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "examples"

_CPP_LANGUAGES = {"cuda_cpp", "cutlass", "cudnn", "cublas"}


# ---------------------------------------------------------------------------
# Test case descriptors
# ---------------------------------------------------------------------------


@dataclass
class Example:
    """Parameters for one example e2e test case."""

    test_id: str
    language: str
    problem: str
    solution_file: str
    expected_count: int
    extra_markers: list[str] = field(default_factory=list)


_EXAMPLES = [
    Example(
        test_id="gemma3_swiglu_python",
        language="pytorch",
        problem="gemma3_swiglu",
        solution_file="solution_python.json",
        expected_count=3,
    ),
    Example(
        test_id="linear_backward_python",
        language="pytorch",
        problem="linear_backward",
        solution_file="solution_python.json",
        expected_count=3,
    ),
    Example(
        test_id="nemotron_rms_norm_triton",
        language="triton",
        problem="nemotron_rms_norm",
        solution_file="solution_triton.json",
        expected_count=3,
    ),
    Example(
        test_id="olmo3_post_norm_triton",
        language="triton",
        problem="olmo3_post_norm",
        solution_file="solution_triton.json",
        expected_count=3,
    ),
    Example(
        test_id="rmsnorm_triton",
        language="triton",
        problem="rmsnorm",
        solution_file="solution_triton.json",
        expected_count=14,
    ),
    Example(
        test_id="rmsnorm_cuda",
        language="cuda_cpp",
        problem="rmsnorm",
        solution_file="solution_cuda.json",
        expected_count=14,
        extra_markers=["cpp"],
    ),
    Example(
        test_id="flux_rope_cuda",
        language="cuda_cpp",
        problem="flux_rope",
        solution_file="solution_cuda.json",
        expected_count=3,
        extra_markers=["cpp"],
    ),
    Example(
        test_id="jamba_attn_proj_cute_dsl",
        language="cute_dsl",
        problem="jamba_attn_proj",
        solution_file="solution_cute_dsl.json",
        expected_count=3,
        extra_markers=["requires_cute_dsl"],
    ),
    Example(
        test_id="jamba_attn_proj_cutile",
        language="cutile",
        problem="jamba_attn_proj",
        solution_file="solution_cutile.json",
        expected_count=3,
        extra_markers=["requires_cutile"],
    ),
    Example(
        test_id="gemm_cutlass",
        language="cutlass",
        problem="gemm",
        solution_file="solution_cutlass.json",
        expected_count=3,
        extra_markers=["cpp"],
    ),
    Example(
        test_id="softmax_cudnn",
        language="cudnn",
        problem="softmax",
        solution_file="solution_cudnn.json",
        expected_count=3,
        extra_markers=["cpp"],
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_example(
    language: str, problem: str, solution_file: str
) -> tuple[Definition, Solution, list[Workload]]:
    """Load definition, solution, and workloads from an example directory."""
    example_dir = _EXAMPLES_DIR / language / problem
    definition = Definition(**json.loads((example_dir / "definition.json").read_text()))
    sol_dict = json.loads((example_dir / solution_file).read_text())
    solution = Solution(**sol_dict)
    workloads = [
        Workload(**json.loads(line))
        for line in (example_dir / "workload.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return definition, solution, workloads


def _run_subprocess(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run a command in the given directory and return the result."""
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _mark_example(case: Example) -> list:
    marks = [pytest.mark.xdist_group("serial")]
    for m in case.extra_markers:
        marks.append(getattr(pytest.mark, m))
    return marks


# ---------------------------------------------------------------------------
# Parametrized example test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c.test_id, marks=_mark_example(c)) for c in _EXAMPLES],
)
def test_example(tmp_path: Path, case: Example):
    """End-to-end evaluation: load example -> package -> compile (if C++) -> execute -> assert PASSED."""
    definition, solution, workloads = _load_example(
        case.language, case.problem, case.solution_file
    )
    config = BenchmarkConfig(lock_clocks=False)

    pkg = ProblemPackager(
        definition=definition,
        workloads=workloads,
        solution=solution,
        config=config,
        output_dir=tmp_path / "staging",
        keep_output_dir=True,
    )

    # Phase 1 (C++ only): compile
    languages = {lang.value for lang in solution.spec.languages}
    if languages & _CPP_LANGUAGES:
        cmd, artifact_path = pkg.compile()
        result = _run_subprocess(cmd, cwd=pkg.output_dir)
        assert result.returncode == 0, (
            f"Compilation failed for {case.test_id}:\n"
            f"  stdout={result.stdout}\n  stderr={result.stderr}"
        )
        assert Path(artifact_path).exists(), (
            f"benchmark_kernel.so not produced for {case.test_id}"
        )

    # Phase 2: GPU evaluation
    cmd = pkg.execute()
    result = _run_subprocess(cmd, cwd=pkg.output_dir)
    assert result.returncode == 0, (
        f"Execution failed for {case.test_id}:\n"
        f"  stdout={result.stdout}\n  stderr={result.stderr}"
    )

    traces = pkg.convert_stdout_to_traces(result.stdout)
    assert len(traces) == case.expected_count, (
        f"Expected {case.expected_count} traces for {case.test_id}, got {len(traces)}"
    )

    failed = [t for t in traces if not t.is_successful()]
    assert not failed, (
        f"{case.test_id}: {len(failed)}/{case.expected_count} workloads did not pass:\n"
        + "\n".join(
            f"  [{t.evaluation.status.value}] uuid={t.workload.uuid}  "
            f"log={t.evaluation.log}"
            for t in failed
        )
    )

    # Per-trace invariants for PASSED workloads
    for trace in traces:
        assert trace.definition == definition.name

        ev = trace.evaluation
        assert ev.correctness is not None, (
            f"PASSED trace missing correctness (uuid={trace.workload.uuid})"
        )
        assert ev.performance is not None, (
            f"PASSED trace missing performance (uuid={trace.workload.uuid})"
        )
        assert ev.performance.latency_ms > 0, (
            f"PASSED trace has zero latency (uuid={trace.workload.uuid})"
        )
        assert ev.environment.hardware, (
            f"Empty hardware field (uuid={trace.workload.uuid})"
        )


# ---------------------------------------------------------------------------
# Source file consistency tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c.test_id) for c in _EXAMPLES],
)
def test_consistency(case: Example):
    """Verify extracted source files match the content in definition.json and solution JSON."""
    example_dir = _EXAMPLES_DIR / case.language / case.problem

    # Check reference.py matches definition.json "reference" field
    definition = json.loads((example_dir / "definition.json").read_text())
    if "reference" in definition:
        ref_path = example_dir / "reference.py"
        assert ref_path.exists(), f"reference.py missing for {case.test_id}"
        assert ref_path.read_text() == definition["reference"], (
            f"reference.py does not match definition.json reference for {case.test_id}"
        )

    # Check each solution source file matches its "content" field
    solution = json.loads((example_dir / case.solution_file).read_text())
    for source in solution.get("sources", []):
        src_path = example_dir / source["path"]
        assert src_path.exists(), f"{source['path']} missing for {case.test_id}"
        assert src_path.read_text() == source["content"], (
            f"{source['path']} does not match {case.solution_file} content for {case.test_id}"
        )
