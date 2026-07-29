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

"""End-to-end SOL ExecBench evaluation tests.

Each test loads a self-contained sample (definition.json + workload.jsonl +
solution_*.json with inline kernel source), packages it via ProblemPackager,
runs the compile and execute phases locally via subprocess, and asserts that
all workloads pass.

Coverage (language-specific examples are in test_examples.py):
  1. Custom inputs                   — custom_inputs_matmul (custom_inputs_entrypoint)
  2. Triton @jit reference           — triton_ref_vecadd   (importlib reference loading)
  3. Reward-hack detection           — evil_* samples
  4. CLI e2e                         — gqa_paged_decode
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
    Trace,
    EvaluationStatus,
)
from sol_execbench.driver.problem_packager import ProblemPackager

_SAMPLES_DIR = Path(__file__).parent / "samples"


# ---------------------------------------------------------------------------
# Test case descriptors
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    """Parameters for one SOL ExecBench e2e test case."""

    test_id: str
    sample: str
    solution_file: str
    expected_count: int
    extra_markers: list[str] = field(default_factory=list)


@dataclass
class EvilCase:
    """Parameters for one reward-hack e2e test case."""

    test_id: str
    sample: str
    expected_log_fragment: str
    config_overrides: dict = field(default_factory=dict)


# Inline problem definition for evil-case e2e tests.
_EVIL_DEFINITION_DICT = {
    "name": "evil_test_vecadd",
    # A var axis with differently-shaped workloads: solutions that keep
    # per-shape state (evil_timing_bypass) then behave the same on both.
    "axes": {"n": {"type": "var"}},
    "inputs": {
        "x": {"shape": ["n"], "dtype": "float32"},
        "y": {"shape": ["n"], "dtype": "float32"},
    },
    "outputs": {"z": {"shape": ["n"], "dtype": "float32"}},
    "reference": "import torch\ndef run(x, y):\n    return x + y",
}

_EVIL_WORKLOAD_DICTS = [
    {
        "axes": {"n": 256},
        "inputs": {"x": {"type": "random"}, "y": {"type": "random"}},
        "uuid": "evil-wkl-0001",
    },
    {
        "axes": {"n": 512},
        "inputs": {"x": {"type": "random"}, "y": {"type": "random"}},
        "uuid": "evil-wkl-0002",
    },
]

_EVIL_CASES = [
    EvilCase("evil_monkey_patch", "evil_monkey_patch", "monkey-patched"),
    EvilCase("evil_thread_inject", "evil_thread_inject", "thread"),
    EvilCase("evil_lazy_output", "evil_lazy_output", "_FakeTensor"),
    EvilCase("evil_timing_bypass", "evil_timing_bypass", "timing began"),
]

_CASES = [
    Sample(
        test_id="custom_inputs_matmul_python",
        sample="custom_inputs_matmul",
        solution_file="solution_python.json",
        expected_count=3,
    ),
    Sample(
        test_id="triton_ref_vecadd_python",
        sample="triton_ref_vecadd",
        solution_file="solution_python.json",
        expected_count=3,
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CPP_LANGUAGES = {"cuda_cpp", "cutlass", "cudnn", "cublas"}


def _load_sample(
    sample: str, solution_file: str
) -> tuple[Definition, Solution, list[Workload]]:
    """Load definition, solution, and workloads from a self-contained sample directory."""
    sample_dir = _SAMPLES_DIR / sample
    definition = Definition(**json.loads((sample_dir / "definition.json").read_text()))
    sol_dict = json.loads((sample_dir / solution_file).read_text())
    solution = Solution(**sol_dict)
    workloads = [
        Workload(**json.loads(line))
        for line in (sample_dir / "workload.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return definition, solution, workloads


def _load_evil_sample(sample: str) -> Solution:
    """Load evil solution with kernel content inlined from the sample directory."""
    sample_dir = _SAMPLES_DIR / sample
    sol_dict = json.loads((sample_dir / "solution.json").read_text())
    kernel_content = (sample_dir / "kernel.py").read_text()
    sol_dict["sources"] = [{"path": "kernel.py", "content": kernel_content}]
    return Solution(**sol_dict)


def _run_subprocess(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run a command in the given directory and return the result."""
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _mark_case(case: Sample) -> list:
    marks = [pytest.mark.xdist_group("serial")]
    for m in case.extra_markers:
        marks.append(getattr(pytest.mark, m))
    return marks


# ---------------------------------------------------------------------------
# Parametrized e2e test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c.test_id, marks=_mark_case(c)) for c in _CASES],
)
def test_e2e(tmp_path: Path, case: Sample):
    """End-to-end evaluation: load sample -> package -> compile (if C++) -> execute -> assert PASSED."""
    definition, solution, workloads = _load_sample(case.sample, case.solution_file)
    config = BenchmarkConfig(lock_clocks=False)

    pkg = ProblemPackager(
        definition=definition,
        workloads=workloads,
        solution=solution,
        config=config,
        output_dir=tmp_path / "staging",
        keep_output_dir=True,
    )

    # Phase 1 (CUDA/C++ only): compile
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
# Reward-hack e2e tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, id=c.test_id, marks=[pytest.mark.xdist_group("serial")])
        for c in _EVIL_CASES
    ],
)
def test_reward_hack_e2e(tmp_path: Path, case: EvilCase):
    """Evil solutions must produce REWARD_HACK traces with the expected log fragment."""
    evil_solution = _load_evil_sample(case.sample)
    definition = Definition(**_EVIL_DEFINITION_DICT)
    workloads = [Workload(**w) for w in _EVIL_WORKLOAD_DICTS]
    config = BenchmarkConfig(lock_clocks=False, **case.config_overrides)

    pkg = ProblemPackager(
        definition=definition,
        workloads=workloads,
        solution=evil_solution,
        config=config,
        output_dir=tmp_path / "staging",
        keep_output_dir=True,
    )

    cmd = pkg.execute()
    result = _run_subprocess(cmd, cwd=pkg.output_dir)
    assert result.returncode == 0, (
        f"Eval driver process failed for {case.test_id}:\n  stderr={result.stderr}"
    )

    traces = pkg.convert_stdout_to_traces(result.stdout)
    assert len(traces) == len(_EVIL_WORKLOAD_DICTS), (
        f"Expected {len(_EVIL_WORKLOAD_DICTS)} traces for {case.test_id}, got {len(traces)}"
    )

    for t in traces:
        assert t.evaluation.status.value == "REWARD_HACK", (
            f"{case.test_id}: expected REWARD_HACK, got {t.evaluation.status.value}; "
            f"uuid={t.workload.uuid}  log={t.evaluation.log}"
        )
        assert case.expected_log_fragment in t.evaluation.log, (
            f"{case.test_id}: expected {case.expected_log_fragment!r} in log; "
            f"got: {t.evaluation.log}"
        )


# ---------------------------------------------------------------------------
# CLI e2e test — runs `sol-execbench <problem_dir> -o <output>`
# ---------------------------------------------------------------------------


@pytest.mark.xdist_group("serial")
def test_cli_gqa_paged_decode(tmp_path: Path):
    """CLI e2e: run sol-execbench on a GQA paged-decode problem with safetensors inputs."""
    sample_dir = _SAMPLES_DIR / "gqa_paged_decode"
    output_file = tmp_path / "traces.jsonl"

    result = subprocess.run(
        [
            "uv",
            "run",
            "sol-execbench",
            str(sample_dir),
            "-o",
            str(output_file),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"CLI failed:\n  stdout={result.stdout}\n  stderr={result.stderr}"
    )

    assert output_file.exists(), "Output file not created"
    lines = [l for l in output_file.read_text().splitlines() if l.strip()]
    assert len(lines) == 2, f"Expected 2 traces, got {len(lines)}"

    for line in lines:
        trace = Trace(**json.loads(line))
        assert trace.evaluation.status == EvaluationStatus.PASSED, (
            f"Workload {trace.workload.uuid} did not pass: "
            f"status={trace.evaluation.status.value} log={trace.evaluation.log}"
        )
