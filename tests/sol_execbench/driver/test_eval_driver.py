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


"""Subprocess integration tests for the SOL ExecBench eval driver.

Runs eval_driver.py directly in a temp directory — no GPU server required.
Verifies that:
  - A correct solution emits PASSED traces.
  - Clock-lock unavailability warnings propagate into trace logs.
  - Reward-hack defenses (monkey-patch, thread injection, lazy outputs) fire and
    emit REWARD_HACK traces with descriptive log messages.
  - @torch.compile solutions do NOT trigger a thread injection false positive.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pytest
import torch

import sol_execbench.driver as _driver_pkg
from sol_execbench.core.bench.reward_hack import RewardHackDetected
from sol_execbench.driver.templates import eval_driver

_TEMPLATES_DIR = Path(_driver_pkg.__file__).parent / "templates"


def build_driver() -> str:
    """Return the eval_driver.py template source."""
    return (_TEMPLATES_DIR / "eval_driver.py").read_text()


def parse_eval_result(stdout: str, stderr: str) -> list[dict]:
    """Parse JSONL Trace dicts from eval_driver stdout."""
    traces = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            traces.append(json.loads(line))
    return traces


# ---------------------------------------------------------------------------
# Minimal problem used by all subprocess tests
# ---------------------------------------------------------------------------

_MINIMAL_DEFINITION = {
    "name": "test_vecadd",
    "op_type": "elementwise",
    "axes": {"n": {"type": "const", "value": 64}},
    "inputs": {
        "x": {"shape": ["n"], "dtype": "float32"},
        "y": {"shape": ["n"], "dtype": "float32"},
    },
    "outputs": {"z": {"shape": ["n"], "dtype": "float32"}},
    "reference": "import torch\ndef run(x, y):\n    return x + y",
}

_MINIMAL_WORKLOAD = {
    "axes": {},
    "inputs": {
        "x": {"type": "random"},
        "y": {"type": "random"},
    },
    "uuid": "sub-test-0001",
}

_SOLUTION_SPEC = {
    "name": "test_vecadd_solution",
    "definition": "test_vecadd",
    "author": "test",
    "spec": {
        "languages": ["pytorch"],
        "target_hardware": ["LOCAL"],
        "entry_point": "kernel.py::run",
        # Must be explicit: driver defaults to True (DPS) if omitted.
        "destination_passing_style": False,
    },
    "sources": [{"path": "kernel.py", "content": "# loaded by test"}],
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_eval_driver(
    tmp_path: Path,
    kernel_code: str,
    bench_config: Optional[dict] = None,
    extra_env: Optional[dict] = None,
    definition: Optional[dict] = None,
    dps: bool = False,
) -> list[dict]:
    """Write all staging files and run eval_driver.py in a subprocess.

    Args:
        tmp_path: Temporary directory to use as the staging directory.
        kernel_code: Python source to write to kernel.py.
        bench_config: BenchmarkConfig overrides to write to config.json.
            Defaults to ``{"lock_clocks": False}`` to suppress nvidia-smi
            warnings on machines where clock locking is not available.
        extra_env: Additional environment variables for the subprocess.
        definition: Problem definition dict. Defaults to ``_MINIMAL_DEFINITION``.

    Returns:
        List of Trace dicts parsed from the driver's stdout.
    """
    (tmp_path / "eval_driver.py").write_text(build_driver())
    (tmp_path / "definition.json").write_text(
        json.dumps(definition if definition is not None else _MINIMAL_DEFINITION)
    )
    workload_path = tmp_path / "workload.jsonl"
    if not workload_path.exists():
        workload_path.write_text(json.dumps(_MINIMAL_WORKLOAD))
    solution = {
        **_SOLUTION_SPEC,
        "spec": {**_SOLUTION_SPEC["spec"], "destination_passing_style": dps},
    }
    (tmp_path / "solution.json").write_text(json.dumps(solution))
    (tmp_path / "kernel.py").write_text(kernel_code)

    # Default config: disable clock-locking (no nvidia-smi).
    cfg = (
        bench_config
        if bench_config is not None
        else {
            "lock_clocks": False,
        }
    )
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    env = {**os.environ, "SOL_EXECBENCH_CLOCKS_LOCKED": "0"}
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        [sys.executable, "-E", "-P", "eval_driver.py"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return parse_eval_result(result.stdout, result.stderr)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_eval_driver_is_valid_python():
    """build_driver() must produce a syntactically valid Python script."""
    source = build_driver()
    ast.parse(source, filename="eval_driver.py")


def test_worker_rechecks_original_integrity_snapshot(monkeypatch):
    guard = eval_driver._make_integrity_guard()
    with guard():
        pass
    monkeypatch.delattr(eval_driver, "normalize_outputs")
    entered = False
    with pytest.raises(RewardHackDetected, match="normalize_outputs"):
        with guard():
            entered = True
    assert not entered


@pytest.mark.xdist_group("serial")
def test_passing_solution(tmp_path):
    """A correct vector-add kernel produces a PASSED trace."""
    kernel = "import torch\ndef run(x, y):\n    return x + y\n"
    traces = _run_eval_driver(tmp_path, kernel)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "PASSED", (
        f"Expected PASSED, got {ev['status']}; log={ev.get('log')}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA timing required")
def test_submission_precision_is_graded_by_workload_tolerance(tmp_path):
    workloads = [
        {
            **_MINIMAL_WORKLOAD,
            "uuid": name,
            "tolerance": {"max_atol": atol, "max_rtol": 0},
        }
        for name, atol in (("strict", 0.01), ("loose", 0.25))
    ]
    (tmp_path / "workload.jsonl").write_text("\n".join(map(json.dumps, workloads)))
    kernel = (
        "import torch\ntorch.set_float32_matmul_precision('high')\n"
        "def run(x, y):\n    return x + y + 0.125"
    )
    traces = _run_eval_driver(tmp_path, kernel)
    assert [trace["workload"]["uuid"] for trace in traces] == ["strict", "loose"]
    strict, loose = (trace["evaluation"] for trace in traces)
    assert strict["status"] == "INCORRECT_NUMERICAL", strict["log"]
    assert strict["performance"] is None
    assert loose["status"] == "PASSED", loose["log"]
    assert loose["performance"]["latency_ms"] > 0


def test_reference_and_judge_do_not_inherit_submission_defaults(tmp_path):
    (tmp_path / "sitecustomize.py").write_text(
        "from pathlib import Path\nPath('startup-hook').touch()\n"
    )
    reference = (
        "import torch\ndef run(x, y):\n"
        "    offset = (torch.tensor(16777216.) + 1.) - 16777216.\n"
        "    return x + y + offset.item()"
    )
    kernel = (
        "import torch\ntorch.set_default_dtype(torch.float64)\n"
        "def run(x, y):\n    return x + y + 1.0"
    )
    traces = _run_eval_driver(
        tmp_path,
        kernel,
        definition={**_MINIMAL_DEFINITION, "reference": reference},
        extra_env={
            "PYTHONPATH": os.pathsep.join(
                [str(tmp_path), os.environ.get("PYTHONPATH", "")]
            )
        },
    )
    assert not (tmp_path / "startup-hook").exists()
    (evaluation,) = (trace["evaluation"] for trace in traces)
    assert evaluation["status"] == "INCORRECT_NUMERICAL"
    assert evaluation["performance"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA timing required")
def test_reference_timing_uses_only_reference_process(tmp_path):
    reference = (
        "import os, torch\nfrom pathlib import Path\n"
        "def run(x, y):\n"
        "    assert str(os.getpid()) != Path('solution.pid').read_text()\n"
        "    return x + y"
    )
    kernel = (
        "import os\nfrom pathlib import Path\n"
        "Path('solution.pid').write_text(str(os.getpid()))\n"
        "def run(x, y, output):\n    output.copy_(x + y)"
    )
    traces = _run_eval_driver(
        tmp_path,
        kernel,
        dps=True,
        definition={**_MINIMAL_DEFINITION, "reference": reference},
        bench_config={"benchmark_reference": True, "warmup_runs": 2, "iterations": 3},
    )
    (evaluation,) = (trace["evaluation"] for trace in traces)
    assert evaluation["status"] == "PASSED", evaluation["log"]
    assert evaluation["performance"]["reference_latency_ms"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA timing required")
def test_optional_reference_timing_failure_keeps_user_score(tmp_path):
    reference = (
        "from pathlib import Path\ncalls = 0\n"
        "def run(x, y):\n"
        "    global calls\n    calls += 1\n"
        "    if calls > 10:\n"
        "        Path('reference-timed').touch()\n"
        "        raise RuntimeError('reference timing unavailable')\n"
        "    return x + y"
    )
    traces = _run_eval_driver(
        tmp_path,
        "def run(x, y):\n    return x + y",
        definition={**_MINIMAL_DEFINITION, "reference": reference},
        bench_config={"benchmark_reference": True, "warmup_runs": 2, "iterations": 3},
    )
    (evaluation,) = (trace["evaluation"] for trace in traces)
    assert (tmp_path / "reference-timed").exists()
    assert evaluation["status"] == "PASSED", evaluation["log"]
    assert evaluation["performance"]["latency_ms"] > 0
    assert evaluation["performance"]["reference_latency_ms"] == 0
    assert evaluation["performance"]["speedup_factor"] == 0


def test_worker_exit_leaves_no_partial_scores(tmp_path):
    workloads = [{**_MINIMAL_WORKLOAD, "uuid": name} for name in ("first", "second")]
    (tmp_path / "workload.jsonl").write_text("\n".join(map(json.dumps, workloads)))
    traces = _run_eval_driver(tmp_path, "def run(x, y):\n    raise SystemExit(3)")
    assert [trace["workload"]["uuid"] for trace in traces] == ["first", "second"]
    assert all(
        trace["evaluation"]["status"] == "RUNTIME_ERROR"
        and trace["evaluation"]["performance"] is None
        for trace in traces
    )


def test_reference_output_alias_is_not_a_worker_input(tmp_path):
    traces = _run_eval_driver(
        tmp_path,
        "def run(x, y):\n    return x.zero_()",
        definition={**_MINIMAL_DEFINITION, "reference": "def run(x, y):\n    return x"},
    )
    assert [trace["evaluation"]["status"] for trace in traces] == [
        "INCORRECT_NUMERICAL"
    ]


def test_monkey_patch_detected(tmp_path):
    """Kernel that patches torch.cuda.Event.elapsed_time at import → REWARD_HACK."""
    kernel = (
        "import torch\n"
        "torch.cuda.Event.elapsed_time = lambda self, end: 0.0\n"
        "\n"
        "def run(x, y):\n"
        "    return x + y\n"
    )
    traces = _run_eval_driver(tmp_path, kernel)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "REWARD_HACK", (
        f"Expected REWARD_HACK, got {ev['status']}; log={ev.get('log')}"
    )
    assert "monkey-patched" in ev.get("log", ""), (
        f"Expected 'monkey-patched' in log; got: {ev.get('log')}"
    )


@pytest.mark.xdist_group("serial")
def test_thread_injection_detected(tmp_path):
    """Kernel that spawns a daemon thread inside run() → REWARD_HACK."""
    kernel = (
        "import threading\n"
        "import time\n"
        "\n"
        "def run(x, y):\n"
        "    threading.Thread(target=lambda: time.sleep(10), daemon=True).start()\n"
        "    return x + y\n"
    )
    traces = _run_eval_driver(tmp_path, kernel)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "REWARD_HACK", (
        f"Expected REWARD_HACK, got {ev['status']}; log={ev.get('log')}"
    )
    assert "thread" in ev.get("log", "").lower(), (
        f"Expected 'thread' in log; got: {ev.get('log')}"
    )


def test_torch_compile_no_reward_hack(tmp_path):
    """@torch.compile must not trigger thread injection false positive.

    TorchInductor spawns background threads on the first call (JIT compilation).
    With the fix, the thread snapshot is taken AFTER the correctness call, so
    those threads are already included in the baseline when timing runs.
    """
    kernel = (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "\n"
        "@torch.compile\n"
        "@torch.no_grad()\n"
        "def run(x, y):\n"
        "    return x + y\n"
    )
    traces = _run_eval_driver(tmp_path, kernel)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "PASSED", (
        f"Expected PASSED (torch.compile false positive fixed), "
        f"got {ev['status']}; log={ev.get('log')}"
    )


def test_lazy_output_detected(tmp_path):
    """Kernel that returns a torch.Tensor subclass → REWARD_HACK."""
    kernel = (
        "import torch\n"
        "\n"
        "class _FakeTensor(torch.Tensor):\n"
        "    pass\n"
        "\n"
        "def run(x, y):\n"
        "    return (x + y).as_subclass(_FakeTensor)\n"
    )
    traces = _run_eval_driver(tmp_path, kernel)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "REWARD_HACK", (
        f"Expected REWARD_HACK, got {ev['status']}; log={ev.get('log')}"
    )
    assert "_FakeTensor" in ev.get("log", ""), (
        f"Expected '_FakeTensor' in log; got: {ev.get('log')}"
    )


# Check if triton is available for the regression test
_triton_available = False
try:
    import triton  # noqa: F401

    _triton_available = True
except ImportError:
    pass


@pytest.mark.skipif(not _triton_available, reason="triton not installed")
def test_triton_jit_reference(tmp_path):
    """Reference with @triton.jit must not crash the eval driver.

    Regression test: exec(compile(code, '<reference>', 'exec')) causes
    triton.JITFunction.__init__ to call inspect.getsourcelines() which
    fails because '<reference>' is not a real file.  The fix writes the
    reference to _reference.py and imports it via importlib.
    """
    # Definition whose reference uses @triton.jit for a vector-add.
    triton_reference = (
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "@triton.jit\n"
        "def _vecadd_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK: tl.constexpr):\n"
        "    pid = tl.program_id(0)\n"
        "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
        "    mask = offs < n\n"
        "    x = tl.load(x_ptr + offs, mask=mask)\n"
        "    y = tl.load(y_ptr + offs, mask=mask)\n"
        "    tl.store(z_ptr + offs, x + y, mask=mask)\n"
        "\n"
        "def run(x, y):\n"
        "    z = torch.empty_like(x)\n"
        "    n = x.numel()\n"
        "    _vecadd_kernel[(n + 255) // 256,](x, y, z, n, BLOCK=256)\n"
        "    return z\n"
    )
    definition = {**_MINIMAL_DEFINITION, "reference": triton_reference}

    kernel = "import torch\ndef run(x, y):\n    return x + y\n"
    traces = _run_eval_driver(tmp_path, kernel, definition=definition)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "PASSED", (
        f"Expected PASSED (triton @jit reference), got {ev['status']}; "
        f"log={ev.get('log')}"
    )


# ---------------------------------------------------------------------------
# Evil tests: load_inline and stream keyword defenses
# ---------------------------------------------------------------------------


@pytest.mark.xdist_group("serial")
def test_load_inline_blocked_in_run(tmp_path):
    """Kernel that calls load_inline inside run() → RUNTIME_ERROR."""
    kernel = (
        "from torch.utils import cpp_extension\n"
        "\n"
        "def run(x, y):\n"
        "    cpp_extension.load_inline(name='evil', cpp_sources='void f(){}')\n"
        "    return x + y\n"
    )
    traces = _run_eval_driver(tmp_path, kernel)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "RUNTIME_ERROR", (
        f"Expected RUNTIME_ERROR, got {ev['status']}; log={ev.get('log')}"
    )
    assert "not permitted" in ev.get("log", ""), (
        f"Expected 'not permitted' in log; got: {ev.get('log')}"
    )


def test_load_inline_blocked_at_import(tmp_path):
    """Kernel that calls load_inline at import time → subprocess crash."""
    kernel = (
        "from torch.utils.cpp_extension import load_inline\n"
        "load_inline(name='evil', cpp_sources='void f(){}')\n"
        "\n"
        "def run(x, y):\n"
        "    return x + y\n"
    )
    (tmp_path / "eval_driver.py").write_text(build_driver())
    (tmp_path / "definition.json").write_text(json.dumps(_MINIMAL_DEFINITION))
    (tmp_path / "workload.jsonl").write_text(json.dumps(_MINIMAL_WORKLOAD))
    (tmp_path / "solution.json").write_text(json.dumps(_SOLUTION_SPEC))
    (tmp_path / "kernel.py").write_text(kernel)
    (tmp_path / "config.json").write_text(json.dumps({"lock_clocks": False}))

    result = subprocess.run(
        [sys.executable, "-E", "-P", "eval_driver.py"],
        cwd=tmp_path,
        env={**os.environ, "SOL_EXECBENCH_CLOCKS_LOCKED": "0"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "not permitted" in result.stderr, (
        f"Expected 'not permitted' in stderr; got:\n{result.stderr}"
    )


def test_cpp_getCurrentCUDAStream_allowed():
    """C++ files may use getCurrentCUDAStream (the default stream)."""
    from sol_execbench.core.data import Solution

    good_cuda = {
        "name": "good_cuda_stream",
        "definition": "test_def",
        "author": "good_agent",
        "spec": {
            "languages": ["cuda_cpp"],
            "target_hardware": ["LOCAL"],
            "entry_point": "main.cpp::run",
            "destination_passing_style": True,
        },
        "sources": [
            {
                "path": "main.cpp",
                "content": "void run() {}",
            },
            {
                "path": "kernel.cu",
                "content": (
                    "auto stream = at::cuda::getCurrentCUDAStream();\n"
                    "kernel<<<grid, block, 0, stream>>>(args);\n"
                ),
            },
        ],
    }
    # Should NOT raise
    Solution(**good_cuda)


# ---------------------------------------------------------------------------
# NaN / Inf JSON compliance tests
# ---------------------------------------------------------------------------


def test_emit_uses_strict_json():
    """_emit() must use allow_nan=False so NaN/Inf crashes loudly (no torch required)."""
    source = build_driver()
    assert "allow_nan=False" in source, (
        "_emit must use json.dumps(..., allow_nan=False) to reject non-finite floats"
    )
    assert "_sanitize_floats" not in source, (
        "_sanitize_floats should not be present — NaN/Inf must crash, not be silently hidden"
    )


@pytest.mark.xdist_group("serial")
def test_nan_kernel_produces_valid_json(tmp_path):
    """Kernel that outputs NaN must produce strictly valid JSON traces.

    Regression test: Python's json.dumps(allow_nan=True) serializes NaN as
    the JavaScript literal ``NaN`` which is not valid JSON per RFC 8259.
    Strict parsers (e.g., Rust serde_json) reject the entire line, silently
    dropping the trace.  The fix sanitizes NaN/Inf to null before dumping.
    """
    kernel = (
        "import torch\n\ndef run(x, y):\n    return torch.full_like(x, float('nan'))\n"
    )
    traces = _run_eval_driver(tmp_path, kernel)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "INCORRECT_NUMERICAL", (
        f"Expected INCORRECT_NUMERICAL, got {ev['status']}; log={ev.get('log')}"
    )

    # The critical check: re-serialize to JSON with allow_nan=False.
    # This mimics what a strict JSON parser (Rust, Go, etc.) would do.
    # It must NOT raise ValueError.
    import json as json_mod

    strict_json = json_mod.dumps(traces[0], allow_nan=False)
    reparsed = json_mod.loads(strict_json)
    assert reparsed["evaluation"]["status"] == "INCORRECT_NUMERICAL"


def test_inf_kernel_produces_valid_json(tmp_path):
    """Kernel that outputs Inf must produce strictly valid JSON traces."""
    kernel = (
        "import torch\n\ndef run(x, y):\n    return torch.full_like(x, float('inf'))\n"
    )
    traces = _run_eval_driver(tmp_path, kernel)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "INCORRECT_NUMERICAL", (
        f"Expected INCORRECT_NUMERICAL, got {ev['status']}; log={ev.get('log')}"
    )

    import json as json_mod

    strict_json = json_mod.dumps(traces[0], allow_nan=False)
    reparsed = json_mod.loads(strict_json)
    assert reparsed["evaluation"]["status"] == "INCORRECT_NUMERICAL"


def test_nan_correctness_fields_are_finite_with_flags(tmp_path):
    """NaN outputs must produce finite correctness values with has_nan=True."""
    kernel = (
        "import torch\n\ndef run(x, y):\n    return torch.full_like(x, float('nan'))\n"
    )
    traces = _run_eval_driver(tmp_path, kernel)

    assert len(traces) == 1
    correctness = traces[0]["evaluation"].get("correctness", {})

    # Error values must be finite (not NaN/Inf)
    max_abs = correctness.get("max_absolute_error", 0.0)
    max_rel = correctness.get("max_relative_error", 0.0)
    assert isinstance(max_abs, (int, float)) and max_abs == max_abs, (
        f"max_absolute_error should be finite, got {max_abs!r}"
    )
    assert isinstance(max_rel, (int, float)) and max_rel == max_rel, (
        f"max_relative_error should be finite, got {max_rel!r}"
    )

    # Boolean flag must indicate NaN was detected
    assert correctness.get("has_nan") is True, (
        f"has_nan should be True for NaN output, got {correctness}"
    )


def test_passing_trace_correctness_unchanged(tmp_path):
    """PASSED traces must preserve finite correctness values (not clobber to null)."""
    kernel = "import torch\ndef run(x, y):\n    return x + y\n"
    traces = _run_eval_driver(tmp_path, kernel)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "PASSED"

    correctness = ev.get("correctness", {})
    max_abs = correctness.get("max_absolute_error")
    max_rel = correctness.get("max_relative_error")

    # For a correct kernel, error values must be finite numbers (not null)
    assert isinstance(max_abs, (int, float)) and max_abs == max_abs, (
        f"Expected finite max_absolute_error for PASSED trace, got {max_abs!r}"
    )
    assert isinstance(max_rel, (int, float)) and max_rel == max_rel, (
        f"Expected finite max_relative_error for PASSED trace, got {max_rel!r}"
    )
