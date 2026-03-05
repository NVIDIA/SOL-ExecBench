"""End-to-end API tests for the evaluate module.

Tests the full evaluation pipeline from definition + workload + solution to
Trace objects, parameterized over solution files.
"""

import logging
from pathlib import Path

import pytest
import torch

from sol_execbench import (
    Definition,
    Solution,
    SupportedHardware,
    BenchmarkConfig,
    Benchmark,
    Workload,
    load_json_file,
    load_jsonl_file,
    EvaluationStatus,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLES_DIR = Path(__file__).parent / "samples"


# ---------------------------------------------------------------------------
# Original rmsnorm fixtures (kept for backward-compat)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dir():
    return SAMPLES_DIR / "rmsnorm"


@pytest.fixture
def definition(sample_dir):
    definition_path = sample_dir / "definition.json"
    if not definition_path.exists():
        pytest.skip(f"Definition not found at {definition_path}")
    return load_json_file(Definition, definition_path)


@pytest.fixture
def workloads(sample_dir):
    workload_path = sample_dir / "workload.jsonl"
    if not workload_path.exists():
        pytest.skip(f"Workload file not found at {workload_path}")
    return load_jsonl_file(Workload, workload_path)


@pytest.fixture(
    params=[
        pytest.param("solution_cuda.json", marks=pytest.mark.cpp),
        "solution_triton.json",
    ]
)
def solution(request, sample_dir):
    solution_path = sample_dir / request.param
    if not solution_path.exists():
        pytest.skip(f"Solution not found at {solution_path}")
    return load_json_file(Solution, solution_path)


@pytest.fixture
def hardware():
    return SupportedHardware.LOCAL


@pytest.fixture
def benchmark_config():
    return BenchmarkConfig(lock_clocks=False)


class TestBenchmarkAPI:
    """Tests for the Benchmark class API."""

    def test_benchmark_initialization(self, hardware, benchmark_config):
        """Test Benchmark class initialization."""
        benchmark = Benchmark(hardware=hardware, config=benchmark_config)
        assert benchmark is not None

    def test_benchmark_run_workload_api(
        self, definition, workloads, solution, hardware, benchmark_config
    ):
        """Test Benchmark.run_workload API."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")

        benchmark = Benchmark(hardware=hardware, config=benchmark_config)
        workload = workloads[0]

        results = benchmark.run_workload(
            definition=definition,
            workload=workload,
            solutions=[solution],
            trace_set_root=Path(__file__).parent.parent.parent.absolute(),
        )

        assert results is not None
        assert isinstance(results, dict)
        assert solution.name in results

        evaluation = results[solution.name]
        assert hasattr(evaluation, "status")
        assert hasattr(evaluation, "performance")

        logger.info("✓ Benchmark.run_workload completed")
        logger.info(f"  - Status: {evaluation.status.value}")
        if evaluation.performance is not None:
            logger.info(f"  - Latency: {evaluation.performance.latency_ms:.3f}ms")


# ---------------------------------------------------------------------------
# Helper to run a single (sample_dir, solution_file) through the benchmark
# ---------------------------------------------------------------------------


def _run_e2e(sample_name: str, solution_file: str):
    """Load a sample problem and evaluate a solution end-to-end."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    sample_dir = SAMPLES_DIR / sample_name
    if not sample_dir.exists():
        pytest.skip(f"Sample directory not found: {sample_dir}")

    definition_path = sample_dir / "definition.json"
    workload_path = sample_dir / "workload.jsonl"
    solution_path = sample_dir / solution_file

    for p in (definition_path, workload_path, solution_path):
        if not p.exists():
            pytest.skip(f"Missing file: {p}")

    definition = load_json_file(Definition, definition_path)
    workloads = load_jsonl_file(Workload, workload_path)
    solution = load_json_file(Solution, solution_path)

    benchmark = Benchmark(
        hardware=SupportedHardware.LOCAL,
        config=BenchmarkConfig(lock_clocks=False),
    )

    results = benchmark.run_workload(
        definition=definition,
        workload=workloads[0],
        solutions=[solution],
        trace_set_root=Path(__file__).parent.parent.parent.absolute(),
    )

    assert results is not None
    assert isinstance(results, dict)
    assert solution.name in results

    evaluation = results[solution.name]
    assert hasattr(evaluation, "status")
    assert hasattr(evaluation, "performance")

    assert evaluation.status == EvaluationStatus.PASSED

    logger.info(
        "✓ %s/%s — status=%s",
        sample_name,
        solution_file,
        evaluation.status.value,
    )
    if evaluation.performance is not None:
        logger.info("  latency=%.3fms", evaluation.performance.latency_ms)

    return evaluation


# ---------------------------------------------------------------------------
# E2E tests for different kernel libraries
# ---------------------------------------------------------------------------


class TestPythonSolutions:
    """E2E tests for pure Python/PyTorch solutions."""

    def test_gemma3_swiglu(self):
        """SwiGLU MLP (Python/PyTorch)."""
        _run_e2e("gemma3_swiglu", "solution_python.json")


class TestTritonSolutions:
    """E2E tests for Triton kernel solutions."""

    def test_olmo3_post_norm_residual(self):
        """Post-norm + residual RMSNorm (Triton)."""
        _run_e2e("olmo3_post_norm", "solution_triton.json")

    def test_nemotron_fused_add_rmsnorm(self):
        """Fused add + RMSNorm (Triton)."""
        _run_e2e("nemotron_rms_norm", "solution_triton.json")


@pytest.mark.cpp
class TestCUDASolutions:
    """E2E tests for CUDA C++ solutions (require compilation)."""

    def test_flux_rope(self):
        """RoPE rotation (CUDA C++, expression axis)."""
        _run_e2e("flux_rope", "solution_cuda.json")


class TestCuteDSLSolutions:
    """E2E tests for CuTe DSL solutions (require cutlass + compilation)."""

    def test_jamba_attn_proj_residual(self):
        """Attention output projection + residual (CuTe DSL)."""
        _run_e2e("jamba_attn_proj", "solution_cute_dsl.json")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="cuTile requires sm_100+",
)
class TestCuTileSolutions:
    """E2E tests for cuTile DSL solutions (require cuda_tile + compilation)."""

    def test_jamba_attn_proj_residual(self):
        """Attention output projection + residual (cuTile)."""
        _run_e2e("jamba_attn_proj", "solution_cutile.json")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
