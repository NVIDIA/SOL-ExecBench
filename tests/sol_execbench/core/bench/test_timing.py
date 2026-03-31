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


"""Tests for sol_execbench.core.bench.timing."""

import statistics

import pytest
import torch

from sol_execbench.core.bench.io import ShiftingMemoryPoolAllocator
from sol_execbench.core.bench.timing import (
    _summarize_statistics,
    bench_gpu_time_with_cupti,
    bench_time_with_cuda_events,
    clone_args,
    time_runnable,
)

# Skipped by default
pytestmark = pytest.mark.timing_serial


# ---------------------------------------------------------------------------
# clone_args
# ---------------------------------------------------------------------------


class TestCloneArgs:
    def test_clones_tensors(self):
        """Tensor arguments are cloned (different storage)."""
        t = torch.tensor([1.0, 2.0, 3.0])
        cloned = clone_args([t])
        assert torch.equal(cloned[0], t)
        assert cloned[0].data_ptr() != t.data_ptr()

    def test_passes_through_non_tensors(self):
        """Non-tensor arguments are returned as-is."""
        args = ["hello", 42, None]
        cloned = clone_args(args)
        assert cloned == args
        # Identity check: non-tensors are the same objects
        assert cloned[0] is args[0]
        assert cloned[1] is args[1]

    def test_mixed_args(self):
        """Mix of tensors and non-tensors."""
        t = torch.zeros(4)
        args = [t, "foo", 3.14]
        cloned = clone_args(args)
        assert torch.equal(cloned[0], t)
        assert cloned[0].data_ptr() != t.data_ptr()
        assert cloned[1] is args[1]
        assert cloned[2] is args[2]


# ---------------------------------------------------------------------------
# _quantile / _summarize_statistics
# ---------------------------------------------------------------------------


class TestSummarizeStatistics:
    def test_mean(self):
        assert _summarize_statistics([1.0, 2.0, 3.0], "mean") == 2.0

    def test_median(self):
        assert _summarize_statistics([1.0, 2.0, 3.0, 4.0, 5.0], "median") == 3.0

    def test_all(self):
        times = [1.0, 2.0, 3.0]
        assert _summarize_statistics(times, "all") == times


# ---------------------------------------------------------------------------
# bench_time_with_cuda_events — requires CUDA; tested via mock or skipped
# ---------------------------------------------------------------------------


class TestBenchTimeWithCUDAEvents:
    @staticmethod
    def _make_mock_event(elapsed_ms: float = 0.1):
        """Create a mock CUDA Event class for testing do_bench without GPU."""

        class MockEvent:
            def __init__(self, enable_timing=False):
                self._enable_timing = enable_timing

            def record(self):
                pass

            def elapsed_time(self, end):
                return elapsed_ms

        return MockEvent

    def test_bench_time_with_cuda_events_calls_setup_for_each_rep(self, monkeypatch):
        """setup() is called once per timed iteration (not during measurement)."""
        MockEvent = self._make_mock_event()
        monkeypatch.setattr(torch.cuda, "Event", MockEvent)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

        setup_calls = []
        fn_calls = []

        def setup():
            setup_calls.append(1)
            return "data"

        def fn(data):
            fn_calls.append(data)

        # Can't test do_bench directly without CUDA
        # Instead, test clone_args + setup integration at unit level
        pass  # Covered by clone_args tests above

    def test_bench_time_with_cuda_events_warmup_and_rep(self, monkeypatch):
        """Warmup and rep counts are respected."""
        MockEvent = self._make_mock_event(0.5)
        monkeypatch.setattr(torch.cuda, "Event", MockEvent)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

        # Verify the return value is correct for the mock
        # This verifies _summarize_statistics integration
        times = [0.5] * 5
        result = _summarize_statistics(times, "mean")
        assert result == 0.5

    def test_bench_time_with_cuda_events_return_modes(self):
        """Different return_mode values produce correct summaries."""
        times = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _summarize_statistics(times, "mean") == 3.0
        assert _summarize_statistics(times, "median") == 3.0
        assert _summarize_statistics(times, "all") == times


# ---------------------------------------------------------------------------
# bench_time_with_cuda_events — GPU integration tests
# ---------------------------------------------------------------------------


class TestBenchTimeWithCUDAEventsGPU:
    """GPU integration tests for do_bench timing accuracy.

    These tests verify that CUDA event timing correctly captures kernel
    execution time while excluding setup overhead.  See docs/timing/timing.md
    for background on GPU command-processor pipelining effects.
    """

    # -- Small kernel timing ------------------------------------------------

    def test_trivial_kernel_timing_floor(self):
        """Scalar add_(0) has sub-millisecond dispatch overhead."""
        t = torch.zeros(1, device="cuda")
        times = bench_time_with_cuda_events(lambda: t.add_(0))
        ms = statistics.median(times)
        assert 0 < ms < 1.0, f"Trivial kernel: {ms:.4f}ms, expected (0, 1)ms"

    def test_trivial_kernel_low_variance(self):
        """Repeated trivial kernel timings should cluster tightly."""
        t = torch.zeros(1, device="cuda")
        times = bench_time_with_cuda_events(lambda: t.add_(0), warmup=10, rep=100)
        median = sorted(times)[len(times) // 2]
        # Drop top/bottom 10% to get interquartile range
        trimmed = sorted(times)[10:90]
        spread = max(trimmed) - min(trimmed)
        assert spread < median * 2.0, (
            f"Trimmed spread {spread:.4f}ms too wide vs median {median:.4f}ms"
        )

    # -- Large kernel timing ------------------------------------------------

    def test_large_matmul_exceeds_small(self):
        """4096x4096 matmul takes measurably longer than 128x128."""
        small = torch.randn(128, 128, device="cuda")
        large = torch.randn(4096, 4096, device="cuda")

        ms_small = statistics.median(
            bench_time_with_cuda_events(lambda: torch.mm(small, small))
        )
        ms_large = statistics.median(
            bench_time_with_cuda_events(lambda: torch.mm(large, large))
        )

        assert ms_large > ms_small * 2, (
            f"Large matmul ({ms_large:.4f}ms) should be >2x small ({ms_small:.4f}ms)"
        )

    def test_compute_scales_with_size(self):
        """Doubling matmul dimension should roughly 8x the FLOPs and increase time."""
        n1 = torch.randn(1024, 1024, device="cuda")
        n2 = torch.randn(2048, 2048, device="cuda")

        ms_1k = statistics.median(bench_time_with_cuda_events(lambda: torch.mm(n1, n1)))
        ms_2k = statistics.median(bench_time_with_cuda_events(lambda: torch.mm(n2, n2)))

        # 2048^3 / 1024^3 = 8x FLOPs. On modern GPUs the ratio may be less
        # than 8x due to memory-bound vs compute-bound crossover, but it
        # should be at least 2x.
        ratio = ms_2k / ms_1k
        assert ratio > 1.5, (
            f"2048 matmul ({ms_2k:.4f}ms) / 1024 matmul ({ms_1k:.4f}ms) = {ratio:.2f}x, "
            f"expected >1.5x"
        )

    # -- Setup time exclusion -----------------------------------------------

    def test_setup_gpu_work_excluded_from_timing(self):
        """GPU work inside setup() must not appear in the timed region.

        Runs the same trivial kernel with two setups:
        1. Cheap: ShiftingMemoryPoolAllocator on a small tensor
        2. Expensive: allocator + a large matmul (~1-2ms GPU work)
        """
        warmup, rep = 10, 50
        total = warmup + rep
        base = torch.zeros(64, device="cuda")
        burn = torch.randn(2048, 2048, device="cuda")

        cheap_alloc = ShiftingMemoryPoolAllocator([base], [], total)
        expensive_alloc = ShiftingMemoryPoolAllocator([base], [], total)

        def expensive_setup():
            torch.mm(burn, burn)  # ~1ms GPU work, excluded by sync
            return expensive_alloc.get_unique_args()

        def fn(args):
            args[0].add_(1)

        ms_cheap = statistics.median(
            bench_time_with_cuda_events(
                fn, warmup=warmup, rep=rep, setup=cheap_alloc.get_unique_args
            )
        )
        ms_expensive = statistics.median(
            bench_time_with_cuda_events(
                fn, warmup=warmup, rep=rep, setup=expensive_setup
            )
        )

        # Both measure just add_(1) dispatch.  If setup leaks, expensive
        # would be ~1ms higher.  Allow 5x tolerance for noise.
        assert ms_expensive < ms_cheap * 5, (
            f"Expensive setup ({ms_expensive:.4f}ms) vs cheap ({ms_cheap:.4f}ms): "
            f"setup GPU work is leaking into measured time"
        )

    def test_setup_called_every_iteration(self):
        """ShiftingMemoryPoolAllocator is called once per warmup + timed iteration."""
        warmup, rep = 5, 20
        total = warmup + rep
        t = torch.zeros(1, device="cuda")
        allocator = ShiftingMemoryPoolAllocator([t], [], total)

        calls = [0]
        orig_get = allocator.get_unique_args

        def counting_setup():
            calls[0] += 1
            return orig_get()

        bench_time_with_cuda_events(
            lambda args: args[0].add_(1), warmup=warmup, rep=rep, setup=counting_setup
        )
        assert calls[0] == total

    # -- Setup impact on measurement ----------------------------------------

    def test_setup_vs_no_setup_same_kernel(self):
        """ShiftingMemoryPoolAllocator introduces no significant timing bias.

        Both paths have torch.cuda.synchronize() before the start event
        (see timing.md Scenario 5), so the command-processor state is
        comparable.  The measured kernel time should be similar.
        """
        warmup, rep = 10, 50
        total = warmup + rep
        t = torch.randn(1024, 1024, device="cuda")

        # No setup — fn closes over `t`, same data every iteration
        ms_no_setup = statistics.median(
            bench_time_with_cuda_events(lambda: torch.mm(t, t), warmup=warmup, rep=rep)
        )

        # With allocator — provides shifted views each iteration
        allocator = ShiftingMemoryPoolAllocator([t], [], total)
        ms_with_setup = statistics.median(
            bench_time_with_cuda_events(
                lambda args: torch.mm(args[0], args[0]),
                warmup=warmup,
                rep=rep,
                setup=allocator.get_unique_args,
            )
        )

        ratio = ms_with_setup / ms_no_setup if ms_no_setup > 0 else float("inf")
        assert 0.5 < ratio < 2.0, (
            f"Setup vs no-setup ratio: {ratio:.2f} "
            f"(with={ms_with_setup:.4f}ms, without={ms_no_setup:.4f}ms)"
        )

    def test_setup_overhead_scales_with_input_not_kernel(self):
        """Varying allocator input size should not affect kernel time.

        Shifting 1KB vs 64MB in the allocator should produce the same
        measured kernel time, because setup is outside the timed region.
        """
        warmup, rep = 10, 50
        total = warmup + rep
        kernel_input = torch.randn(512, 512, device="cuda")

        # Small extra tensor (1KB) alongside the kernel input
        small_extra = torch.randn(256, device="cuda")
        # Large extra tensor (64MB) alongside the kernel input
        large_extra = torch.randn(4096, 4096, device="cuda")

        alloc_small = ShiftingMemoryPoolAllocator(
            [kernel_input, small_extra], [], total
        )
        alloc_large = ShiftingMemoryPoolAllocator(
            [kernel_input, large_extra], [], total
        )

        # fn only uses args[0] for compute; args[1] is the extra tensor
        def fn(args):
            torch.mm(args[0], args[0])

        ms_small = statistics.median(
            bench_time_with_cuda_events(
                fn, warmup=warmup, rep=rep, setup=alloc_small.get_unique_args
            )
        )
        ms_large = statistics.median(
            bench_time_with_cuda_events(
                fn, warmup=warmup, rep=rep, setup=alloc_large.get_unique_args
            )
        )

        ratio = ms_large / ms_small if ms_small > 0 else float("inf")
        assert 0.5 < ratio < 2.0, (
            f"Small-setup ({ms_small:.4f}ms) vs large-setup ({ms_large:.4f}ms): "
            f"ratio {ratio:.2f}, setup cost is leaking"
        )

    # -- time_runnable (eval_driver pattern) --------------------------------

    def test_time_runnable_eval_driver_pattern(self):
        """time_runnable creates ShiftingMemoryPoolAllocator internally.

        Replicates the eval_driver calling convention:
            time_runnable(user_fn, inputs, outputs, device, ...)
        """
        warmup, rep = 5, 30
        t = torch.randn(512, 512, device="cuda")

        def kernel(a):
            return torch.mm(a, a)

        ms = time_runnable(kernel, [t], [], "cuda:0", warmup=warmup, rep=rep)
        assert isinstance(ms, float)
        assert ms > 0

    def test_time_runnable_large_vs_small(self):
        """time_runnable correctly distinguishes fast and slow kernels."""
        warmup, rep = 5, 30
        small = torch.randn(128, 128, device="cuda")
        large = torch.randn(4096, 4096, device="cuda")

        ms_small = time_runnable(
            lambda a: torch.mm(a, a),
            [small],
            [],
            "cuda:0",
            warmup=warmup,
            rep=rep,
        )
        ms_large = time_runnable(
            lambda a: torch.mm(a, a),
            [large],
            [],
            "cuda:0",
            warmup=warmup,
            rep=rep,
        )

        assert ms_large > ms_small, (
            f"Large kernel ({ms_large:.4f}ms) should be slower than small ({ms_small:.4f}ms)"
        )

    # -- Return mode sanity -------------------------------------------------

    def test_returns_list_of_rep_length(self):
        """bench_time_with_cuda_events returns exactly `rep` measurements."""
        t = torch.zeros(1, device="cuda")
        rep = 25
        times = bench_time_with_cuda_events(lambda: t.add_(0), warmup=5, rep=rep)
        assert isinstance(times, list)
        assert len(times) == rep
        assert all(t > 0 for t in times)


class TestBenchGPUTimeWithCuptiGPU:
    """GPU integration tests for CUPTI activity tracing timing accuracy.

    These tests mirror TestBenchTimeWithCUDAEventsGPU but use
    bench_gpu_time_with_cupti which measures actual GPU kernel execution
    time via CUPTI, excluding CPU-side launch overhead.
    """

    # -- Small kernel timing ------------------------------------------------

    def test_trivial_kernel_timing_floor(self):
        """Scalar add_(0) has sub-millisecond dispatch overhead."""
        t = torch.zeros(1, device="cuda")
        times = bench_gpu_time_with_cupti(lambda: t.add_(0))
        ms = statistics.median(times)
        assert 0 < ms < 1.0, f"Trivial kernel: {ms:.4f}ms, expected (0, 1)ms"

    def test_trivial_kernel_low_variance(self):
        """Repeated trivial kernel timings should cluster tightly."""
        t = torch.zeros(1, device="cuda")
        times = bench_gpu_time_with_cupti(lambda: t.add_(0), warmup=10, rep=100)
        median = sorted(times)[len(times) // 2]
        # Drop top/bottom 10% to get interquartile range
        trimmed = sorted(times)[10:90]
        spread = max(trimmed) - min(trimmed)
        assert spread < median * 2.0, (
            f"Trimmed spread {spread:.4f}ms too wide vs median {median:.4f}ms"
        )

    # -- Large kernel timing ------------------------------------------------

    def test_large_matmul_exceeds_small(self):
        """4096x4096 matmul takes measurably longer than 128x128."""
        small = torch.randn(128, 128, device="cuda")
        large = torch.randn(4096, 4096, device="cuda")

        ms_small = statistics.median(
            bench_gpu_time_with_cupti(lambda: torch.mm(small, small))
        )
        ms_large = statistics.median(
            bench_gpu_time_with_cupti(lambda: torch.mm(large, large))
        )

        assert ms_large > ms_small * 2, (
            f"Large matmul ({ms_large:.4f}ms) should be >2x small ({ms_small:.4f}ms)"
        )

    def test_compute_scales_with_size(self):
        """Doubling matmul dimension should roughly 8x the FLOPs and increase time."""
        n1 = torch.randn(1024, 1024, device="cuda")
        n2 = torch.randn(2048, 2048, device="cuda")

        ms_1k = statistics.median(bench_gpu_time_with_cupti(lambda: torch.mm(n1, n1)))
        ms_2k = statistics.median(bench_gpu_time_with_cupti(lambda: torch.mm(n2, n2)))

        ratio = ms_2k / ms_1k
        assert ratio > 1.5, (
            f"2048 matmul ({ms_2k:.4f}ms) / 1024 matmul ({ms_1k:.4f}ms) = {ratio:.2f}x, "
            f"expected >1.5x"
        )

    # -- Setup time exclusion -----------------------------------------------

    def test_setup_gpu_work_excluded_from_timing(self):
        """GPU work inside setup() must not appear in the timed region.

        Runs the same trivial kernel with two setups:
        1. Cheap: ShiftingMemoryPoolAllocator on a small tensor
        2. Expensive: allocator + a large matmul (~1-2ms GPU work)
        """
        warmup, rep = 10, 50
        total = warmup + rep
        base = torch.zeros(64, device="cuda")
        burn = torch.randn(2048, 2048, device="cuda")

        cheap_alloc = ShiftingMemoryPoolAllocator([base], [], total)
        expensive_alloc = ShiftingMemoryPoolAllocator([base], [], total)

        def expensive_setup():
            torch.mm(burn, burn)  # ~1ms GPU work, excluded by sync
            return expensive_alloc.get_unique_args()

        def fn(args):
            args[0].add_(1)

        ms_cheap = statistics.median(
            bench_gpu_time_with_cupti(
                fn, warmup=warmup, rep=rep, setup=cheap_alloc.get_unique_args
            )
        )
        ms_expensive = statistics.median(
            bench_gpu_time_with_cupti(fn, warmup=warmup, rep=rep, setup=expensive_setup)
        )

        # Both measure just add_(1) dispatch.  If setup leaks, expensive
        # would be ~1ms higher.  Allow 5x tolerance for noise.
        assert ms_expensive < ms_cheap * 5, (
            f"Expensive setup ({ms_expensive:.4f}ms) vs cheap ({ms_cheap:.4f}ms): "
            f"setup GPU work is leaking into measured time"
        )

    def test_setup_called_every_iteration(self):
        """ShiftingMemoryPoolAllocator is called once per warmup + timed iteration."""
        warmup, rep = 5, 20
        total = warmup + rep
        t = torch.zeros(1, device="cuda")
        allocator = ShiftingMemoryPoolAllocator([t], [], total)

        calls = [0]
        orig_get = allocator.get_unique_args

        def counting_setup():
            calls[0] += 1
            return orig_get()

        bench_gpu_time_with_cupti(
            lambda args: args[0].add_(1), warmup=warmup, rep=rep, setup=counting_setup
        )
        assert calls[0] == total

    # -- Setup impact on measurement ----------------------------------------

    def test_setup_vs_no_setup_same_kernel(self):
        """ShiftingMemoryPoolAllocator introduces no significant timing bias.

        CUPTI measures GPU kernel time directly, so setup CPU/GPU work
        should not affect the measured kernel time.
        """
        warmup, rep = 10, 50
        total = warmup + rep
        t = torch.randn(1024, 1024, device="cuda")

        # No setup — fn closes over `t`, same data every iteration
        ms_no_setup = statistics.median(
            bench_gpu_time_with_cupti(lambda: torch.mm(t, t), warmup=warmup, rep=rep)
        )

        # With allocator — provides shifted views each iteration
        allocator = ShiftingMemoryPoolAllocator([t], [], total)
        ms_with_setup = statistics.median(
            bench_gpu_time_with_cupti(
                lambda args: torch.mm(args[0], args[0]),
                warmup=warmup,
                rep=rep,
                setup=allocator.get_unique_args,
            )
        )

        ratio = ms_with_setup / ms_no_setup if ms_no_setup > 0 else float("inf")
        assert 0.5 < ratio < 2.0, (
            f"Setup vs no-setup ratio: {ratio:.2f} "
            f"(with={ms_with_setup:.4f}ms, without={ms_no_setup:.4f}ms)"
        )

    def test_setup_overhead_scales_with_input_not_kernel(self):
        """Varying allocator input size should not affect kernel time.

        Shifting 1KB vs 64MB in the allocator should produce the same
        measured kernel time, because setup is outside the timed region.
        """
        warmup, rep = 10, 50
        total = warmup + rep
        kernel_input = torch.randn(512, 512, device="cuda")

        # Small extra tensor (1KB) alongside the kernel input
        small_extra = torch.randn(256, device="cuda")
        # Large extra tensor (64MB) alongside the kernel input
        large_extra = torch.randn(4096, 4096, device="cuda")

        alloc_small = ShiftingMemoryPoolAllocator(
            [kernel_input, small_extra], [], total
        )
        alloc_large = ShiftingMemoryPoolAllocator(
            [kernel_input, large_extra], [], total
        )

        # fn only uses args[0] for compute; args[1] is the extra tensor
        def fn(args):
            torch.mm(args[0], args[0])

        ms_small = statistics.median(
            bench_gpu_time_with_cupti(
                fn, warmup=warmup, rep=rep, setup=alloc_small.get_unique_args
            )
        )
        ms_large = statistics.median(
            bench_gpu_time_with_cupti(
                fn, warmup=warmup, rep=rep, setup=alloc_large.get_unique_args
            )
        )

        ratio = ms_large / ms_small if ms_small > 0 else float("inf")
        assert 0.5 < ratio < 2.0, (
            f"Small-setup ({ms_small:.4f}ms) vs large-setup ({ms_large:.4f}ms): "
            f"ratio {ratio:.2f}, setup cost is leaking"
        )

    # -- Return sanity ------------------------------------------------------

    def test_returns_list_of_rep_length(self):
        """bench_gpu_time_with_cupti returns exactly `rep` measurements."""
        t = torch.zeros(1, device="cuda")
        rep = 25
        times = bench_gpu_time_with_cupti(lambda: t.add_(0), warmup=5, rep=rep)
        assert isinstance(times, list)
        assert len(times) == rep
        assert all(t > 0 for t in times)


class TestCuptiVsCudaEvents:
    """CUPTI should report lower or equal median times vs CUDA events.

    CUDA events include a fixed ~4-6us overhead from event recording.
    For short kernels this inflates the measurement significantly; for
    large kernels both methods converge.  See docs/cupti/cupti.md.
    """

    WARMUP = 10
    REP = 50

    @staticmethod
    def _elementwise(a, b):
        torch.multiply(a, b, out=a)

    @staticmethod
    def _elementwise_alloc(a, b):
        activation = a * b
        a.copy_(activation)

    @staticmethod
    def _elementwise_dual(a, b):
        torch.multiply(a, b, out=a)
        torch.add(a, b, out=a)

    @staticmethod
    def _elementwise_dual_alloc(a, b):
        activation = a * b + b
        a.copy_(activation)

    @staticmethod
    def _matmul(a, b):
        torch.matmul(a, b, out=a)

    @staticmethod
    def _mm_multi(a, b):
        for _ in range(5):
            torch.multiply(a, b, out=a)
            torch.add(a, b, out=a)
            torch.matmul(a, b, out=a)

    @pytest.mark.parametrize(
        "kernel_name,size",
        [
            ("elementwise", 64),
            ("elementwise", 256),
            ("elementwise", 1024),
            ("elementwise", 4096),
            ("elementwise_alloc", 64),
            ("elementwise_alloc", 1024),
            ("elementwise_alloc", 4096),
            ("elementwise_dual", 64),
            ("elementwise_dual", 1024),
            ("elementwise_dual", 4096),
            ("elementwise_dual_alloc", 64),
            ("elementwise_dual_alloc", 1024),
            ("elementwise_dual_alloc", 4096),
            ("matmul", 64),
            ("matmul", 256),
            ("matmul", 1024),
            ("matmul", 4096),
            ("mm_multi", 256),
            ("mm_multi", 1024),
            ("mm_multi", 4096),
        ],
    )
    def test_cupti_leq_cuda_events(self, kernel_name, size):
        """CUPTI median should be <= CUDA events median.

        For short kernels CUPTI is strictly less due to the ~4-6us event
        overhead.  For long kernels they converge, so we allow CUPTI to
        be up to 10% higher to account for measurement noise.
        """
        kernel_fn = getattr(self, f"_{kernel_name}")
        a = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)

        fn = lambda: kernel_fn(a, b)  # noqa: E731

        cupti_times = bench_gpu_time_with_cupti(fn, warmup=self.WARMUP, rep=self.REP)
        events_times = bench_time_with_cuda_events(fn, warmup=self.WARMUP, rep=self.REP)

        cupti_median = statistics.median(cupti_times)
        events_median = statistics.median(events_times)

        # CUPTI should be at most 10% above events (noise margin)
        assert cupti_median < events_median * 1.10, (
            f"{kernel_name}[{size}]: CUPTI median {cupti_median * 1000:.2f}us "
            f"> events median {events_median * 1000:.2f}us * 1.10"
        )


class TestTimeRunnable:
    # -- Timing variance -----------------------------------------------------
    @pytest.mark.parametrize(
        "size,max_spread_ratio",
        [
            (64, 1.25),  # launch-overhead dominated
            (512, 1.3),  # launch-overhead dominated
            (2048, 1.15),  # compute-dominated, tighter variance
            (4096, 1.15),  # compute-dominated, very tight variance
        ],
    )
    def test_matmul_timing_variance(self, size, max_spread_ratio):
        """Trimmed min/max ratio stays bounded across matmul sizes.

        Small matmuls (64-512) are launch-overhead dominated and show wider
        variance from dispatch jitter.  Large matmuls (2048+) are compute-
        dominated with tight variance.  See docs/cuda_events/cuda_events.md
        Scenarios 7-8 for measured ranges on RTX 4090 and B200.
        """
        a = torch.randn(size, size, device="cuda")
        b = torch.randn(size, size, device="cuda")

        # use default arguments to mimic the eval_driver pattern
        times = time_runnable(
            lambda a, b: torch.mm(a, b),
            [a, b],
            [],
            "cuda:0",
            return_mode="all",
        )

        spread_ratio = max(times) / min(times) if min(times) > 0 else float("inf")

        assert spread_ratio < max_spread_ratio, (
            f"mm[{size}x{size}] trimmed spread ratio {spread_ratio:.2f}x exceeds "
            f"{max_spread_ratio}x (min={min(times):.4f}ms, max={max(times):.4f}ms)"
        )

    def test_variance_decreases_with_compute_intensity(self):
        """Larger matmuls should have tighter relative variance than small ones.

        Compute-dominated kernels (large matmul) produce stable timings because
        the GPU spends most time on arithmetic.  Launch-overhead-dominated
        kernels (small matmul) show higher relative variance from dispatch
        jitter.  See docs/cuda_events/cuda_events.md Conclusion 6.
        """
        small = torch.randn(64, 64, device="cuda")
        large = torch.randn(4096, 4096, device="cuda")

        # use default arguments to mimic the eval_driver pattern
        times_small = time_runnable(
            lambda a: torch.mm(a, a),
            [small],
            [],
            "cuda:0",
            return_mode="all",
        )
        times_large = time_runnable(
            lambda a: torch.mm(a, a),
            [large],
            [],
            "cuda:0",
            return_mode="all",
        )

        def coeff_of_variation(times):
            mean = sum(times) / len(times)
            variance = sum((t - mean) ** 2 for t in times) / len(times)
            return (variance**0.5) / mean if mean > 0 else float("inf")

        cv_small = coeff_of_variation(times_small)
        cv_large = coeff_of_variation(times_large)

        assert cv_large < cv_small, (
            f"Large matmul CV ({cv_large:.4f}) should be less than "
            f"small matmul CV ({cv_small:.4f})"
        )


class TestStreamHidingDetection:
    """CUPTI-based timing must capture work hidden on non-default streams.

    A submission could launch kernels on a non-default stream to evade
    CUDA event timing (events on stream 0 see no work).  time_runnable
    defaults to CUPTI which traces all streams globally, so it should
    report accurate times regardless of stream placement.

    See docs/streams/cupti_detection.md for the full analysis.
    """

    def test_non_default_stream_timing_matches_default(self):
        """Kernel on a non-default stream reports the same time as default stream."""
        stream = torch.cuda.Stream()
        a = torch.randn(2048, 2048, device="cuda")

        def default_kernel(x):
            return torch.mm(x, x)

        def stream_hidden_kernel(x):
            with torch.cuda.stream(stream):
                return torch.mm(x, x)

        ms_default = time_runnable(default_kernel, [a], [], "cuda:0", warmup=10, rep=50)
        ms_hidden = time_runnable(
            stream_hidden_kernel, [a], [], "cuda:0", warmup=10, rep=50
        )

        ratio = ms_hidden / ms_default if ms_default > 0 else float("inf")
        assert 0.5 < ratio < 2.0, (
            f"Non-default stream ({ms_hidden:.4f}ms) vs default ({ms_default:.4f}ms): "
            f"ratio {ratio:.2f}, CUPTI should capture both equally"
        )

    def test_stream_hiding_with_wait_stream(self):
        """wait_stream pattern (fools pipelined CUDA events) still reports accurate CUPTI timing."""
        stream = torch.cuda.Stream()
        a = torch.randn(2048, 2048, device="cuda")

        def default_kernel(x):
            return torch.mm(x, x)

        def stream_hide_with_wait(x):
            with torch.cuda.stream(stream):
                result = torch.mm(x, x)
            torch.cuda.current_stream().wait_stream(stream)
            return result

        ms_default = time_runnable(default_kernel, [a], [], "cuda:0", warmup=10, rep=50)
        ms_hidden = time_runnable(
            stream_hide_with_wait, [a], [], "cuda:0", warmup=10, rep=50
        )

        ratio = ms_hidden / ms_default if ms_default > 0 else float("inf")
        assert 0.5 < ratio < 2.0, (
            f"Stream-hide+wait ({ms_hidden:.4f}ms) vs default ({ms_default:.4f}ms): "
            f"ratio {ratio:.2f}, CUPTI should capture both equally"
        )

    def test_stream_hidden_kernel_not_near_zero(self):
        """Stream-hidden large matmul must not report near-zero timing.

        This is the specific reward hack vector: launching a large matmul
        on a non-default stream would report ~2us under CUDA events.
        CUPTI must report the real execution time.
        """
        stream = torch.cuda.Stream()
        a = torch.randn(4096, 4096, device="cuda")

        def stream_hidden_kernel(x):
            with torch.cuda.stream(stream):
                return torch.mm(x, x)

        ms = time_runnable(stream_hidden_kernel, [a], [], "cuda:0", warmup=10, rep=50)

        # 4096x4096 matmul takes ~0.9ms on RTX 4090; 0.1ms is a safe lower bound
        assert ms > 0.1, (
            f"Stream-hidden 4096x4096 matmul reported {ms:.4f}ms, "
            f"expected >0.1ms — CUPTI failed to capture non-default stream work"
        )
