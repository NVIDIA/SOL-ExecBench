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

from __future__ import annotations

import bisect
import statistics
from collections.abc import Callable
from typing import Any, Literal, Union

import torch
from cuda.bindings import runtime as cuda_runtime
from cupti import cupti

from sol_execbench.core.bench.cupti_utils import (
    CuptiKernelInfo,
    collect_cupti_activities,
    kernel_activity_counts,
    kernel_activity_sequence,
    select_activity_sequence,
)
from sol_execbench.core.bench.io import ShiftingMemoryPoolAllocator


GPU_TIMING_ACTIVITY_KINDS = (
    cupti.ActivityKind.CONCURRENT_KERNEL,
    cupti.ActivityKind.MEMCPY,
    cupti.ActivityKind.MEMSET,
)
DEFAULT_CUPTI_CLOCK_SKEW_TOLERANCE_NS = 5_000_000
DEFAULT_CUPTI_CLOCK_DRIFT_TOLERANCE_NS = 50_000


def get_l2_cache_size(device) -> int:
    """
    Get L2 cache size in bytes for the given CUDA device.

    Args:
        device: CUDA device (int, torch.device, or None for current device).

    Returns:
        L2 cache size in bytes.
    """
    props = torch.cuda.get_device_properties(device)
    return props.L2_cache_size


def _summarize_statistics(
    times: list[float],
    return_mode: Literal["mean", "median", "all"],
) -> Union[float, list[float]]:
    """Summarize timing statistics based on return mode."""
    if return_mode == "all":
        return times
    elif return_mode == "mean":
        return statistics.mean(times)
    elif return_mode == "median":
        return statistics.median(times)
    raise ValueError(f"Unknown return_mode: {return_mode}")


def _get_empty_cache_for_benchmark(device) -> torch.Tensor:
    """Create a 256 MB buffer for clearing L2 cache before benchmark runs."""
    # Double the L2 cache size just to be safe
    cache_size = get_l2_cache_size(device) * 2
    return torch.empty(int(cache_size), dtype=torch.int8, device=device)


def _clear_cache(cache: torch.Tensor) -> None:
    """Clear the cache buffer by zeroing it."""
    cache.zero_()


def _reset_persisting_l2_cache(
    device: int | str | torch.device | None = None,
) -> None:
    """Reset persisting L2 cache lines to normal status."""

    def reset_current_device() -> None:
        result = cuda_runtime.cudaCtxResetPersistingL2Cache()
        if isinstance(result, tuple):
            result = result[0]
        torch.cuda.check_error(int(result))

    if device is not None:
        with torch.cuda.device(device):
            reset_current_device()
        return

    reset_current_device()


def clone_args(args: list[Any]) -> list[Any]:
    """Clone tensor arguments to prevent cross-iteration data contamination.

    Returns fresh copies of all tensor arguments so each benchmark iteration
    starts with independent data.  Non-tensor arguments are passed through.
    """
    return [arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args]


def _has_consistent_clock_offset(
    aligned: list[list[CuptiKernelInfo]],
    iter_timestamps: list[tuple[float, float]],
    *,
    max_offset_ns: float,
    max_drift_ns: float,
) -> bool:
    """Verify all aligned iterations share a bounded GPU-to-CPU clock offset.

    For iteration ``i``, every kernel start falls inside
    ``[start_cpu_i + delta_i, end_cpu_i + delta_i]`` iff
    ``lower_i <= delta_i <= upper_i`` where
    ``lower_i = max(k.start - end_cpu_i)`` and
    ``upper_i = min(k.start - start_cpu_i)``.
    Requiring one global offset within ``max_offset_ns`` and per-run drift
    within ``max_drift_ns`` accommodates real CPU/GPU clock skew while
    rejecting back-to-back launches in one iteration that skip the next.
    """
    lowers = [
        max(k.start - end_cpu for k in iter_kernels)
        for iter_kernels, (_, end_cpu) in zip(aligned, iter_timestamps)
    ]
    uppers = [
        min(k.start - start_cpu for k in iter_kernels)
        for iter_kernels, (start_cpu, _) in zip(aligned, iter_timestamps)
    ]
    min_shared_offset = max(lowers)
    max_shared_offset = min(uppers)
    if min_shared_offset - max_shared_offset > 2 * max_drift_ns:
        return False
    midpoint_offset = 0.5 * (min_shared_offset + max_shared_offset)
    return abs(midpoint_offset) <= max_offset_ns


def _attribute_cupti_iterations(
    sorted_kernels: list[CuptiKernelInfo],
    iter_timestamps: list[tuple[float, float]],
    expected_kernel_names: list[str],
    *,
    clock_skew_tolerance_ns: float = DEFAULT_CUPTI_CLOCK_SKEW_TOLERANCE_NS,
    clock_drift_tolerance_ns: float = DEFAULT_CUPTI_CLOCK_DRIFT_TOLERANCE_NS,
) -> list[list[CuptiKernelInfo]]:
    """Attribute CUPTI GPU activities to each synchronized timed iteration.

    Each timed iteration ends with ``torch.cuda.synchronize()``, so GPU
    activities from iteration ``i`` are strictly ordered after iteration
    ``i - 1``. Because ``cupti.get_timestamp()`` is sampled in the CPU clock
    domain while ``CONCURRENT_KERNEL`` timestamps use CUPTI's GPU-to-CPU clock
    correlation, small clock-offset skew on virtualized hosts can place an
    iteration's kernel start slightly outside ``[start_cpu, end_cpu]``.

    Pass 1 attempts exact ``[start_cpu, end_cpu]`` window matching with
    monotonic consumption. If any iteration fails due to clock-domain skew,
    Pass 2 reconciles the synchronized sequence within
    ``clock_skew_tolerance_ns`` and verifies consistent clock-offset drift
    while ensuring every iteration claims a disjoint, monotonically advancing
    activity sequence.
    """
    kernel_starts = [kernel.start for kernel in sorted_kernels]
    kernel_to_index = {id(kernel): i for i, kernel in enumerate(sorted_kernels)}

    saved_strict_error: ValueError | None = None
    try:
        matched: list[list[CuptiKernelInfo]] = []
        min_idx = 0
        for idx, (start_cpu, end_cpu) in enumerate(iter_timestamps):
            left_idx = max(min_idx, bisect.bisect_left(kernel_starts, start_cpu))
            right_idx = max(left_idx, bisect.bisect_right(kernel_starts, end_cpu))
            iter_kernels = select_activity_sequence(
                sorted_kernels[left_idx:right_idx],
                expected_kernel_names,
                iteration=idx,
            )
            matched.append(iter_kernels)
            min_idx = max(kernel_to_index[id(k)] for k in iter_kernels) + 1
        return matched
    except ValueError as strict_error:
        if clock_skew_tolerance_ns <= 0:
            raise
        saved_strict_error = strict_error

    expected_name_set = set(expected_kernel_names)
    expected_count = len(expected_kernel_names)
    candidate_kernels = [
        kernel
        for kernel in sorted_kernels
        if kernel.kernel_string() in expected_name_set
    ]
    if len(candidate_kernels) == len(iter_timestamps) * expected_count:
        aligned: list[list[CuptiKernelInfo]] = []
        valid_alignment = True
        for idx, (start_cpu, end_cpu) in enumerate(iter_timestamps):
            chunk = candidate_kernels[
                idx * expected_count : (idx + 1) * expected_count
            ]
            try:
                iter_kernels = select_activity_sequence(
                    chunk,
                    expected_kernel_names,
                    iteration=idx,
                )
            except ValueError:
                valid_alignment = False
                break
            skew_ns = max(
                max(0.0, start_cpu - k.start, k.start - end_cpu) for k in iter_kernels
            )
            if skew_ns > clock_skew_tolerance_ns:
                valid_alignment = False
                break
            aligned.append(iter_kernels)
        if valid_alignment and _has_consistent_clock_offset(
            aligned,
            iter_timestamps,
            max_offset_ns=clock_skew_tolerance_ns,
            max_drift_ns=clock_drift_tolerance_ns,
        ):
            return aligned

    try:
        matched = []
        min_idx = 0
        for idx, (start_cpu, end_cpu) in enumerate(iter_timestamps):
            left_idx = max(
                min_idx,
                bisect.bisect_left(kernel_starts, start_cpu - clock_skew_tolerance_ns),
            )
            right_idx = max(
                left_idx,
                bisect.bisect_right(kernel_starts, end_cpu + clock_skew_tolerance_ns),
            )
            iter_kernels = select_activity_sequence(
                sorted_kernels[left_idx:right_idx],
                expected_kernel_names,
                iteration=idx,
                target_window=(start_cpu, end_cpu),
            )
            matched.append(iter_kernels)
            min_idx = max(kernel_to_index[id(k)] for k in iter_kernels) + 1
        if _has_consistent_clock_offset(
            matched,
            iter_timestamps,
            max_offset_ns=clock_skew_tolerance_ns,
            max_drift_ns=clock_drift_tolerance_ns,
        ):
            return matched
    except ValueError:
        pass

    assert saved_strict_error is not None
    raise saved_strict_error from None


def bench_gpu_time_with_cupti(
    fn: Callable,
    warmup: int = 10,
    rep: int = 100,
    setup: Callable[[], Any] | None = None,
    cold_l2_cache: bool = True,
    device="cuda",
):
    """Benchmark GPU time using the discovered user CUPTI activity sequence.

    Setup and cache-management work is excluded by discovering the user-call
    sequence after warmup, then selecting only that sequence from every timed
    iteration. The end timestamp is captured after synchronization so delayed
    or non-default-stream work remains inside the attribution window.
    """
    if setup is None:
        _fn = fn

        def fn(_):
            return _fn()

        def setup():
            return None

    buffer = None
    if cold_l2_cache:
        buffer = _get_empty_cache_for_benchmark(device)

    # Prepare runner (either direct fn or CUDA graph replay)
    runner: Callable = fn
    runner_args: Callable = setup

    def prepare_iteration(*, synchronize: bool = True):
        args = runner_args()
        if cold_l2_cache:
            _reset_persisting_l2_cache(device)
            _clear_cache(buffer)
        if synchronize:
            torch.cuda.synchronize()
        return args

    torch.cuda.synchronize()
    for _ in range(warmup):
        args = prepare_iteration()
        runner(args)
    torch.cuda.synchronize()

    # Discover the user-call GPU activity sequence after setup/cache work drains.
    args = prepare_iteration()
    with collect_cupti_activities(
        activity_kinds=GPU_TIMING_ACTIVITY_KINDS
    ) as discovery_buffers:
        runner(args)
        torch.cuda.synchronize()
    expected_kernels = sorted(
        discovery_buffers.kernels,
        key=lambda kernel: (kernel.start, kernel.end, kernel.correlation_id),
    )
    expected_kernel_names = kernel_activity_sequence(expected_kernels)
    if not expected_kernel_names:
        raise ValueError("No kernel activities recorded during discovery iteration")
    expected_kernel_counts = kernel_activity_counts(expected_kernels)

    iter_timestamps = []
    with collect_cupti_activities(
        activity_kinds=GPU_TIMING_ACTIVITY_KINDS
    ) as cupti_buffers:
        torch.cuda.synchronize()
        for _ in range(rep):
            args = prepare_iteration(synchronize=False)
            start_cpu = cupti.get_timestamp()
            runner(args)
            torch.cuda.synchronize()
            end_cpu = cupti.get_timestamp()
            iter_timestamps.append((start_cpu, end_cpu))
        torch.cuda.synchronize()

    sorted_kernels = sorted(
        cupti_buffers.kernels,
        key=lambda kernel: (kernel.start, kernel.end, kernel.correlation_id),
    )
    iter_kernel_sequences = _attribute_cupti_iterations(
        sorted_kernels,
        iter_timestamps,
        expected_kernel_names,
    )
    measured_times = []
    for iter_kernels in iter_kernel_sequences:
        assert kernel_activity_counts(iter_kernels) == expected_kernel_counts
        min_start = min(k.start for k in iter_kernels)
        max_end = max(k.end for k in iter_kernels)
        measured_times.append((max_end - min_start) / 1e6)
    return measured_times


def bench_time_with_cuda_events(
    fn: Callable[..., Any],
    warmup: int = 10,
    rep: int = 100,
    setup: Callable[[], Any] | None = None,
    device: str = "cuda",
) -> Union[float, list[float]]:
    """Benchmark the runtime of the provided function.

    Derived from triton.testing.do_bench (MIT licence), with fixes from
    sol-bench: explicit synchronization before each start event, L2 cache
    clearing during warmup, and a setup callback for argument cloning.

    Parameters
    ----------
    fn : Callable[..., Any]
        The function to benchmark.  If *setup* is provided, *fn* receives
        the return value of *setup* as its sole argument.
    warmup : int
        Number of warmup iterations (default: 10).
    rep : int
        Number of timed iterations (default: 100).
    setup : Callable[[], Any] | None
        Called before each timed iteration; its return value is passed to *fn*.
        Setup time is **not** included in measurements. This method should only enqueue operations onto the default stream and should not explcitly synchronize.
    device : str
        CUDA device for cache-clearing buffer (default: ``"cuda"``).

    Returns
    -------
    float | list[float]
        Benchmark result(s) in milliseconds.
    """
    cache = _get_empty_cache_for_benchmark(device)
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    torch.cuda.synchronize()

    if setup is None:
        _fn = fn

        def fn(_):
            return _fn()

        def setup():
            return None

    for _ in range(warmup):
        args = setup()
        # always clear cache after setup to prevent data residing in L2
        _reset_persisting_l2_cache(device)
        _clear_cache(cache)
        fn(args)

    # Timed iterations.
    # Avoid synchronizations after warmup and in this hot loop
    # to keep the driver's GPU queue full.
    for i in range(rep):
        args = setup()
        _reset_persisting_l2_cache(device)
        _clear_cache(cache)
        start_events[i].record()
        fn(args)
        end_events[i].record()

    torch.cuda.synchronize()
    measured_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return measured_times


def time_runnable(
    fn: Any,
    inputs: list,
    outputs: list,
    device: str,
    warmup: int = 10,
    rep: int = 100,
    return_mode: Literal["mean", "median", "all"] = "median",
    methodology: Literal["cuda_events", "cupti"] = "cupti",
    seed: int = 0,
) -> Union[float, list[float]]:
    """Time the execution of a callable using CUDA events.

    Creates a :class:`ShiftingMemoryPoolAllocator` from *inputs* and *outputs*
    so each timed iteration receives arguments with a unique ``data_ptr``.
    Allocator setup time is excluded from measurements. Crucially, the allocator
    pre-allocates all tensors before the benchmark loop, so the timed region
    is not affected by cudaMalloc times (which increase measured kernel time by 300%).

    Parameters
    ----------
    fn : callable
        The function to benchmark.  Receives unpacked arguments each iteration.
    inputs : list
        Input tensors/scalars as returned by :func:`gen_inputs`.
    outputs : list
        Pre-allocated output tensors for DPS kernels (from
        :func:`allocate_outputs`), or an empty list for non-DPS kernels.
    device : str
        The CUDA device to run the benchmark on (e.g. ``"cuda:0"``).
    warmup : int
        Number of warmup iterations (default: 10).
    rep : int
        Number of timed iterations (default: 100).
    return_mode : {"mean", "median", "all"}
        How to summarize the timing results (default: ``"median"``).
    methodology : {"cuda_events", "cupti"}
        The methodology to use for timing (default: ``"cupti"``). CUPTI is used by nsys to measure the actual GPU kernel execution time, excluding CPU-side launch overhead.
    seed : int
        Seed for the allocator's randomized pointer-shift sequence.

    Returns
    -------
    float | list[float]
        Benchmark result(s) in milliseconds.
    """
    total_iterations = warmup + rep
    with torch.cuda.device(device):
        if methodology == "cuda_events":
            allocator = ShiftingMemoryPoolAllocator(
                inputs, outputs, total_iterations, seed=seed
            )
            try:
                times = bench_time_with_cuda_events(
                    fn=lambda args: fn(*args),
                    warmup=warmup,
                    rep=rep,
                    setup=allocator.get_unique_args,
                    device=device,
                )
            finally:
                del allocator
        elif methodology == "cupti":
            allocator = ShiftingMemoryPoolAllocator(
                inputs, outputs, total_iterations + 1, seed=seed
            )
            try:
                times = bench_gpu_time_with_cupti(
                    fn=lambda args: fn(*args),
                    warmup=warmup,
                    rep=rep,
                    setup=allocator.get_unique_args,
                    device=device,
                )
            finally:
                del allocator
        else:
            raise ValueError(f"Unknown methodology: {methodology}")
        if not times:
            raise ValueError(f"No timing results for methodology: {methodology}")
        return _summarize_statistics(times, return_mode)
