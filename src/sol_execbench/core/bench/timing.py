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
import ctypes
import ctypes.util
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Union

import torch
from cupti import cupti

from sol_execbench.core.bench.io import ShiftingMemoryPoolAllocator


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


def clone_args(args: list[Any]) -> list[Any]:
    """Clone tensor arguments to prevent cross-iteration data contamination.

    Returns fresh copies of all tensor arguments so each benchmark iteration
    starts with independent data.  Non-tensor arguments are passed through.
    """
    return [arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args]


_libstdcxx = ctypes.CDLL(ctypes.util.find_library("stdc++"))
_libstdcxx.__cxa_demangle.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_int),
]
_libstdcxx.__cxa_demangle.restype = ctypes.c_char_p


def _demangle(name: str) -> str:
    status = ctypes.c_int()
    result = _libstdcxx.__cxa_demangle(name.encode(), None, None, ctypes.byref(status))
    if status.value != 0:
        return name
    return result.decode().replace(" >", ">")


@dataclass(frozen=True)
class CuptiKernelInfo:
    name: str
    start: float
    end: float
    correlation_id: int
    copy_kind: int
    bytes: int
    value: int
    kind: cupti.ActivityKind
    _activity: Any

    @classmethod
    def from_activity(cls, activity):
        return cls(
            name=cls.set_kernel_name(activity),
            start=activity.start,
            end=activity.end,
            correlation_id=activity.correlation_id,
            copy_kind=cls.get_copy_kind(activity),
            bytes=cls.get_bytes(activity),
            value=cls.get_value(activity),
            kind=activity.kind,
            _activity=activity,
        )

    @staticmethod
    def set_kernel_name(activity):
        if activity.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
            return _demangle(activity.name)
        elif activity.kind == cupti.ActivityKind.MEMCPY:
            return "MEMCPY"
        elif activity.kind == cupti.ActivityKind.MEMSET:
            return "MEMSET"

    @staticmethod
    def get_bytes(activity):
        if activity.kind in (cupti.ActivityKind.MEMCPY, cupti.ActivityKind.MEMSET):
            return activity.bytes
        else:
            return 0

    @staticmethod
    def get_copy_kind(activity):
        if activity.kind == cupti.ActivityKind.MEMCPY:
            return activity.copy_kind
        else:
            return 0

    @staticmethod
    def get_value(activity):
        if activity.kind == cupti.ActivityKind.MEMSET:
            return activity.value
        else:
            return 0


def bench_gpu_time_with_cupti(
    fn: Callable,
    warmup: int = 10,
    rep: int = 100,
    setup: Callable[[], Any] | None = None,
    cold_l2_cache: bool = True,
    device="cuda",
):
    """
    Benchmark GPU time using CUPTI activity tracing for precise kernel timing.

    CUPTI (CUDA Profiling Tools Interface) provides hardware-level profiling that
    measures actual GPU kernel execution time, excluding CPU-side launch overhead.
    This gives the most accurate kernel performance measurements.

    Cold L2 cache is achieved via L2 flush between iterations. CUPTI measures
    per-iteration, so L2 flush works correctly.

    Behavior:
    - Uses CUPTI (requires version >= 13, i.e., CUDA 13+) to trace kernel activities
      and compute per-iteration GPU time from recorded start/end timestamps.

    Args:
        fn (Callable): The kernel function to benchmark.
        warmup (int, optional): Number of warmup iterations (not timed).
            If None, computed from dry_run_time_ms.
        rep (int, optional): Number of measured iterations.
        setup (Callable, optional): Called before each timed iteration; its return value is passed to *fn*.
            Setup time is **not** included in measurements. This method should only enqueue operations onto the default stream and should not explcitly synchronize.
        cold_l2_cache (bool): If True, flush L2 cache before each iteration to
            ensure cold-cache performance measurements (default: True).

    Returns:
        List[float]: Per-iteration GPU kernel execution times in milliseconds.

    Example:
        Basic CUPTI benchmarking (requires cupti-python >= 13):

        >>> def my_kernel(a, b):
        ...     return torch.matmul(a, b.T)
        >>> q = torch.randn(1024, 128, device="cuda")
        >>> k = torch.randn(1024, 128, device="cuda")
        >>> times = bench_gpu_time_with_cupti(
        ...     fn=my_kernel,
        ...     input_args=(q, k),
        ... )
        >>> print(f"Median GPU time: {np.median(times):.3f} ms")
    """

    # CUPTI buffer callbacks
    def func_buffer_requested():
        buffer_size = 8 * 1024 * 1024
        max_num_records = 0
        return buffer_size, max_num_records

    def func_buffer_completed(
        launches: list[tuple[float, float, int, int, int]],
        kernels: list[CuptiKernelInfo],
        activities: list,
    ):
        for activity in activities:
            if activity.kind in (
                cupti.ActivityKind.CONCURRENT_KERNEL,
                cupti.ActivityKind.MEMCPY,
                cupti.ActivityKind.MEMSET,
            ):
                # Kernel activity
                kernels.append(CuptiKernelInfo.from_activity(activity))
            elif activity.kind in (
                cupti.ActivityKind.RUNTIME,
                cupti.ActivityKind.DRIVER,
            ):
                # Runtime or Driver activity
                launches.append(
                    (
                        activity.start,
                        activity.end,
                        activity.correlation_id,
                        activity.cbid,
                        activity.kind,
                    )
                )

    # Check if args are provided (determines how we call fn)
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

    # Dry runs
    torch.cuda.synchronize()
    for _ in range(warmup):
        args = runner_args()
        if cold_l2_cache:
            _clear_cache(buffer)
        runner(args)
    torch.cuda.synchronize()

    # CUPTI measurement
    launches: list[tuple[float, float, int, int, int]] = []
    kernels: list[CuptiKernelInfo] = []
    iter_timestamps = []
    cupti.activity_enable(cupti.ActivityKind.RUNTIME)
    cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)
    cupti.activity_enable(cupti.ActivityKind.DRIVER)
    cupti.activity_enable(cupti.ActivityKind.MEMCPY)
    cupti.activity_enable(cupti.ActivityKind.MEMSET)
    cupti.activity_register_callbacks(
        func_buffer_requested, partial(func_buffer_completed, launches, kernels)
    )
    torch.cuda.synchronize()
    for _ in range(rep):
        args = runner_args()
        if cold_l2_cache:
            _clear_cache(buffer)
        start_cpu = cupti.get_timestamp()
        runner(args)
        end_cpu = cupti.get_timestamp()
        # keep this synchronize here to ensure timestamps are consistent
        torch.cuda.synchronize()
        iter_timestamps.append((start_cpu, end_cpu))
    torch.cuda.synchronize()
    cupti.activity_flush_all(0)
    cupti.activity_disable(cupti.ActivityKind.RUNTIME)
    cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
    cupti.activity_disable(cupti.ActivityKind.DRIVER)
    cupti.activity_disable(cupti.ActivityKind.MEMCPY)
    cupti.activity_disable(cupti.ActivityKind.MEMSET)
    cupti.finalize()

    def generate_kernel_string(kernel: CuptiKernelInfo):
        # No start, end, correlation_id is considered in the kernel string
        return f"{kernel.name}_{kernel.copy_kind}_{kernel.bytes}_{kernel.value}_{kernel.kind}"

    # Process activities - OPTIMIZED O(N + M log M) algorithm
    # Step 1: Sort launches by start timestamp - O(M log M)
    sorted_launches = sorted(launches, key=lambda la: la[0])
    launch_starts = [la[0] for la in sorted_launches]

    # Step 2: Build correlation_id -> kernels mapping - O(K)
    corr_id_to_kernels: dict[int, list[CuptiKernelInfo]] = {}
    for k in kernels:
        corr_id = k.correlation_id
        if corr_id not in corr_id_to_kernels:
            corr_id_to_kernels[corr_id] = []
        corr_id_to_kernels[corr_id].append(k)

    measured_times = []
    kernel_names = None
    for idx, (start_cpu, end_cpu) in enumerate(iter_timestamps):
        # Use binary search to find launches within time range - O(log M)
        left_idx = bisect.bisect_left(launch_starts, start_cpu)
        right_idx = bisect.bisect_right(launch_starts, end_cpu)

        # Get correlation IDs for launches in range - O(range size)
        corr_ids = set(sorted_launches[i][2] for i in range(left_idx, right_idx))

        # Find all GPU kernels using the mapping - O(range size)
        iter_kernels: list[CuptiKernelInfo] = []
        for corr_id in corr_ids:
            if corr_id in corr_id_to_kernels:
                iter_kernels.extend(corr_id_to_kernels[corr_id])

        if not iter_kernels:
            raise ValueError(f"No kernel activities recorded for iteration {idx}")
        current_kernel_names = set(generate_kernel_string(k) for k in iter_kernels)
        # check if the kernel names are consistent
        if kernel_names is None:
            kernel_names = current_kernel_names
        else:
            if kernel_names != current_kernel_names:
                raise ValueError(
                    f"Inconsistent kernel names: {kernel_names} != {current_kernel_names}"
                )
        min_start = min(k.start for k in iter_kernels)
        max_end = max(k.end for k in iter_kernels)
        span_ms = (max_end - min_start) / 1e6  # ns to ms
        measured_times.append(span_ms)
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
        _clear_cache(cache)
        fn(args)

    # Timed iterations.
    # Avoid synchronizations after warmup and in this hot loop
    # to keep the driver's GPU queue full.
    for i in range(rep):
        args = setup()
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

    Returns
    -------
    float | list[float]
        Benchmark result(s) in milliseconds.
    """
    total_iterations = warmup + rep
    allocator = ShiftingMemoryPoolAllocator(inputs, outputs, total_iterations)
    with torch.cuda.device(device):
        if methodology == "cuda_events":
            times = bench_time_with_cuda_events(
                fn=lambda args: fn(*args),
                warmup=warmup,
                rep=rep,
                setup=allocator.get_unique_args,
                device=device,
            )
        elif methodology == "cupti":
            times = bench_gpu_time_with_cupti(
                fn=lambda args: fn(*args),
                warmup=warmup,
                rep=rep,
                setup=allocator.get_unique_args,
                device=device,
            )
        if not times:
            raise ValueError(f"No timing results for methodology: {methodology}")
        return _summarize_statistics(times, return_mode)
