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

# This file includes code derived from Triton (https://github.com/openai/triton)
#
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
Timing utilities for benchmarking FlashInfer-Bench kernel solutions.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from multiprocessing import Lock
from multiprocessing.synchronize import Lock as LockType
from typing import Any, Literal, Union

import torch

from .compile import Runnable

# Device-specific lock registry to ensure multiprocess-safe benchmarking
_device_locks: dict[str, LockType] = {}
_registry_lock = Lock()


def _device_lock(device: str) -> LockType:
    """Get or create a multiprocessing lock for the specified device.

    This function maintains a registry of locks per device to ensure that
    benchmarking operations on the same device are serialized, preventing
    interference between concurrent measurements.

    Parameters
    ----------
    device : str
        The device identifier (e.g., "cuda:0", "cuda:1").

    Returns
    -------
    LockType
        A lock object specific to the given device.
    """
    with _registry_lock:
        lock = _device_locks.get(device)
        if lock is None:
            lock = Lock()
            _device_locks[device] = lock
        return lock


def _quantile(a: list[float], q: Sequence[float]) -> list[float]:
    """Compute quantiles of a list of values."""
    n = len(a)
    a = sorted(a)

    def get_quantile(q: float) -> float:
        if not (0 <= q <= 1):
            raise ValueError("Quantiles must be in the range [0, 1]")
        point = q * (n - 1)
        lower = math.floor(point)
        upper = math.ceil(point)
        t = point - lower
        return (1 - t) * a[lower] + t * a[upper]

    return [get_quantile(qi) for qi in q]


def _summarize_statistics(
    times: list[float],
    quantiles: Sequence[float] | None,
    return_mode: Literal["min", "max", "mean", "median", "all"],
) -> Union[float, list[float]]:
    """Summarize timing statistics based on return mode."""
    if quantiles is not None:
        ret = _quantile(times, quantiles)
        if len(ret) == 1:
            return ret[0]
        return ret
    if return_mode == "all":
        return times
    elif return_mode == "min":
        return min(times)
    elif return_mode == "max":
        return max(times)
    elif return_mode == "mean":
        return statistics.mean(times)
    elif return_mode == "median":
        return statistics.median(times)
    raise ValueError(f"Unknown return_mode: {return_mode}")


def _get_empty_cache_for_benchmark(device: str = "cuda") -> torch.Tensor:
    """Create a buffer for clearing L2 cache before benchmark runs.

    We maintain a buffer of 256 MB that we clear before each kernel call
    to make sure that the L2 cache doesn't contain any input data before the run.

    Parameters
    ----------
    device : str
        The CUDA device to allocate the buffer on (default: "cuda").
    """
    cache_size = 256 * 1024 * 1024
    return torch.empty(int(cache_size // 4), dtype=torch.int, device=device)


def _clear_cache(cache: torch.Tensor) -> None:
    """Clear the cache buffer by zeroing it."""
    cache.zero_()


def do_bench(
    fn: Callable[..., Any],
    warmup: int = 10,
    rep: int = 100,
    grad_to_none: list[torch.Tensor] | None = None,
    quantiles: Sequence[float] | None = None,
    return_mode: Literal["min", "max", "mean", "median", "all"] = "mean",
    setup: Callable[[], Any] | None = None,
    device: str = "cuda",
) -> Union[float, list[float]]:
    """Benchmark the runtime of the provided function.

    Parameters
    ----------
    fn : Callable[..., Any]
        The function to benchmark. If setup is provided, fn receives setup's
        return value as its argument.
    warmup : int
        Number of warmup iterations (default: 10).
    rep : int
        Number of timed iterations (default: 100).
    grad_to_none : list[torch.Tensor] | None
        List of tensors whose gradients should be cleared before each run.
    quantiles : Sequence[float] | None
        If provided, return the specified quantiles instead of using return_mode.
    return_mode : Literal["min", "max", "mean", "median", "all"]
        How to summarize the timing results (default: "mean").
    setup : Callable[[], Any] | None
        Optional setup function called before each timed iteration. Its return
        value is passed to fn. Setup time is NOT included in measurements.
    device : str
        The CUDA device for cache clearing buffer (default: "cuda").

    Returns
    -------
    float | list[float]
        The benchmark result(s) in milliseconds.

    Notes
    -----
    Clock locking is handled at the time_runnable() level, not here.
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    cache = _get_empty_cache_for_benchmark(device)
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    # Warm-up
    for _ in range(warmup):
        _clear_cache(cache)
        if setup is not None:
            fn(setup())
        else:
            fn()
    torch.cuda.synchronize()

    # Benchmark
    for i in range(rep):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        _clear_cache(cache)
        # setup before timing (not included in measurement)
        if setup is not None:
            setup_result = setup()
            torch.cuda.synchronize()
            start_events[i].record()
            fn(setup_result)
        else:
            torch.cuda.synchronize()
            start_events[i].record()
            fn()
        end_events[i].record()

    # Record clocks
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return _summarize_statistics(times, quantiles, return_mode)


def _clone_args(args: list[Any]) -> list[Any]:
    """Clone tensor arguments to prevent cross-iteration information leakage.

    This ensures each benchmark iteration starts with fresh, independent data,
    preventing kernels from exploiting shared state across iterations.
    """
    return [arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args]


def time_runnable(
    fn: Runnable,
    args: list[Any],
    warmup: int,
    iters: int,
    device: str,
    execution_device: Any,
    lock_clocks: bool = True,
) -> float:
    """Time the execution of a value-returning style Runnable kernel.

        Parameters
        ----------
    fn : Runnable
            The kernel function to benchmark (must be value-returning style).
        args : List[Any]
            List of arguments in definition order.
        warmup : int
            Number of warmup iterations before timing.
        iters : int
            Number of timing iterations to average over.
        device : str
            The CUDA device to run the benchmark on.
        execution_device : ExecutionDevice
            ExecutionDevice for GPU clock locking during benchmarking.
            Clocks are locked right after acquiring the device lock.
        lock_clocks : bool
            Whether to lock GPU clocks during benchmarking (default: True).
            When False, benchmarks run without clock locking.

        Returns
        -------
        float
            The average execution time in milliseconds.

        Notes
        -----
        Each iteration uses cloned tensor arguments to prevent kernels from
        exploiting shared state (e.g., via in-place modifications or caching
        results in output tensors) to artificially improve benchmark performance.
        Clone time is excluded from measurements via the setup mechanism.

        Clock locking happens at this level (after device lock acquisition) to ensure:
        1. Only one process accesses the device at a time (device lock)
        2. GPU runs at fixed frequency for consistent measurements (clock lock)
        3. Clocks are unlocked when done (automatic cleanup)
    """
    lock = _device_lock(device)
    clock_ctx = execution_device.lock_clocks() if lock_clocks else nullcontext()
    with lock:
        # Lock clocks right after acquiring device lock (if enabled)
        with clock_ctx:
            return do_bench(
                fn=lambda cloned_args: fn(*cloned_args),
                warmup=warmup,
                rep=iters,
                setup=lambda: _clone_args(args),
                device=device,
            )
