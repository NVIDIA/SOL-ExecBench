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

"""Reward hack defenses for SOLBench evaluation.

Provides detection functions for four common reward-hacking patterns.
The identity of torch.cuda.Event.elapsed_time is captured at module load
time — before any user code is imported — so patching after the fact is
detected.
"""

from __future__ import annotations

import time
from typing import Any, List

import torch

# ---------------------------------------------------------------------------
# Capture timing function identity at module load, before any user code runs.
# Used by check_monkey_patch() to detect post-import patching.
# ---------------------------------------------------------------------------
_ELAPSED_TIME_ADDR: int | None = None

try:
    import torch.cuda as _tc_init

    _ELAPSED_TIME_ADDR = id(_tc_init.Event.elapsed_time)
except Exception:
    pass


class RewardHackDetected(RuntimeError):
    """Raised when a reward-hacking pattern is detected in a submission."""


def check_monkey_patch() -> None:
    """Detect if torch.cuda.Event.elapsed_time has been patched.

    Compares the current function identity against the address captured at
    module load time.  Must be called before the timed section.

    Raises:
        RewardHackDetected: If the timing function has been replaced.
    """
    try:
        import torch.cuda as _tc

        if (
            _ELAPSED_TIME_ADDR is not None
            and id(_tc.Event.elapsed_time) != _ELAPSED_TIME_ADDR
        ):
            raise RewardHackDetected(
                "torch.cuda.Event.elapsed_time has been monkey-patched"
            )
    except RewardHackDetected:
        raise
    except Exception:
        pass


def check_thread_injection(threads_before: int, threads_after: int) -> None:
    """Detect if user code spawned background threads.

    Capture ``threading.active_count()`` before and after the user call and
    pass both values here.

    Raises:
        RewardHackDetected: If the thread count increased.
    """
    if threads_after > threads_before:
        raise RewardHackDetected(
            f"Thread injection detected: "
            f"{threads_after} threads after call vs {threads_before} before"
        )


def check_lazy_outputs(outputs: List[Any]) -> None:
    """Detect lazy/proxy tensors in the user output.

    Uses strict ``type()`` equality — not ``isinstance`` — so any subclass
    (including ``FakeTensor``) is rejected.

    Raises:
        RewardHackDetected: If any output is not exactly ``torch.Tensor``.
    """
    for t in outputs:
        if type(t) is not torch.Tensor:
            raise RewardHackDetected(
                f"Lazy evaluation detected: output is {type(t).__name__}, not torch.Tensor"
            )


def check_stream_injection(
    user_fn: Any,
    args: List[Any],
    timed_latency_ms: float,
    multiplier: float,
    device: str,
) -> None:
    """Detect work hidden on a non-default CUDA stream.

    Runs one additional synchronizing pass and compares its wall-clock latency
    against the timed latency.  A kernel that enqueues work on a side stream
    will appear fast during timing (which only waits on the default stream) but
    slow here.

    Args:
        user_fn: The user's kernel callable.
        args: Arguments to pass to ``user_fn``.
        timed_latency_ms: The measured median latency in milliseconds.
        multiplier: Threshold factor; sync latency must stay below
            ``multiplier * timed_latency_ms`` to pass.
        device: CUDA device string (e.g. ``"cuda:0"``).

    Raises:
        RewardHackDetected: If the sync latency exceeds the threshold.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    user_fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    sync_latency_s = time.perf_counter() - t0
    threshold = multiplier * timed_latency_ms / 1000.0
    if sync_latency_s > threshold:
        raise RewardHackDetected(
            f"Stream injection detected: sync latency {sync_latency_s * 1000:.3f} ms "
            f"exceeds {multiplier}x timed latency {timed_latency_ms:.3f} ms"
        )
