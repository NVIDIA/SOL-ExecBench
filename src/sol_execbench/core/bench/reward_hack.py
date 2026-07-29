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

"""Reward hack defenses for SOL ExecBench evaluation.

Provides detection functions for five common reward-hacking patterns.
The identity of torch.cuda.Event.elapsed_time is captured at module load
time — before any user code is imported — so patching after the fact is
detected.
"""

from __future__ import annotations

from typing import Any, List, Sequence

import torch

from sol_execbench.core.bench.correctness import compute_error_stats

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


def check_timed_outputs(
    outputs: Sequence[Any],
    reference: Sequence[Any],
    tolerance: Any,
) -> None:
    """Verify that a call made *during the timing loop* is still correct.

    Correctness and performance are otherwise measured over two disjoint sets
    of calls, and the outputs produced during timing are never inspected.  A
    submission can therefore compute real results while correctness is being
    checked and skip the work once timing starts — the measured latency then
    reflects an empty kernel rather than the kernel under test.

    The caller obtains *outputs* from an extra, untimed call placed at an
    unpredictable point inside the measured loop (see
    :func:`~sol_execbench.core.bench.timing.time_runnable`), so a submission
    cannot tell it apart from a measured call.

    Args:
        outputs: Outputs produced by the audited call.
        reference: Ground-truth outputs for the same inputs.
        tolerance: Workload tolerance, as passed to
            :func:`~sol_execbench.core.bench.correctness.compute_error_stats`.

    Raises:
        RewardHackDetected: If the audited call produced a different number of
            outputs, changed a shape/dtype, or exceeded the tolerance.
    """
    if len(outputs) != len(reference):
        raise RewardHackDetected(
            f"Timed call produced {len(outputs)} output(s), expected "
            f"{len(reference)}: the solution behaves differently once timing "
            f"begins"
        )

    for idx, (got, want) in enumerate(zip(outputs, reference)):
        if not isinstance(got, torch.Tensor):
            raise RewardHackDetected(
                f"Timed call output {idx} is {type(got).__name__}, not a Tensor"
            )
        if got.shape != want.shape or got.dtype != want.dtype:
            raise RewardHackDetected(
                f"Timed call output {idx} changed from {tuple(want.shape)}/"
                f"{want.dtype} to {tuple(got.shape)}/{got.dtype} after the "
                f"correctness phase"
            )

        stats, exceeds = compute_error_stats(got, want, tolerance)
        if exceeds:
            raise RewardHackDetected(
                f"Timed call output {idx} does not match the reference "
                f"(max_absolute_error={stats.max_absolute_error:.3e}): the "
                f"solution stopped producing correct results once timing began"
            )


def snapshot_critical_functions(namespace: dict, names: List[str]) -> dict[str, int]:
    """Capture ``id()`` of named functions from a namespace.

    Call this **before** user code is imported.  Pass the returned dict to
    :func:`check_eval_integrity` after user code runs.

    Args:
        namespace: The globals dict to snapshot (typically ``globals()``).
        names: Function names to capture.

    Returns:
        Mapping of name → ``id()`` for each name present in *namespace*.
    """
    return {name: id(namespace[name]) for name in names if name in namespace}


def check_eval_integrity(snapshot: dict[str, int], namespace: dict) -> None:
    """Verify that critical eval-driver functions have not been replaced.

    Compares the current ``id()`` of each snapshotted name against the
    value captured before user code was imported.

    Args:
        snapshot: The dict returned by :func:`snapshot_critical_functions`.
        namespace: The current globals dict to check.

    Raises:
        RewardHackDetected: If any function identity has changed.
    """
    for name, expected_id in snapshot.items():
        current = namespace.get(name)
        if current is None or id(current) != expected_id:
            raise RewardHackDetected(
                f"Eval driver integrity violated: '{name}' has been monkey-patched"
            )
