# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for normalizing and selecting CUPTI activity records."""

import ctypes
import ctypes.util
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import Any

from cupti import cupti

DEFAULT_CUPTI_BUFFER_SIZE_BYTES = 8 * 1024 * 1024
DEFAULT_CUPTI_MAX_BUFFER_RECORDS = 0
DEFAULT_CUPTI_ACTIVITY_KINDS = (
    cupti.ActivityKind.RUNTIME,
    cupti.ActivityKind.CONCURRENT_KERNEL,
    cupti.ActivityKind.DRIVER,
    cupti.ActivityKind.MEMCPY,
    cupti.ActivityKind.MEMSET,
)

_libstdcxx = ctypes.CDLL(ctypes.util.find_library("stdc++"))
_libstdcxx.__cxa_demangle.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_int),
]
_libstdcxx.__cxa_demangle.restype = ctypes.c_char_p


@lru_cache(maxsize=4096)
def _demangle(name: str) -> str:
    """Return the demangled C++ symbol name when libstdc++ can decode it."""
    status = ctypes.c_int()
    result = _libstdcxx.__cxa_demangle(name.encode(), None, None, ctypes.byref(status))
    if status.value != 0:
        return name
    return result.decode().replace(" >", ">")


@dataclass(frozen=True)
class CuptiKernelInfo:
    """Normalized CUPTI activity for a kernel, memcpy, or memset operation."""

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
        if activity.kind == cupti.ActivityKind.MEMCPY:
            return "MEMCPY"
        if activity.kind == cupti.ActivityKind.MEMSET:
            return "MEMSET"
        return None

    @staticmethod
    def get_bytes(activity):
        if activity.kind in (cupti.ActivityKind.MEMCPY, cupti.ActivityKind.MEMSET):
            return activity.bytes
        return 0

    @staticmethod
    def get_copy_kind(activity):
        if activity.kind == cupti.ActivityKind.MEMCPY:
            return activity.copy_kind
        return 0

    @staticmethod
    def get_value(activity):
        if activity.kind == cupti.ActivityKind.MEMSET:
            return activity.value
        return 0

    def kernel_string(self) -> str:
        """Return the stable activity identity used across timing iterations."""
        return f"{self.name}_{self.copy_kind}_{self.bytes}_{self.value}_{self.kind}"


def kernel_activity_sequence(kernels: list[CuptiKernelInfo]) -> list[str]:
    return [kernel.kernel_string() for kernel in kernels]


def kernel_activity_counts(kernels: list[CuptiKernelInfo]) -> Counter[str]:
    return Counter(kernel_activity_sequence(kernels))


def _relative_order_score(candidate_names: list[str], expected_names: list[str]) -> int:
    """Return the longest-common-subsequence score for two activity lists."""
    scores = [0] * (len(expected_names) + 1)
    for candidate_name in candidate_names:
        previous_diagonal = 0
        for idx, expected_name in enumerate(expected_names, start=1):
            previous_score = scores[idx]
            if candidate_name == expected_name:
                scores[idx] = max(scores[idx], previous_diagonal + 1)
            else:
                scores[idx] = max(scores[idx], scores[idx - 1])
            previous_diagonal = previous_score
    return scores[-1]


def _kernel_activity_span(kernels: list[CuptiKernelInfo]) -> float:
    return max(kernel.end for kernel in kernels) - min(
        kernel.start for kernel in kernels
    )


def select_activity_sequence(
    kernels: list[CuptiKernelInfo],
    expected_kernel_names: list[str],
    *,
    iteration: int,
) -> list[CuptiKernelInfo]:
    """Select the discovered user activity sequence from a noisy timing window."""
    expected_count = len(expected_kernel_names)
    if expected_count == 0:
        raise ValueError("No expected kernel activities recorded")

    expected_name_set = set(expected_kernel_names)
    expected_counts = Counter(expected_kernel_names)
    candidate_kernels = [
        kernel for kernel in kernels if kernel.kernel_string() in expected_name_set
    ]
    candidate_names = kernel_activity_sequence(candidate_kernels)

    for start_idx in range(len(candidate_kernels) - expected_count + 1):
        end_idx = start_idx + expected_count
        if candidate_names[start_idx:end_idx] == expected_kernel_names:
            return candidate_kernels[start_idx:end_idx]

    best_kernels: list[CuptiKernelInfo] | None = None
    best_score: tuple[int, float] | None = None
    for start_idx in range(len(candidate_kernels) - expected_count + 1):
        end_idx = start_idx + expected_count
        window_names = candidate_names[start_idx:end_idx]
        if Counter(window_names) != expected_counts:
            continue
        window_kernels = candidate_kernels[start_idx:end_idx]
        score = (
            _relative_order_score(window_names, expected_kernel_names),
            -_kernel_activity_span(window_kernels),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_kernels = window_kernels

    if best_kernels is not None:
        return best_kernels

    raise ValueError(
        f"Expected kernel activity sequence not found at iteration {iteration}: "
        f"{expected_kernel_names} not in {candidate_names}"
    )


@dataclass
class CuptiActivityBuffers:
    """Mutable CUPTI activity buffers populated by activity callbacks."""

    kernels: list[CuptiKernelInfo] = field(default_factory=list)


def request_cupti_activity_buffer() -> tuple[int, int]:
    return DEFAULT_CUPTI_BUFFER_SIZE_BYTES, DEFAULT_CUPTI_MAX_BUFFER_RECORDS


def collect_cupti_activity_records(
    buffers: CuptiActivityBuffers, activities: list
) -> None:
    for activity in activities:
        if activity.kind in (
            cupti.ActivityKind.CONCURRENT_KERNEL,
            cupti.ActivityKind.MEMCPY,
            cupti.ActivityKind.MEMSET,
        ):
            buffers.kernels.append(CuptiKernelInfo.from_activity(activity))


@contextmanager
def collect_cupti_activities(
    activity_kinds: tuple[Any, ...] = DEFAULT_CUPTI_ACTIVITY_KINDS,
) -> Iterator[CuptiActivityBuffers]:
    """Collect normalized CUPTI activities and always clean up tracing state."""
    buffers = CuptiActivityBuffers()
    enabled_activity_kinds: list[Any] = []
    try:
        for activity_kind in activity_kinds:
            cupti.activity_enable(activity_kind)
            enabled_activity_kinds.append(activity_kind)
        cupti.activity_register_callbacks(
            request_cupti_activity_buffer,
            partial(collect_cupti_activity_records, buffers),
        )
        yield buffers
    finally:
        if enabled_activity_kinds:
            try:
                cupti.activity_flush_all(0)
            finally:
                for activity_kind in enabled_activity_kinds:
                    cupti.activity_disable(activity_kind)
                cupti.finalize()
