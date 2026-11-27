# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sol_execbench.core.bench import cupti_utils
from sol_execbench.core.bench.cupti_utils import (
    CUPTI_ACTIVITY_FLAG_FLUSH_FORCED,
    _demangle,
    collect_cupti_activities,
    select_activity_sequence,
)


class Kernel:
    def __init__(self, name, correlation_id, start, end):
        self.name = name
        self.correlation_id = correlation_id
        self.start = start
        self.end = end

    def kernel_string(self):
        return self.name


def test_demangle_caches_repeated_symbol():
    _demangle.cache_clear()

    assert _demangle("_Z3foov") == "foo()"
    first = _demangle.cache_info()
    assert first.misses == 1

    assert _demangle("_Z3foov") == "foo()"
    second = _demangle.cache_info()
    assert second.hits == first.hits + 1
    assert second.misses == first.misses


def test_select_activity_sequence_scores_repeated_name_windows():
    kernels = [
        Kernel("user_a", correlation_id=1, start=100, end=105),
        Kernel("user_c", correlation_id=2, start=110, end=115),
        Kernel("user_a", correlation_id=3, start=120, end=125),
        Kernel("user_b", correlation_id=4, start=130, end=135),
        Kernel("user_a", correlation_id=5, start=150, end=155),
    ]

    selected = select_activity_sequence(
        kernels,
        ["user_a", "user_b", "user_a", "user_c"],
        iteration=0,
    )

    assert [kernel.correlation_id for kernel in selected] == [2, 3, 4, 5]


def test_select_activity_sequence_tiebreaks_reordered_windows_by_span():
    kernels = [
        Kernel("user_b", correlation_id=1, start=100, end=101),
        Kernel("user_a", correlation_id=2, start=200, end=202),
        Kernel("user_c", correlation_id=3, start=205, end=208),
        Kernel("user_b", correlation_id=4, start=209, end=210),
    ]

    selected = select_activity_sequence(
        kernels,
        ["user_a", "user_b", "user_c"],
        iteration=0,
    )

    assert [kernel.correlation_id for kernel in selected] == [2, 3, 4]


def test_select_activity_sequence_no_match_raises():
    kernels = [
        Kernel("setup_copy", correlation_id=1, start=100, end=105),
        Kernel("cache_clear", correlation_id=2, start=110, end=115),
    ]

    with pytest.raises(ValueError, match="Expected kernel activity sequence not found"):
        select_activity_sequence(kernels, ["user_a"], iteration=0)


def test_select_activity_sequence_prefers_match_closest_to_target_window():
    kernels = [
        Kernel("user_a", correlation_id=1, start=20, end=25),
        Kernel("user_a", correlation_id=2, start=125, end=130),
        Kernel("user_a", correlation_id=3, start=250, end=255),
    ]

    selected = select_activity_sequence(
        kernels,
        ["user_a"],
        iteration=0,
        target_window=(100, 120),
    )

    assert [kernel.correlation_id for kernel in selected] == [2]


def test_collect_cupti_activities_registers_callbacks_first_and_forces_flush(
    monkeypatch,
):
    events = []

    monkeypatch.setattr(
        cupti_utils.cupti,
        "activity_register_callbacks",
        lambda req, comp: events.append("register"),
    )
    monkeypatch.setattr(
        cupti_utils.cupti,
        "activity_enable",
        lambda kind: events.append(f"enable:{kind}"),
    )
    monkeypatch.setattr(
        cupti_utils.cupti,
        "activity_flush_all",
        lambda flag: events.append(f"flush:{flag}"),
    )
    monkeypatch.setattr(
        cupti_utils.cupti,
        "activity_disable",
        lambda kind: events.append(f"disable:{kind}"),
    )
    monkeypatch.setattr(
        cupti_utils.cupti,
        "finalize",
        lambda: events.append("finalize"),
    )

    with collect_cupti_activities(activity_kinds=("k1", "k2")):
        events.append("body")

    assert events == [
        "register",
        "enable:k1",
        "enable:k2",
        "body",
        f"flush:{CUPTI_ACTIVITY_FLAG_FLUSH_FORCED}",
        "disable:k1",
        "disable:k2",
        "finalize",
    ]
