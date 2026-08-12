# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CUPTI activity normalization helpers."""

import types
import pytest

from sol_execbench.core.bench.cupti_utils import CuptiKernelInfo


def test_kernel_string_for_activity_without_name():
    """Unexpected activity kinds without a name must still produce a string identity."""
    activity = types.SimpleNamespace(
        kind="RUNTIME",
        name=None,
        start=0.0,
        end=1.0,
        correlation_id=0,
        bytes=0,
        copy_kind=0,
        value=0,
    )
    info = CuptiKernelInfo.from_activity(activity)
    assert isinstance(info.name, str)
    assert info.kernel_string() == "RUNTIME_0_0_0_RUNTIME"


