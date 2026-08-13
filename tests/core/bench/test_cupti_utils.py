# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CUPTI utility helpers."""

import unittest

from cupti import cupti
from sol_execbench.core.bench.cupti_utils import CuptiKernelInfo, _kernel_activity_span


class KernelActivitySpanTests(unittest.TestCase):
    def _make_kernel(self, name, start, end):
        return CuptiKernelInfo(
            name=name,
            start=start,
            end=end,
            correlation_id=0,
            copy_kind=0,
            bytes=0,
            value=0,
            kind=cupti.ActivityKind.CONCURRENT_KERNEL,
            _activity=None,
        )

    def test_empty_list_raises(self):
        with self.assertRaisesRegex(
            ValueError, "kernels must contain at least one CUPTI activity"
        ):
            _kernel_activity_span([])

    def test_span_from_window(self):
        kernels = [
            self._make_kernel("a", start=1.0, end=3.0),
            self._make_kernel("b", start=2.0, end=5.0),
        ]
        self.assertAlmostEqual(_kernel_activity_span(kernels), 4.0)


if __name__ == "__main__":
    unittest.main()
