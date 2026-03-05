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

"""Smoke test for CuTe DSL (nvidia-cutlass-dsl) — vectorized elementwise add."""

import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from _cutedsl_kernels import _elementwise_add_2d


def test_cute_dsl_elementwise_add():
    """Vectorized bf16 elementwise add using CuTe DSL with 128-bit copies."""
    M, N = 64, 256
    a = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    c = torch.empty_like(a)

    a_c = from_dlpack(a).mark_layout_dynamic()
    b_c = from_dlpack(b).mark_layout_dynamic()
    c_c = from_dlpack(c).mark_layout_dynamic()

    compiled_fn = cute.compile(_elementwise_add_2d, a_c, b_c, c_c)
    compiled_fn(a_c, b_c, c_c)

    torch.testing.assert_close(c, a + b)
    print("✓ CuTe DSL elementwise add passed!")


if __name__ == "__main__":
    test_cute_dsl_elementwise_add()
