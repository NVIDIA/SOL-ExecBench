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

"""Correctness computation utilities."""

from __future__ import annotations

import torch

from ..data.workload import CorrectnessSpec


def compute_error_stats(
    output: torch.Tensor, reference: torch.Tensor, correctness: CorrectnessSpec
) -> tuple[float, float, bool, float]:
    x = output.to(torch.float32)
    y = reference.to(torch.float32)

    abs_error = torch.abs(x - y)

    total_elements = abs_error.numel()
    if total_elements == 0:
        return 0.0, 0.0, False, 1.0

    # torch.allclose style: |a - b| <= atol + rtol * |b|
    tolerance = correctness.max_atol + correctness.max_rtol * torch.abs(y)

    # ensure nans automatically exceed tolerance
    exceeds_tol_mask = (abs_error > tolerance) | ~torch.isfinite(abs_error)

    exceeds_count = float(exceeds_tol_mask.sum().item())
    matched_ratio = 1.0 - (exceeds_count / float(total_elements))
    matched_ratio = max(0.0, min(1.0, matched_ratio))

    exceeds_tol = matched_ratio < correctness.required_matched_ratio

    max_abs = float(abs_error.max().item())
    # Relative error using max_atol as floor to avoid division-by-near-zero
    rel_error = abs_error / torch.clamp(torch.abs(y), min=correctness.max_atol)
    max_rel = float(rel_error.max().item())

    return max_abs, max_rel, exceeds_tol, matched_ratio
