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

import pytest
import torch

from sol_execbench.bench.correctness import compute_error_stats
from sol_execbench.data.workload import CorrectnessSpec


def _spec(
    max_atol: float = 1e-5, max_rtol: float = 1e-5, required_matched_ratio: float = 1.0
) -> CorrectnessSpec:
    return CorrectnessSpec(
        max_atol=max_atol,
        max_rtol=max_rtol,
        required_matched_ratio=required_matched_ratio,
    )


class TestComputeErrorStats:
    """Tests for compute_error_stats, focusing on near-zero edge cases."""

    # ------------------------------------------------------------------
    # Basic / happy-path
    # ------------------------------------------------------------------

    def test_identical_tensors(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        max_abs, max_rel, exceeds, ratio = compute_error_stats(t, t, _spec())
        assert max_abs == 0.0
        assert max_rel == 0.0
        assert not exceeds
        assert ratio == 1.0

    def test_within_tolerance(self):
        ref = torch.tensor([1.0, 2.0, 3.0])
        out = ref + 1e-6
        max_abs, max_rel, exceeds, ratio = compute_error_stats(out, ref, _spec())
        assert not exceeds
        assert ratio == 1.0

    def test_exceeds_tolerance(self):
        ref = torch.tensor([1.0, 2.0, 3.0])
        out = ref + 1.0  # way outside tolerance
        _, _, exceeds, ratio = compute_error_stats(out, ref, _spec())
        assert exceeds
        assert ratio == 0.0

    # ------------------------------------------------------------------
    # Empty tensor
    # ------------------------------------------------------------------

    def test_empty_tensors(self):
        t = torch.tensor([])
        max_abs, max_rel, exceeds, ratio = compute_error_stats(t, t, _spec())
        assert max_abs == 0.0
        assert max_rel == 0.0
        assert not exceeds
        assert ratio == 1.0

    # ------------------------------------------------------------------
    # Near-zero reference values
    # ------------------------------------------------------------------

    def test_reference_exactly_zero(self):
        """When reference is 0, rel_error denominator is clamped to atol."""
        ref = torch.tensor([0.0])
        out = torch.tensor([1e-6])
        cfg = _spec(max_atol=1e-5, max_rtol=0.0)
        max_abs, max_rel, exceeds, ratio = compute_error_stats(out, ref, cfg)
        # abs_error = 1e-6, tolerance = atol = 1e-5 → within tolerance
        assert not exceeds
        assert ratio == 1.0
        # rel_error = 1e-6 / clamp(0, min=1e-5) = 1e-6 / 1e-5 = 0.1
        assert max_rel == pytest.approx(0.1, rel=1e-4)

    def test_reference_zero_output_exceeds_atol(self):
        """Error at zero reference that exceeds atol."""
        ref = torch.tensor([0.0])
        out = torch.tensor([1e-3])
        cfg = _spec(max_atol=1e-5, max_rtol=0.0)
        max_abs, max_rel, exceeds, ratio = compute_error_stats(out, ref, cfg)
        assert exceeds
        assert ratio == 0.0
        # rel_error = 1e-3 / 1e-5 = 100.0
        assert max_rel == pytest.approx(100.0, rel=1e-4)

    def test_reference_near_zero_small_perturbation(self):
        """Reference near zero with a perturbation smaller than atol."""
        ref = torch.tensor([1e-10])
        out = torch.tensor([1e-10 + 1e-7])
        cfg = _spec(max_atol=1e-5, max_rtol=0.0)
        max_abs, max_rel, exceeds, ratio = compute_error_stats(out, ref, cfg)
        # abs_error ≈ 1e-7, tolerance = atol = 1e-5 → within tolerance
        assert not exceeds
        # rel_error uses clamp(|1e-10|, min=1e-5) = 1e-5 as denominator
        # rel = 1e-7 / 1e-5 = 0.01
        assert max_rel == pytest.approx(0.01, rel=1e-2)

    def test_both_zero(self):
        """Both output and reference are zero — no error."""
        ref = torch.tensor([0.0, 0.0])
        out = torch.tensor([0.0, 0.0])
        cfg = _spec(max_atol=1e-8, max_rtol=0.0)
        max_abs, max_rel, exceeds, ratio = compute_error_stats(out, ref, cfg)
        assert max_abs == 0.0
        assert max_rel == 0.0
        assert not exceeds
        assert ratio == 1.0

    def test_mixed_near_zero_and_large(self):
        """Mix of near-zero and large reference values."""
        ref = torch.tensor([0.0, 1.0, 1e-12, 100.0])
        out = torch.tensor([1e-6, 1.0 + 1e-6, 1e-12, 100.0 + 1e-6])
        cfg = _spec(max_atol=1e-5, max_rtol=1e-5)
        _, _, exceeds, ratio = compute_error_stats(out, ref, cfg)
        assert not exceeds
        assert ratio == 1.0

    # ------------------------------------------------------------------
    # Negative near-zero values
    # ------------------------------------------------------------------

    def test_negative_near_zero_reference(self):
        """Negative near-zero reference — clamp uses abs, so same behavior."""
        ref = torch.tensor([-1e-12])
        out = torch.tensor([1e-12])
        cfg = _spec(max_atol=1e-5, max_rtol=0.0)
        max_abs, max_rel, exceeds, ratio = compute_error_stats(out, ref, cfg)
        # abs_error = 2e-12, within atol=1e-5
        assert not exceeds
        # rel_error = 2e-12 / clamp(1e-12, min=1e-5) = 2e-12 / 1e-5 = 2e-7
        assert max_rel == pytest.approx(2e-7, rel=1e-2)

    # ------------------------------------------------------------------
    # Tolerance formula: |a - b| <= atol + rtol * |b|
    # ------------------------------------------------------------------

    def test_rtol_dominates_for_large_reference(self):
        """For large references, rtol term dominates tolerance."""
        ref = torch.tensor([1000.0])
        out = torch.tensor([1000.5])
        cfg = _spec(max_atol=1e-5, max_rtol=1e-3)
        # tolerance = 1e-5 + 1e-3 * 1000 = 1.00001
        # abs_error = 0.5 < 1.00001 → pass
        _, _, exceeds, _ = compute_error_stats(out, ref, cfg)
        assert not exceeds

    def test_atol_dominates_for_small_reference(self):
        """For near-zero references, atol term dominates tolerance."""
        ref = torch.tensor([1e-10])
        out = torch.tensor([1e-10 + 5e-6])
        cfg = _spec(max_atol=1e-5, max_rtol=1e-5)
        # tolerance = 1e-5 + 1e-5 * 1e-10 ≈ 1e-5
        # abs_error = 5e-6 < 1e-5 → pass
        _, _, exceeds, _ = compute_error_stats(out, ref, cfg)
        assert not exceeds

    def test_exact_tolerance_boundary(self):
        """Error exactly at the tolerance boundary."""
        ref = torch.tensor([1.0])
        atol, rtol = 1e-3, 1e-3
        tol = atol + rtol * 1.0  # 2e-3
        # Just within
        out_in = torch.tensor([1.0 + tol - 1e-7])
        _, _, exceeds_in, _ = compute_error_stats(
            out_in, ref, _spec(max_atol=atol, max_rtol=rtol)
        )
        assert not exceeds_in
        # Just outside
        out_out = torch.tensor([1.0 + tol + 1e-7])
        _, _, exceeds_out, _ = compute_error_stats(
            out_out, ref, _spec(max_atol=atol, max_rtol=rtol)
        )
        assert exceeds_out

    # ------------------------------------------------------------------
    # required_matched_ratio
    # ------------------------------------------------------------------

    def test_required_matched_ratio_default_is_one(self):
        """When required_matched_ratio is None, defaults to 1.0 (all must match)."""
        ref = torch.tensor([1.0, 2.0, 3.0])
        out = ref.clone()
        out[0] += 1.0  # one element way off
        _, _, exceeds, ratio = compute_error_stats(out, ref, _spec())
        assert exceeds  # 1/3 elements fail → ratio < 1.0
        assert ratio == pytest.approx(2.0 / 3.0, rel=1e-6)

    def test_partial_match_ratio_passes(self):
        """Passes when enough elements match, even if some don't."""
        ref = torch.ones(100)
        out = ref.clone()
        out[:5] += 1.0  # 5% mismatch
        cfg = _spec(required_matched_ratio=0.9)
        _, _, exceeds, ratio = compute_error_stats(out, ref, cfg)
        assert not exceeds
        assert ratio == pytest.approx(0.95, rel=1e-6)

    def test_partial_match_ratio_fails(self):
        """Fails when too many elements don't match."""
        ref = torch.ones(100)
        out = ref.clone()
        out[:20] += 1.0  # 20% mismatch
        cfg = _spec(required_matched_ratio=0.9)
        _, _, exceeds, ratio = compute_error_stats(out, ref, cfg)
        assert exceeds
        assert ratio == pytest.approx(0.80, rel=1e-6)

    def test_zero_matched_ratio_always_passes(self):
        """With required_matched_ratio=0, even all-wrong passes."""
        ref = torch.ones(10)
        out = ref + 100.0
        cfg = _spec(required_matched_ratio=0.0)
        _, _, exceeds, _ = compute_error_stats(out, ref, cfg)
        assert not exceeds

    # ------------------------------------------------------------------
    # Subnormal / denormalized floats
    # ------------------------------------------------------------------

    def test_subnormal_reference(self):
        """Subnormal float reference values don't cause issues."""
        smallest_normal = torch.finfo(torch.float32).tiny  # ~1.17e-38
        ref = torch.tensor([smallest_normal / 2])  # subnormal
        out = torch.tensor([smallest_normal / 2 + 1e-40])
        cfg = _spec(max_atol=1e-5, max_rtol=0.0)
        max_abs, max_rel, exceeds, ratio = compute_error_stats(out, ref, cfg)
        assert not exceeds
        assert ratio == 1.0

    # ------------------------------------------------------------------
    # Scalar (0-d) tensors
    # ------------------------------------------------------------------

    def test_scalar_tensors(self):
        """0-d tensors work correctly."""
        ref = torch.tensor(5.0)
        out = torch.tensor(5.001)
        max_abs, max_rel, exceeds, ratio = compute_error_stats(
            out, ref, _spec(max_atol=0.01, max_rtol=0.01)
        )
        assert not exceeds
        assert ratio == 1.0
        assert max_abs == pytest.approx(0.001, abs=1e-4)

    # ------------------------------------------------------------------
    # dtype casting
    # ------------------------------------------------------------------

    def test_float16_inputs(self):
        """float16 inputs are upcast to float32 internally."""
        ref = torch.tensor([1.0, 2.0], dtype=torch.float16)
        out = torch.tensor([1.0, 2.0], dtype=torch.float16)
        max_abs, max_rel, exceeds, ratio = compute_error_stats(out, ref, _spec())
        assert not exceeds
        assert ratio == 1.0

    def test_bfloat16_inputs(self):
        """bfloat16 inputs are upcast to float32 internally."""
        ref = torch.tensor([1.0, 0.0, -1.0], dtype=torch.bfloat16)
        out = ref.clone()
        max_abs, max_rel, exceeds, ratio = compute_error_stats(out, ref, _spec())
        assert max_abs == 0.0
        assert not exceeds

    # ------------------------------------------------------------------
    # Large tensors with sparse errors near zero
    # ------------------------------------------------------------------

    def test_large_tensor_single_near_zero_outlier(self):
        """One near-zero element with error, rest are fine."""
        ref = torch.ones(10000)
        ref[5000] = 0.0
        out = ref.clone()
        out[5000] = 1e-4  # error at the zero element
        cfg = _spec(max_atol=1e-5, max_rtol=1e-5)
        _, _, exceeds, ratio = compute_error_stats(out, ref, cfg)
        # 1 out of 10000 exceeds → ratio = 0.9999
        assert ratio == pytest.approx(9999.0 / 10000.0, rel=1e-6)
        # With default required_matched_ratio=1.0, this fails
        assert exceeds

    def test_large_tensor_near_zero_outlier_with_relaxed_ratio(self):
        """Same as above but with relaxed ratio — passes."""
        ref = torch.ones(10000)
        ref[5000] = 0.0
        out = ref.clone()
        out[5000] = 1e-4
        cfg = _spec(max_atol=1e-5, max_rtol=1e-5, required_matched_ratio=0.999)
        _, _, exceeds, ratio = compute_error_stats(out, ref, cfg)
        assert not exceeds

    # ------------------------------------------------------------------
    # Inf / NaN behavior
    # ------------------------------------------------------------------

    def test_inf_in_output(self):
        """Inf in output produces inf abs_error and fails."""
        ref = torch.tensor([1.0])
        out = torch.tensor([float("inf")])
        _, _, exceeds, _ = compute_error_stats(out, ref, _spec())
        assert exceeds

    def test_nan_in_output(self):
        """NaN in output produces non-finite abs_error and fails."""
        ref = torch.tensor([1.0])
        out = torch.tensor([float("nan")])
        _, _, exceeds, ratio = compute_error_stats(out, ref, _spec())
        assert exceeds
        assert ratio == 0.0
