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

import math
import warnings
from pathlib import Path

import pytest
import torch

import sol_execbench.core.bench.io as io_mod
from sol_execbench.core.bench.io import (
    ShiftingMemoryPoolAllocator,
    _cast_to_fp4x2,
    _generate_heuristic_tensor,
    _is_binary_mask,
    _is_causal_attention_mask,
    _is_norm_bias,
    _is_norm_weight,
    _is_positive_tensor,
    _is_rope_cos_sin,
    _is_softmax_output,
    _is_ssm_decay,
    _is_weight_matrix,
    _rand_tensor,
    _resolve_blob_path,
    gen_inputs,
    load_safetensors,
    normalize_outputs,
)
from sol_execbench.core.data import Definition, Workload

# ------------------------------------------------------------------
# _rand_tensor
# ------------------------------------------------------------------


class TestRandTensor:
    def test_float32(self):
        t = _rand_tensor([4, 8], torch.float32, torch.device("cpu"))
        assert t.shape == (4, 8)
        assert t.dtype == torch.float32

    def test_float16(self):
        t = _rand_tensor([3], torch.float16, torch.device("cpu"))
        assert t.dtype == torch.float16

    def test_bfloat16(self):
        t = _rand_tensor([2, 2], torch.bfloat16, torch.device("cpu"))
        assert t.dtype == torch.bfloat16

    def test_float8_e4m3fn(self):
        t = _rand_tensor([16], torch.float8_e4m3fn, torch.device("cpu"))
        assert t.dtype == torch.float8_e4m3fn

    def test_float8_e5m2(self):
        t = _rand_tensor([16], torch.float8_e5m2, torch.device("cpu"))
        assert t.dtype == torch.float8_e5m2

    def test_bool(self):
        t = _rand_tensor([100], torch.bool, torch.device("cpu"))
        assert t.dtype == torch.bool
        assert set(t.unique().tolist()).issubset({True, False})

    def test_int8(self):
        t = _rand_tensor([100], torch.int8, torch.device("cpu"))
        assert t.dtype == torch.int8
        assert t.min().item() >= -128
        assert t.max().item() < 128

    def test_int32(self):
        t = _rand_tensor([50], torch.int32, torch.device("cpu"))
        assert t.dtype == torch.int32

    def test_int64(self):
        t = _rand_tensor([50], torch.int64, torch.device("cpu"))
        assert t.dtype == torch.int64

    def test_unsupported_dtype(self):
        with pytest.raises(ValueError, match="Unsupported random dtype"):
            _rand_tensor([4], torch.complex64, torch.device("cpu"))


# ------------------------------------------------------------------
# _cast_to_fp4x2
# ------------------------------------------------------------------


class TestCastToFP4x2:
    def test_output_shape(self):
        x = torch.randn(4, 8)
        packed = _cast_to_fp4x2(x)
        assert packed.shape == (4, 4)

    def test_zeros_map_to_zero_code(self):
        x = torch.zeros(2, 4)
        packed = _cast_to_fp4x2(x)
        # Zero maps to code 0, so packed bytes should be 0
        assert packed.view(torch.uint8).sum().item() == 0


# ------------------------------------------------------------------
# Classification helpers
# ------------------------------------------------------------------


class TestIsWeightMatrix:
    @pytest.mark.parametrize(
        "name",
        ["weight", "q_proj_weight", "kv_proj", "down_proj", "weight1"],
    )
    def test_true_cases(self, name):
        assert _is_weight_matrix(name, (64, 128))

    @pytest.mark.parametrize("name", ["bias", "input", "hidden_states"])
    def test_false_cases(self, name):
        assert not _is_weight_matrix(name, (64, 128))

    def test_1d_always_false(self):
        assert not _is_weight_matrix("weight", (128,))


class TestIsNormWeight:
    @pytest.mark.parametrize(
        "name",
        [
            "norm_weight",
            "input_layernorm_weight",
            "q_norm_weight",
            "norm1_weight",
            "group_norm_weight",
        ],
    )
    def test_true_cases(self, name):
        assert _is_norm_weight(name)

    @pytest.mark.parametrize("name", ["weight", "q_proj_weight", "norm_bias"])
    def test_false_cases(self, name):
        assert not _is_norm_weight(name)


class TestIsNormBias:
    @pytest.mark.parametrize(
        "name",
        ["norm_bias", "input_layernorm_bias", "q_norm_bias", "norm1_bias"],
    )
    def test_true_cases(self, name):
        assert _is_norm_bias(name)

    @pytest.mark.parametrize("name", ["bias", "proj_bias", "norm_weight"])
    def test_false_cases(self, name):
        assert not _is_norm_bias(name)


class TestIsCausalAttentionMask:
    def test_square_attention_mask(self):
        assert _is_causal_attention_mask("attention_mask", (1, 32, 32), None)

    def test_causal_mask_name(self):
        assert _is_causal_attention_mask("causal_mask", (32, 32), None)

    def test_description_match(self):
        assert _is_causal_attention_mask(
            "mask", (32, 32), "Causal attention mask for decoder"
        )

    def test_non_square_false(self):
        assert not _is_causal_attention_mask("attention_mask", (32, 64), None)

    def test_1d_false(self):
        assert not _is_causal_attention_mask("attention_mask", (32,), None)


class TestIsBinaryMask:
    @pytest.mark.parametrize(
        "name", ["x_mask", "text_mask", "aspect_ratio_mask", "drop_mask"]
    )
    def test_known_mask_names(self, name):
        assert _is_binary_mask(name, None)

    def test_suffix_with_binary_description(self):
        assert _is_binary_mask("padding_mask", "binary mask, 1.0 for valid tokens")

    def test_suffix_without_description(self):
        assert not _is_binary_mask("padding_mask", None)

    def test_non_mask_name(self):
        assert not _is_binary_mask("hidden_states", "some description")


class TestIsRopeCosSin:
    @pytest.mark.parametrize(
        "name", ["cos", "sin", "cos_cached", "sin_cached", "rope_cos", "rope_sin"]
    )
    def test_true_cases(self, name):
        assert _is_rope_cos_sin(name)

    def test_false_cases(self):
        assert not _is_rope_cos_sin("cosine_similarity")


class TestIsPositiveTensor:
    @pytest.mark.parametrize(
        "name", ["rstd", "std", "variance", "q_rstd", "x_var", "variance1"]
    )
    def test_true_cases(self, name):
        assert _is_positive_tensor(name, None)

    def test_false_cases(self):
        assert not _is_positive_tensor("hidden_states", None)


class TestIsSsmDecay:
    @pytest.mark.parametrize("name", ["A", "A_log", "A_cumsum", "g"])
    def test_true_cases(self, name):
        assert _is_ssm_decay(name)

    def test_false_cases(self):
        assert not _is_ssm_decay("B")


class TestIsSoftmaxOutput:
    @pytest.mark.parametrize(
        "name", ["attn_weights", "attention_weights", "routing_weights"]
    )
    def test_true_cases(self, name):
        assert _is_softmax_output(name, None)

    def test_description_match(self):
        assert _is_softmax_output("probs", "softmax output probabilities")

    def test_false_cases(self):
        assert not _is_softmax_output("hidden_states", None)


# ------------------------------------------------------------------
# _generate_heuristic_tensor
# ------------------------------------------------------------------


class TestGenerateHeuristicTensor:
    def test_norm_weight_ones(self):
        t = _generate_heuristic_tensor(
            "norm_weight", (128,), torch.float32, torch.device("cpu")
        )
        assert t is not None
        assert torch.allclose(t, torch.ones(128))

    def test_norm_bias_zeros(self):
        t = _generate_heuristic_tensor(
            "input_layernorm_bias", (64,), torch.float32, torch.device("cpu")
        )
        assert t is not None
        assert torch.allclose(t, torch.zeros(64))

    def test_causal_mask_upper_triangular(self):
        t = _generate_heuristic_tensor(
            "attention_mask", (1, 8, 8), torch.float32, torch.device("cpu")
        )
        assert t is not None
        # Diagonal and below should be 0
        assert t[0, 0, 0].item() == 0.0
        assert t[0, 7, 0].item() == 0.0
        # Above diagonal should be -inf (finfo.min)
        assert t[0, 0, 1].item() == torch.finfo(torch.float32).min

    def test_binary_mask_values(self):
        t = _generate_heuristic_tensor(
            "x_mask", (100,), torch.float32, torch.device("cpu")
        )
        assert t is not None
        unique = set(t.unique().tolist())
        assert unique.issubset({0.0, 1.0})

    def test_rope_cos_in_range(self):
        t = _generate_heuristic_tensor(
            "cos", (32, 64), torch.float32, torch.device("cpu")
        )
        assert t is not None
        assert t.min().item() >= -1.0
        assert t.max().item() <= 1.0

    def test_rope_sin_in_range(self):
        t = _generate_heuristic_tensor(
            "sin", (32, 64), torch.float32, torch.device("cpu")
        )
        assert t is not None
        assert t.min().item() >= -1.0
        assert t.max().item() <= 1.0

    def test_positive_tensor(self):
        t = _generate_heuristic_tensor(
            "rstd", (16, 32), torch.float32, torch.device("cpu")
        )
        assert t is not None
        assert t.min().item() > 0.0

    def test_ssm_decay_A_negative(self):
        t = _generate_heuristic_tensor("A", (8, 16), torch.float32, torch.device("cpu"))
        assert t is not None
        assert t.max().item() < 0.0

    def test_ssm_decay_A_cumsum_monotonic(self):
        t = _generate_heuristic_tensor(
            "A_cumsum", (4, 32), torch.float32, torch.device("cpu")
        )
        assert t is not None
        # Should be non-increasing along last dim
        diffs = t[:, 1:] - t[:, :-1]
        assert diffs.max().item() <= 0.0

    def test_softmax_output_sums_to_one(self):
        t = _generate_heuristic_tensor(
            "attn_weights", (4, 8), torch.float32, torch.device("cpu")
        )
        assert t is not None
        row_sums = t.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(4), atol=1e-5)

    def test_weight_matrix_xavier_scale(self):
        shape = (256, 512)
        t = _generate_heuristic_tensor(
            "weight", shape, torch.float32, torch.device("cpu")
        )
        assert t is not None
        # Xavier scale: std ≈ 1/sqrt(fan_in) = 1/sqrt(512) ≈ 0.044
        expected_std = 1.0 / math.sqrt(shape[-1])
        assert abs(t.std().item() - expected_std) < 0.01

    def test_no_heuristic_returns_none(self):
        t = _generate_heuristic_tensor(
            "hidden_states", (4, 128), torch.float32, torch.device("cpu")
        )
        assert t is None

    def test_integer_dtype_returns_none(self):
        t = _generate_heuristic_tensor(
            "norm_weight", (128,), torch.int32, torch.device("cpu")
        )
        assert t is None

    @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_fp8_dtype_returns_none(self, dtype):
        """FP8 dtypes must skip heuristics (torch.randn etc. don't support FP8)."""
        # These names would match heuristics for float32 but must return None for FP8
        for name in ("norm_weight", "rstd", "weight", "attn_weights"):
            t = _generate_heuristic_tensor(name, (64, 64), dtype, torch.device("cpu"))
            assert t is None, f"Expected None for FP8 heuristic '{name}', got tensor"


# ------------------------------------------------------------------
# _resolve_blob_path
# ------------------------------------------------------------------


class TestResolveBlobPath:
    def test_exact_match(self, tmp_path):
        """Full relative path exists directly under blob root."""
        (tmp_path / "foo" / "bar").mkdir(parents=True)
        (tmp_path / "foo" / "bar" / "data.safetensors").touch()

        result = _resolve_blob_path(Path("foo/bar/data.safetensors"), [tmp_path])
        assert result == tmp_path / "foo" / "bar" / "data.safetensors"

    def test_partial_overlap(self, tmp_path):
        """Blob root already contains leading components of the relative path."""
        root = tmp_path / "foo"
        (root / "bar").mkdir(parents=True)
        (root / "bar" / "data.safetensors").touch()

        result = _resolve_blob_path(Path("foo/bar/data.safetensors"), [root])
        assert result == root / "bar" / "data.safetensors"

    def test_deep_overlap(self, tmp_path):
        """Blob root contains multiple leading components of the relative path."""
        root = tmp_path / "foo" / "bar"
        root.mkdir(parents=True)
        (root / "data.safetensors").touch()

        result = _resolve_blob_path(Path("foo/bar/data.safetensors"), [root])
        assert result == root / "data.safetensors"

    def test_no_match_returns_none(self, tmp_path):
        """No blob root contains the file."""
        result = _resolve_blob_path(Path("foo/bar/data.safetensors"), [tmp_path])
        assert result is None

    def test_prefers_full_path_over_stripped(self, tmp_path):
        """Full path match is preferred even when a stripped match also exists."""
        # Create both full and stripped matches
        (tmp_path / "foo" / "bar").mkdir(parents=True)
        (tmp_path / "foo" / "bar" / "data.safetensors").touch()
        (tmp_path / "bar").mkdir(parents=True)
        (tmp_path / "bar" / "data.safetensors").touch()

        result = _resolve_blob_path(Path("foo/bar/data.safetensors"), [tmp_path])
        # Full match (start=0) should win
        assert result == tmp_path / "foo" / "bar" / "data.safetensors"

    def test_multiple_roots_first_match_wins(self, tmp_path):
        """First blob root with a match is used."""
        root1 = tmp_path / "root1"
        root2 = tmp_path / "root2"
        (root1 / "data").mkdir(parents=True)
        (root2 / "data").mkdir(parents=True)
        (root1 / "data" / "file.safetensors").touch()
        (root2 / "data" / "file.safetensors").touch()

        result = _resolve_blob_path(Path("data/file.safetensors"), [root1, root2])
        assert result == root1 / "data" / "file.safetensors"

    def test_falls_through_to_second_root(self, tmp_path):
        """Falls through to second blob root when first has no match."""
        root1 = tmp_path / "root1"
        root2 = tmp_path / "root2"
        root1.mkdir()
        (root2 / "data").mkdir(parents=True)
        (root2 / "data" / "file.safetensors").touch()

        result = _resolve_blob_path(Path("data/file.safetensors"), [root1, root2])
        assert result == root2 / "data" / "file.safetensors"

    def test_single_component_path(self, tmp_path):
        """Relative path with a single component (just the filename)."""
        (tmp_path / "weights.safetensors").touch()

        result = _resolve_blob_path(Path("weights.safetensors"), [tmp_path])
        assert result == tmp_path / "weights.safetensors"


# ------------------------------------------------------------------
# ShiftingMemoryPoolAllocator
# ------------------------------------------------------------------


class TestShiftingMemoryPoolAllocator:
    def test_rebase_falls_back_for_dlpack_unsupported_dtype(self, monkeypatch):
        backing = torch.arange(32, dtype=torch.uint8)
        source = backing[3:19]

        def unsupported(_tensor):
            raise BufferError("float8 types are not supported by dlpack")

        monkeypatch.setattr(io_mod.dlpack, "from_dlpack", unsupported)

        rebased = io_mod._rebase_tensor_storage(source)

        assert rebased.storage_offset() == 0
        assert rebased._base is None
        assert rebased.data_ptr() == source.data_ptr()
        assert rebased.untyped_storage().data_ptr() == source.data_ptr()
        assert torch.equal(rebased, source)

    def test_unique_data_ptr_per_call(self):
        """Each call to get_unique_args returns tensors with distinct data_ptr."""
        inputs = [torch.randn(4, 8), torch.randn(16)]
        alloc = ShiftingMemoryPoolAllocator(inputs, [], total_iterations=10)

        ptrs = []
        for _ in range(10):
            args = alloc.get_unique_args()
            ptrs.append(tuple(a.data_ptr() for a in args))

        # All pointer tuples should be unique
        assert len(set(ptrs)) == 10

    def test_data_ptr_shifts_by_random_alignment_multiple(self):
        align = ShiftingMemoryPoolAllocator._POOL_ALIGNMENT
        max_shift = ShiftingMemoryPoolAllocator._MAX_SHIFT_BYTES
        t = torch.randn(64)
        alloc = ShiftingMemoryPoolAllocator([t], [], total_iterations=64, seed=200)

        ptrs = [alloc.get_unique_args()[0].data_ptr() for _ in range(64)]
        diffs = [ptrs[i + 1] - ptrs[i] for i in range(len(ptrs) - 1)]
        assert all(d > 0 and d % align == 0 for d in diffs)
        assert all(align <= d <= max_shift for d in diffs)
        assert len(set(diffs)) > 1

    def test_shift_sequence_is_reproducible_per_seed(self):
        t = torch.randn(64)

        def diffs_for_seed(seed):
            alloc = ShiftingMemoryPoolAllocator([t], [], 32, seed=seed)
            ptrs = [alloc.get_unique_args()[0].data_ptr() for _ in range(32)]
            return [ptrs[i + 1] - ptrs[i] for i in range(len(ptrs) - 1)]

        assert diffs_for_seed(200) == diffs_for_seed(200)
        assert diffs_for_seed(200) != diffs_for_seed(201)

    def test_visible_storage_rebased_to_shifted_data_ptr(self):
        alloc = ShiftingMemoryPoolAllocator([torch.randn(64)], [], 5)

        storage_ids = []
        for _ in range(5):
            shifted = alloc.get_unique_args()[0]
            storage_ids.append(shifted.untyped_storage()._cdata)
            assert shifted.storage_offset() == 0
            assert shifted._base is None
            assert shifted.untyped_storage().data_ptr() == shifted.data_ptr()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                assert shifted.storage().data_ptr() == shifted.data_ptr()

        assert len(set(storage_ids)) == 5

    def test_input_data_preserved(self):
        """Returned views contain the same data as the original input."""
        src = torch.randn(256 * 1024, dtype=torch.float32)  # 1 MiB
        alloc = ShiftingMemoryPoolAllocator([src], [], total_iterations=5)

        for _ in range(5):
            args = alloc.get_unique_args()
            assert torch.equal(args[0], src)

    def test_input_data_survives_inplace_mutation(self):
        """Source data is re-copied each call, so in-place mutation of a view
        does not corrupt subsequent calls."""
        src = torch.tensor([1.0, 2.0, 3.0, 4.0])
        alloc = ShiftingMemoryPoolAllocator([src], [], total_iterations=3)

        # First call — mutate the returned view in-place
        args = alloc.get_unique_args()
        args[0].fill_(999.0)

        # Second call should still contain the original data
        args2 = alloc.get_unique_args()
        assert torch.equal(args2[0], src)

    def test_shape_preserved(self):
        """Returned tensors have the same shape as the original input."""
        src = torch.randn(3, 4, 5)
        alloc = ShiftingMemoryPoolAllocator([src], [], total_iterations=3)

        for _ in range(3):
            args = alloc.get_unique_args()
            assert args[0].shape == (3, 4, 5)

    def test_dtype_preserved(self):
        """Returned tensors have the same dtype as the original input."""
        for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.int32):
            src = torch.ones(8, dtype=dtype)
            alloc = ShiftingMemoryPoolAllocator([src], [], total_iterations=2)
            args = alloc.get_unique_args()
            assert args[0].dtype == dtype

    def test_scalars_passed_through(self):
        """Non-tensor inputs (scalars) are returned unchanged."""
        inputs = [torch.randn(4), 42, 3.14, "hello"]
        alloc = ShiftingMemoryPoolAllocator(inputs, [], total_iterations=3)

        for _ in range(3):
            args = alloc.get_unique_args()
            assert len(args) == 4
            assert isinstance(args[0], torch.Tensor)
            assert args[1] == 42
            assert args[2] == 3.14
            assert args[3] == "hello"

    def test_outputs_zero_filled(self):
        """DPS output tensors are zero-filled on every call."""
        inputs = [torch.randn(4)]
        outputs = [torch.ones(3, 3)]  # non-zero source
        alloc = ShiftingMemoryPoolAllocator(inputs, outputs, total_iterations=5)

        for _ in range(5):
            args = alloc.get_unique_args()
            assert len(args) == 2
            out = args[1]
            assert out.shape == (3, 3)
            assert torch.equal(out, torch.zeros(3, 3))

    def test_output_unique_data_ptr(self):
        """DPS output tensors also get unique data_ptr per call."""
        outputs = [torch.zeros(8)]
        alloc = ShiftingMemoryPoolAllocator([], outputs, total_iterations=5)

        ptrs = [alloc.get_unique_args()[0].data_ptr() for _ in range(5)]
        assert len(set(ptrs)) == 5

    def test_raises_on_exhaustion(self):
        """Raises RuntimeError when called more than total_iterations times."""
        src = torch.randn(16)
        alloc = ShiftingMemoryPoolAllocator([src], [], total_iterations=3)

        for _ in range(3):
            alloc.get_unique_args()

        with pytest.raises(RuntimeError, match="exhausted"):
            alloc.get_unique_args()

    def test_pool_size_is_compact(self):
        """Pool overhead is proportional to total_iterations * alignment, not numel * iters."""
        src = torch.randn(1024)  # 4096 bytes
        iters = 20
        alloc = ShiftingMemoryPoolAllocator([src], [], total_iterations=iters)

        entry = alloc._input_entries[0]
        pool_numel = entry["pool"].numel()
        block_numel = 256 // src.element_size()
        assert (
            pool_numel == entry["storage_span"] + alloc._offset_blocks[-1] * block_numel
        )
        overhead = pool_numel - entry["storage_span"]
        max_shift_numel = (
            ShiftingMemoryPoolAllocator._MAX_SHIFT_BYTES // src.element_size()
        )
        assert (iters - 1) * block_numel <= overhead <= (iters - 1) * max_shift_numel
        assert pool_numel < src.numel() * iters

    def test_expanded_tensor_preserves_strides(self):
        """Expanded (stride-0) tensors store only physical elements, not logical numel."""
        batch, seq = 8, 64
        row = torch.arange(seq, dtype=torch.int64)
        src = row.unsqueeze(0).expand(batch, -1)  # shape (8,64), stride (0,1)
        assert not src.is_contiguous()
        assert src.numel() == batch * seq

        iters = 5
        alloc = ShiftingMemoryPoolAllocator([src], [], total_iterations=iters)

        entry = alloc._input_entries[0]
        # Pool should be sized by storage_span (seq), not logical numel (batch*seq)
        assert entry["storage_span"] == seq
        expected_pool = seq + alloc._offset_blocks[-1] * (256 // src.element_size())
        assert entry["pool"].numel() == expected_pool

        for _ in range(iters):
            args = alloc.get_unique_args()
            assert args[0].shape == (batch, seq)
            assert args[0].stride() == (0, 1)
            assert args[0].storage_offset() == 0
            assert args[0]._base is None
            assert args[0].untyped_storage().data_ptr() == args[0].data_ptr()
            # Every row should equal the original
            assert torch.equal(args[0][0], row)
            assert torch.equal(args[0][batch - 1], row)

    def test_mixed_inputs_and_outputs(self):
        """Correct ordering: inputs (tensors + scalars) then outputs."""
        inputs = [torch.randn(4), 99, torch.randn(2, 3)]
        outputs = [torch.zeros(5), torch.zeros(2, 2)]
        alloc = ShiftingMemoryPoolAllocator(inputs, outputs, total_iterations=3)

        args = alloc.get_unique_args()
        assert len(args) == 5
        assert args[0].shape == (4,)
        assert args[1] == 99
        assert args[2].shape == (2, 3)
        assert args[3].shape == (5,)
        assert args[4].shape == (2, 2)
        # Outputs are zeroed
        assert torch.equal(args[3], torch.zeros(5))
        assert torch.equal(args[4], torch.zeros(2, 2))

    def test_empty_inputs_and_outputs(self):
        """Works with no inputs and no outputs."""
        alloc = ShiftingMemoryPoolAllocator([], [], total_iterations=1)
        assert alloc.get_unique_args() == []

    def test_scalar_tensor_input(self):
        """0-dim (scalar) tensors are handled correctly."""
        src = torch.tensor(5.0)
        alloc = ShiftingMemoryPoolAllocator([src], [], total_iterations=3)

        ptrs = []
        for _ in range(3):
            args = alloc.get_unique_args()
            assert args[0].shape == ()
            assert args[0].item() == 5.0
            ptrs.append(args[0].data_ptr())
        assert len(set(ptrs)) == 3


# ------------------------------------------------------------------
# Shared helpers for gen_inputs / load_safetensors / normalize_outputs
# ------------------------------------------------------------------

_REFERENCE = "def run(a): return a"


def _make_definition(**overrides):
    base = dict(
        name="test_op",
        op_type="test",
        axes={"N": {"type": "var"}},
        inputs={"a": {"shape": ["N"], "dtype": "float32"}},
        outputs={"b": {"shape": ["N"], "dtype": "float32"}},
        reference=_REFERENCE,
    )
    base.update(overrides)
    return Definition(**base)


def _make_workload(**overrides):
    base = dict(uuid="test-uuid", axes={"N": 4}, inputs={"a": {"type": "random"}})
    base.update(overrides)
    return Workload(**base)


# ------------------------------------------------------------------
# normalize_outputs
# ------------------------------------------------------------------


class TestNormalizeOutputs:
    _CPU = torch.device("cpu")
    _NAMES = ["out"]
    _DTYPES = {"out": torch.float32}

    def test_single_tensor_passthrough(self):
        t = torch.zeros(3)
        result = normalize_outputs(
            t, device=self._CPU, output_names=self._NAMES, output_dtypes=self._DTYPES
        )
        assert torch.equal(result["out"], t)

    def test_dict_passthrough(self):
        t = torch.ones(3)
        result = normalize_outputs(
            {"out": t},
            device=self._CPU,
            output_names=self._NAMES,
            output_dtypes=self._DTYPES,
        )
        assert torch.equal(result["out"], t)

    def test_tuple_maps_to_output_names(self):
        t1, t2 = torch.zeros(2), torch.ones(2)
        result = normalize_outputs(
            (t1, t2),
            device=self._CPU,
            output_names=["a", "b"],
            output_dtypes={"a": torch.float32, "b": torch.float32},
        )
        assert torch.equal(result["a"], t1)
        assert torch.equal(result["b"], t2)

    def test_scalar_converted_to_tensor(self):
        result = normalize_outputs(
            3.0, device=self._CPU, output_names=self._NAMES, output_dtypes=self._DTYPES
        )
        assert isinstance(result["out"], torch.Tensor)
        assert abs(float(result["out"]) - 3.0) < 1e-6

    def test_single_tensor_with_multiple_outputs_raises(self):
        with pytest.raises(RuntimeError):
            normalize_outputs(
                torch.zeros(3),
                device=self._CPU,
                output_names=["a", "b"],
                output_dtypes={"a": torch.float32, "b": torch.float32},
            )

    def test_tuple_wrong_length_raises(self):
        with pytest.raises(RuntimeError):
            normalize_outputs(
                (torch.zeros(3),),
                device=self._CPU,
                output_names=["a", "b"],
                output_dtypes={"a": torch.float32, "b": torch.float32},
            )


# ------------------------------------------------------------------
# gen_inputs
# ------------------------------------------------------------------


class TestGenInputs:
    def test_random_input_has_correct_shape(self):
        d = _make_definition()
        wkl = _make_workload()  # N=4
        inputs = gen_inputs(d, wkl, "cpu")
        assert len(inputs) == 1
        assert inputs[0].shape == torch.Size([4])

    def test_random_input_has_correct_dtype(self):
        d = _make_definition()
        wkl = _make_workload()
        inputs = gen_inputs(d, wkl, "cpu")
        assert inputs[0].dtype == torch.float32

    def test_scalar_workload_input_returned_as_value(self):
        d = _make_definition(
            axes={"N": {"type": "var"}},
            inputs={
                "a": {"shape": ["N"], "dtype": "float32"},
                "s": {"shape": ["N"], "dtype": "float32"},
            },
            outputs={"b": {"shape": ["N"], "dtype": "float32"}},
            reference="def run(a, s): return a",
        )
        wkl = Workload(
            uuid="u",
            axes={"N": 4},
            inputs={"a": {"type": "random"}, "s": {"type": "scalar", "value": 0.5}},
        )
        inputs = gen_inputs(d, wkl, "cpu")
        assert inputs[1] == 0.5

    def test_missing_safetensors_raises(self):
        d = _make_definition(
            inputs={
                "a": {"shape": ["N"], "dtype": "float32"},
                "b": {"shape": ["N"], "dtype": "float32"},
            },
            outputs={"c": {"shape": ["N"], "dtype": "float32"}},
            reference="def run(a, b): return a",
        )
        wkl = Workload(
            uuid="u",
            axes={"N": 4},
            inputs={
                "a": {"type": "random"},
                "b": {
                    "type": "safetensors",
                    "path": "x.safetensors",
                    "tensor_key": "k",
                },
            },
        )
        with pytest.raises(RuntimeError, match="safetensors"):
            gen_inputs(d, wkl, "cpu")

    def test_missing_custom_tensors_raises(self):
        d = _make_definition(
            inputs={
                "a": {"shape": ["N"], "dtype": "float32"},
                "b": {"shape": ["N"], "dtype": "float32"},
            },
            outputs={"c": {"shape": ["N"], "dtype": "float32"}},
            custom_inputs_entrypoint="gen",
            reference="def run(a, b): return a\ndef gen(axes, device): return {}",
        )
        wkl = Workload(
            uuid="u",
            axes={"N": 4},
            inputs={"a": {"type": "custom"}, "b": {"type": "custom"}},
        )
        with pytest.raises(RuntimeError, match="CustomInput"):
            gen_inputs(d, wkl, "cpu")


# ------------------------------------------------------------------
# load_safetensors
# ------------------------------------------------------------------


class TestLoadSafetensors:
    def test_resolves_relative_path_from_blob_root(self, tmp_path):
        st = pytest.importorskip("safetensors.torch")
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        st.save_file({"data": t}, tmp_path / "tensor.safetensors")

        d = _make_definition()
        wkl = Workload(
            uuid="u",
            axes={"N": 4},
            inputs={
                "a": {
                    "type": "safetensors",
                    "path": "tensor.safetensors",
                    "tensor_key": "data",
                }
            },
        )
        result = load_safetensors(d, wkl, blob_roots=[tmp_path])
        assert "a" in result
        assert result["a"].shape == torch.Size([4])

    def test_tries_second_root_when_first_misses(self, tmp_path):
        st = pytest.importorskip("safetensors.torch")
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        st.save_file({"data": t}, tmp_path / "tensor.safetensors")

        d = _make_definition()
        wkl = Workload(
            uuid="u",
            axes={"N": 4},
            inputs={
                "a": {
                    "type": "safetensors",
                    "path": "tensor.safetensors",
                    "tensor_key": "data",
                }
            },
        )
        wrong_root = tmp_path / "nonexistent"
        result = load_safetensors(d, wkl, blob_roots=[wrong_root, tmp_path])
        assert "a" in result

    def test_missing_file_raises(self, tmp_path):
        pytest.importorskip("safetensors")
        d = _make_definition()
        wkl = Workload(
            uuid="u",
            axes={"N": 4},
            inputs={
                "a": {
                    "type": "safetensors",
                    "path": "missing.safetensors",
                    "tensor_key": "k",
                }
            },
        )
        with pytest.raises(Exception):
            load_safetensors(d, wkl, blob_roots=[tmp_path])

    def test_skips_non_safetensors_inputs(self, tmp_path):
        """Random inputs are not loaded by load_safetensors — result dict is empty."""
        pytest.importorskip("safetensors")
        d = _make_definition()
        wkl = _make_workload()  # all random
        result = load_safetensors(d, wkl, blob_roots=[tmp_path])
        assert result == {}
