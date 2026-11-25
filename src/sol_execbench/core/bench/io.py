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

"""Input generation utilities for benchmark execution."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator
from os import urandom
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils import dlpack

from sol_execbench.core.bench.reward_hack import RewardHackDetected
from sol_execbench.core.data import (
    Definition,
    Workload,
)
from sol_execbench.core.data.workload import CustomInput, SafetensorsInput, ScalarInput
from sol_execbench.core.data.dtypes import dtype_str_to_torch_dtype


def _cast_to_fp4x2(x: torch.Tensor) -> torch.Tensor:
    """Quantize a tensor to FP4 E2M1 and pack into uint8 (2 FP4 values per byte).

    Args:
        x: Input tensor of shape (..., cols) with values in range [-6, 6]

    Returns:
        uint8 tensor of shape (..., cols//2) with packed FP4 values
    """
    result = torch.zeros_like(x, dtype=torch.uint8)

    # Positive values
    result[(x >= 0.0) & (x <= 0.25)] = 0
    result[(x > 0.25) & (x < 0.75)] = 1
    result[(x >= 0.75) & (x <= 1.25)] = 2
    result[(x > 1.25) & (x < 1.75)] = 3
    result[(x >= 1.75) & (x <= 2.5)] = 4
    result[(x > 2.5) & (x < 3.5)] = 5
    result[(x >= 3.5) & (x <= 5.0)] = 6
    result[x > 5.0] = 7

    # Negative values
    result[(x >= -0.25) & (x < 0.0)] = 8
    result[(x < -0.25) & (x > -0.75)] = 9
    result[(x <= -0.75) & (x >= -1.25)] = 10
    result[(x < -1.25) & (x > -1.75)] = 11
    result[(x <= -1.75) & (x >= -2.5)] = 12
    result[(x < -2.5) & (x > -3.5)] = 13
    result[(x <= -3.5) & (x >= -5.0)] = 14
    result[x < -5.0] = 15

    # Pack two FP4 values into one byte along cols dimension
    packed = result[..., ::2] + result[..., 1::2] * 16
    return packed.view(torch.float4_e2m1fn_x2)


def _storage_span_from_first_element(tensor: torch.Tensor) -> int:
    """Return physical storage elements reachable from the logical first element."""
    if tensor.numel() == 0:
        return 0

    span = 1
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        if stride < 0:
            raise ValueError("negative tensor strides are not supported")
        if size > 1:
            span += (size - 1) * stride
    return span


def _rebase_tensor_storage(tensor: torch.Tensor) -> torch.Tensor:
    """Return a tensor whose visible storage starts at ``tensor.data_ptr()``."""
    try:
        return dlpack.from_dlpack(tensor)
    except (BufferError, RuntimeError) as error:
        if "not supported by dlpack" not in str(error):
            raise

    # Older PyTorch builds reject FP8/FP4 tensors during DLPack export. Rebase
    # the same device pointer through a non-owning storage in that case.
    storage_span = _storage_span_from_first_element(tensor)
    storage = torch._C._construct_storage_from_data_pointer(
        tensor.data_ptr(),
        tensor.device,
        storage_span * tensor.element_size(),
    )
    return torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
        storage,
        0,
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def _rand_tensor(
    shape: List[int], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device=device)

    # low-precision floats
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        t = torch.randn(shape, dtype=torch.float32, device=device).clamp_(-2.0, 2.0)
        return t.to(dtype)
    elif dtype == torch.float4_e2m1fn_x2:
        return _cast_to_fp4x2(torch.randn(shape, dtype=torch.float32, device=device))

    # booleans
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device=device)

    # integers
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        ranges = {
            torch.int8: (-128, 128),
            torch.int16: (-1024, 1024),
            torch.int32: (-1024, 1024),
            torch.int64: (-1024, 1024),
        }
        low, high = ranges[dtype]
        return torch.randint(low, high, shape, device=device, dtype=dtype)

    raise ValueError(f"Unsupported random dtype: {dtype}")


# ── Heuristic tensor generation ──────────────────────────────────────────────


def is_sampling_operation(definition: Definition) -> bool:
    return getattr(definition, "op_type", None) == "sampling"


def _is_weight_matrix(name: str, shape: tuple[int, ...]) -> bool:
    if len(shape) < 2:
        return False
    weight_suffixes = (
        "_weight",
        "_weights",
        "_proj",
        "_projs",
        "_proj_weight",
        "_proj_weights",
        "_weight_matrix",
    )
    if name.endswith(weight_suffixes) or name == "weight":
        return True
    stripped = name.rstrip("0123456789")
    if stripped and stripped in ("weight",) or stripped.endswith(weight_suffixes):
        return True
    return False


def _is_norm_weight(name: str) -> bool:
    if name == "norm_weight":
        return True
    if name.endswith("_weight"):
        prefix = name[: -len("_weight")]
        if prefix.endswith(("_norm", "_layernorm", "layernorm")):
            return True
        stripped = prefix.rstrip("0123456789")
        if stripped and stripped.endswith(("norm", "layernorm")):
            return True
    return False


def _is_norm_bias(name: str) -> bool:
    if name == "norm_bias":
        return True
    if name.endswith("_bias"):
        prefix = name[: -len("_bias")]
        if prefix.endswith(("_norm", "_layernorm", "layernorm")):
            return True
        stripped = prefix.rstrip("0123456789")
        if stripped and stripped.endswith(("norm", "layernorm")):
            return True
    return False


def _is_causal_attention_mask(
    name: str, shape: tuple[int, ...], description: Optional[str]
) -> bool:
    if len(shape) < 2 or shape[-1] != shape[-2]:
        return False
    if name in ("attention_mask", "causal_mask"):
        return True
    if (
        description
        and "attention mask" in description.lower()
        and "causal" in description.lower()
    ):
        return True
    return False


def _is_binary_mask(name: str, description: Optional[str]) -> bool:
    mask_names = (
        "x_mask",
        "text_mask",
        "aspect_ratio_mask",
        "drop_mask",
        "attention_mask",
    )
    if name in mask_names:
        return True
    if name.endswith("_mask") and description:
        desc_lower = description.lower()
        if any(
            kw in desc_lower
            for kw in ("binary", "{0, 1}", "0 or 1", "1.0 for valid", "0.0 for masked")
        ):
            return True
    return False


def _is_rope_cos_sin(name: str) -> bool:
    return name in ("cos", "sin", "cos_cached", "sin_cached", "rope_cos", "rope_sin")


def _is_positive_tensor(name: str, description: Optional[str]) -> bool:
    positive_names = ("rstd", "std", "variance", "var")
    if name in positive_names:
        return True
    if name.endswith(("_rstd", "_std", "_variance", "_var")):
        return True
    base = name.rstrip("0123456789")
    if base in positive_names or base.endswith(("_rstd", "_std", "_var")):
        return True
    return False


def _is_ssm_decay(name: str) -> bool:
    return name in ("A", "A_log", "A_cumsum", "g")


def _is_softmax_output(name: str, description: Optional[str]) -> bool:
    if name in ("attn_weights", "attention_weights", "routing_weights"):
        return True
    if (
        description
        and "softmax" in description.lower()
        and "output" in description.lower()
    ):
        return True
    return False


def _generate_heuristic_tensor(
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    description: Optional[str] = None,
    *,
    audit_changes: list[Callable[[], Any]] | None = None,
) -> Optional[torch.Tensor]:
    """Generate a tensor using heuristics based on the input name.

    Returns None if no heuristic matches, falling back to _rand_tensor.
    """
    if not dtype.is_floating_point:
        return None

    # Low-precision floats (FP8, FP4) don't support ops like torch.randn/ones/empty.uniform_;
    # fall back to _rand_tensor() which generates in float32 then converts.
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2):
        return None

    if _is_norm_weight(name):
        weight = torch.ones(shape, dtype=dtype, device=device)
        if audit_changes is not None and weight.numel():
            audit_changes.append(lambda: weight.view(-1)[::2].fill_(0.5))
        return weight

    if _is_norm_bias(name):
        bias = torch.zeros(shape, dtype=dtype, device=device)
        if audit_changes is not None and bias.numel():
            audit_changes.append(lambda: bias.view(-1)[::2].fill_(0.25))
        return bias

    if _is_causal_attention_mask(name, shape, description):
        seq_len = shape[-1]
        mask = torch.zeros(shape, dtype=dtype, device=device)
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        mask[..., causal] = torch.finfo(dtype).min
        if audit_changes is not None and seq_len > 1:
            audit_changes.append(lambda: mask[..., 1::4].fill_(torch.finfo(dtype).min))
        return mask

    if _is_binary_mask(name, description):
        mask = torch.randint(0, 2, shape, device=device).to(dtype)
        if audit_changes is not None and mask.numel() and shape and shape[-1] >= 2:

            def refresh_binary_mask():
                mask[..., 0].copy_(mask[..., 0] == 0)
                mask[..., 1].copy_(mask[..., 0] == 0)

            audit_changes.append(refresh_binary_mask)
        return mask

    if _is_rope_cos_sin(name):
        t = torch.randn(shape, dtype=torch.float32, device=device).clamp_(
            -math.pi, math.pi
        )
        if name in ("cos", "cos_cached", "rope_cos"):
            return torch.cos(t).to(dtype)
        else:
            return torch.sin(t).to(dtype)

    if _is_positive_tensor(name, description):
        return torch.randn(shape, dtype=dtype, device=device).abs() + 0.1

    if _is_ssm_decay(name):
        if name == "A_cumsum":
            raw = torch.empty(shape, dtype=dtype, device=device).uniform_(-0.1, 0.0)
            return raw.cumsum(dim=-1)
        elif name == "A_log":
            return torch.empty(shape, dtype=dtype, device=device).uniform_(-5.0, -1.0)
        elif name == "A":
            return torch.empty(shape, dtype=dtype, device=device).uniform_(-1.0, -0.001)
        else:  # g (decay gate)
            return torch.empty(shape, dtype=dtype, device=device).uniform_(-5.0, 0.0)

    if _is_softmax_output(name, description):
        logits = torch.randn(shape, dtype=torch.float32, device=device)
        return torch.softmax(logits, dim=-1).to(dtype)

    if _is_weight_matrix(name, shape):
        fan_in = shape[-1]
        weight = torch.randn(shape, dtype=dtype, device=device) / math.sqrt(fan_in)
        if audit_changes is not None and weight.numel() >= 4:
            quarter = weight.numel() // 4
            audit_changes.append(
                lambda: weight.view(-1).narrow(0, quarter, 2 * quarter).neg_()
            )
        return weight

    return None


# ── Safetensors loading ──────────────────────────────────────────────────────


def _resolve_blob_path(rel: Path, blob_roots: List[Path]) -> Optional[Path]:
    """Resolve a relative path against blob roots, handling partial overlap.

    Tries ``root / rel`` first, then progressively strips leading components
    from *rel* so that paths like ``foo/bar/data.safetensors`` can match a
    blob root that already contains the ``foo/`` prefix.
    """
    parts = rel.parts
    for root in blob_roots:
        for start in range(len(parts)):
            candidate = root / Path(*parts[start:])
            if candidate.exists():
                return candidate
    return None


def load_safetensors(
    definition: Definition,
    workload: Workload,
    blob_roots: Optional[List[Path]] = None,
) -> Dict[str, torch.Tensor]:
    """Load safetensors inputs for a workload.

    Safetensors blobs are resolved from the staging directory (passed via
    ``blob_roots``).  The first root whose ``root / path`` exists is used.
    If none match, the path is passed as-is to safetensors (which will raise
    a FileNotFoundError with a clear message).
    """
    try:
        import safetensors.torch as st
    except Exception as e:
        raise RuntimeError(
            "safetensors is not available in the current environment"
        ) from e

    expected = definition.get_input_shapes(workload.axes)

    safe_tensors: Dict[str, torch.Tensor] = {}
    loaded_files: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, input_spec in workload.inputs.items():
        if input_spec.type != "safetensors":
            continue

        path = input_spec.path
        if not Path(path).is_absolute() and blob_roots:
            resolved = _resolve_blob_path(Path(path), blob_roots)
            if resolved is not None:
                path = str(resolved)

        # Resolve to canonical absolute path so different representations
        # of the same file (symlinks, ".." components) share a cache entry.
        path = str(Path(path).resolve())

        if path not in loaded_files:
            loaded_files[path] = st.load_file(path)
        tensors = loaded_files[path]
        if input_spec.tensor_key not in tensors:
            raise ValueError(f"Missing key '{input_spec.tensor_key}' in '{path}'")
        t = tensors[input_spec.tensor_key]
        if tuple(t.shape) != expected[name]:
            raise ValueError(f"'{name}' expected {expected[name]}, got {list(t.shape)}")
        expect_dtype = dtype_str_to_torch_dtype(definition.inputs[name].dtype)
        if t.dtype != expect_dtype:
            raise ValueError(f"'{name}' expected {expect_dtype}, got {t.dtype}")

        try:
            t = t.contiguous().pin_memory()
        except Exception:
            t = t.contiguous()
        safe_tensors[name] = t
    return safe_tensors


# ── Input generation ─────────────────────────────────────────────────────────


def gen_inputs(
    definition: Definition,
    workload: Workload,
    device: str,
    safe_tensors: Optional[Dict[str, torch.Tensor]] = None,
    custom_inputs_fn: Optional[Any] = None,
    *,
    audit_changes: list[Callable[[], Any]] | None = None,
) -> List[Any]:
    """Generate input tensors in definition order.

    Returns a list of input values (tensors or scalars) in the same order
    as definition.inputs.

    custom_inputs_fn: if provided, called each invocation to regenerate
        custom tensors instead of reusing *custom_tensors*.  Signature:
        ``(axes_and_scalars: dict, device: torch.device) -> dict[str, Tensor]``.
    audit_changes: when supplied, collect in-place variations of generated
        random inputs for a second correctness check. Never alter fixtures,
        custom inputs or explicit workload scalars.
    """
    shapes = definition.get_input_shapes(workload.axes)
    dev = torch.device(device)
    out: List[Any] = []
    custom_tensors = None

    # Regenerate custom tensors on the fly when a factory is provided.
    if custom_inputs_fn is not None:
        axes_and_scalars = {
            **definition.get_resolved_axes_values(workload.axes),
            **workload.get_scalar_inputs(),
        }
        custom_tensors = custom_inputs_fn(axes_and_scalars, dev)

    for name, spec in definition.inputs.items():
        dtype = dtype_str_to_torch_dtype(spec.dtype)

        if name in workload.inputs and isinstance(
            workload.inputs[name], SafetensorsInput
        ):
            if safe_tensors is None or name not in safe_tensors:
                raise RuntimeError(f"Missing required safetensors input '{name}'")
            t_cpu = safe_tensors[name]
            out.append(t_cpu.to(device=dev, non_blocking=True))
        elif name in workload.inputs and isinstance(workload.inputs[name], ScalarInput):
            out.append(workload.inputs[name].value)
        elif name in workload.inputs and isinstance(workload.inputs[name], CustomInput):
            if custom_tensors is None or name not in custom_tensors:
                raise RuntimeError(
                    f"CustomInput for '{name}' must be pre-generated by caller via "
                    f"definition.custom_inputs_entrypoint"
                )
            val = custom_tensors[name]
            if isinstance(val, torch.Tensor):
                out.append(val.to(device=dev, non_blocking=True))
            else:
                # Scalar values (int, float, bool) from custom_inputs_entrypoint
                out.append(val)
        else:  # random
            shape = shapes[name]

            if shape is None:
                value = _rand_tensor((), dtype, dev).item()
            else:
                value = _generate_heuristic_tensor(
                    name,
                    tuple(shape),
                    dtype,
                    dev,
                    spec.description,
                    audit_changes=audit_changes,
                )
                if value is None:
                    value = _rand_tensor(shape, dtype, dev)

                if is_sampling_operation(definition) and name == "probs":
                    value = torch.softmax(value, dim=-1)

            out.append(value)
    return out


def timing_input_stream(
    generate: Callable[[], list],
    device: str,
    *,
    record: list[int] | None = None,
    replay: Iterator[int] | None = None,
    _entropy=urandom,
) -> Callable[[], list]:
    """Draw a fresh OS seed per call, or replay seeds from completed samples.

    Custom factories using private generators must seed those deterministically.
    """
    dev = torch.device(device)
    devices = [dev] if dev.type == "cuda" else []
    seen = set()

    def next_inputs():
        if replay is not None:
            sample_seed = next(replay)
        else:
            if urandom is not _entropy:
                raise RewardHackDetected("Input sampling entropy source was replaced")
            for _ in range(2):
                sample_seed = int.from_bytes(_entropy(8), "little") >> 1
                if sample_seed not in seen:
                    seen.add(sample_seed)
                    break
            else:
                raise RuntimeError(
                    "Input sampling repeatedly produced a duplicate seed"
                )
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            with torch.random.fork_rng(devices=devices):
                random.seed(sample_seed)
                np.random.seed(sample_seed % 2**32)
                torch.default_generator.manual_seed(sample_seed)
                if devices:
                    with torch.cuda.device(dev):
                        torch.cuda.default_generators[
                            torch.cuda.current_device()
                        ].manual_seed(sample_seed)
                inputs = generate()
                if record is not None:
                    record.append(sample_seed)
                return inputs
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    return next_inputs


class TimingInputCheck:
    """Check for repeated draws of built-in inputs that varied in all correctness rounds.

    Clone at most 16 elements before the call; batch host reads outside timing.
    Custom factories and literals have no contract that their values must vary.
    """

    def __init__(self, definition: Definition, workload: Workload):
        self.samples = {
            (pos, name): []
            for pos, name in enumerate(definition.inputs)
            if not isinstance(
                workload.inputs.get(name), (ScalarInput, SafetensorsInput, CustomInput)
            )
        }
        self._wide_baselines: dict[tuple[int, str], set[bytes]] = {}

    def observe(self, inputs):
        for (pos, _), samples in self.samples.items():
            value = inputs[pos]
            if not isinstance(value, torch.Tensor):
                samples.append(value)
                continue
            if value.ndim and value.numel():
                axis = max(range(value.ndim), key=value.size)
                index = [0] * value.ndim
                index[axis] = slice(None, None, (value.size(axis) + 15) // 16)
                line = value[tuple(index)]
                if line.stride(0) == 0:
                    line = line[:1]
            else:
                line = value.reshape(-1)
            if line.element_size() == 1:
                line = line.view(torch.uint8)
            probe = line.detach().clone(memory_format=torch.contiguous_format)
            samples.append((value.shape, probe))
        return inputs

    @staticmethod
    def _values(samples):
        if not samples or not isinstance(samples[0], tuple):
            return samples
        input_shape, first = samples[0]
        if any(
            shape != input_shape
            or probe.shape != first.shape
            or probe.dtype != first.dtype
            for shape, probe in samples
        ):
            return []
        rows = torch.stack([probe for _, probe in samples])
        return [row.tobytes() for row in rows.view(torch.uint8).cpu().numpy()]

    @staticmethod
    def _is_wide(samples, values):
        """At least eight sampled elements each took a distinct value on every draw."""
        if not isinstance(samples[0], tuple):
            return False
        width = samples[0][1].element_size()
        varying_positions = sum(
            len({row[start : start + width] for row in values}) == len(values)
            for start in range(0, len(values[0]), width)
        )
        return varying_positions >= 8

    def monitor(self, inputs):
        self._correctness_rounds = len(next(iter(self.samples.values()), ()))
        monitored = {}
        for key, samples in self.samples.items():
            values = self._values(samples)
            unique = set(values)
            if len(samples) < 4 or len(unique) != len(samples):
                continue
            monitored[key] = []
            if self._is_wide(samples, values):
                self._wide_baselines[key] = unique
        self.samples = monitored
        if not self.samples or not callable(inputs):
            return inputs
        return lambda: self.observe(inputs())

    def check(self):
        """Reject any reuse for wide probes, but allow natural collisions for narrow ones.

        On longer runs a small input domain can naturally repeat individual values.
        Count distinct four-sample sequences to detect cycling beyond that baseline.
        """
        while self.samples:
            key, samples = self.samples.popitem()
            name = key[1]
            values = self._values(samples)
            count = len(values)
            unique = set(values)
            distinct = len(unique)
            if key in self._wide_baselines:
                reused = len(self._wide_baselines.pop(key) & unique)
                if distinct != count or reused:
                    raise RewardHackDetected(
                        f"Timing input '{name}' repeated sampled values "
                        f"({distinct}/{count} distinct samples; {reused} reused from correctness rounds)"
                    )
                continue
            if count < 4 or distinct == count:
                continue
            baseline = min(count, self._correctness_rounds)
            windows = set(zip(values, values[1:], values[2:], values[3:]))
            window_count = count - 3
            if (
                baseline - distinct > max(1, baseline // 10)
                or window_count - len(windows) > max(1, window_count // 10)
                or any(len(set(window)) == 1 for window in windows)
            ):
                raise RewardHackDetected(
                    f"Timing input '{name}' repeated sampled values after varying on "
                    f"every correctness round ({distinct}/{count} distinct samples; "
                    f"{len(windows)}/{window_count} distinct four-sample sequences)"
                )


def prepare_timing_inputs(
    definition: Definition,
    workload: Workload,
    generate: Callable[[], list],
    device: str,
):
    """Use fixed CUPTI discovery for literals/fixtures; otherwise replay sampled seeds."""
    if all(
        isinstance(workload.inputs.get(name), (ScalarInput, SafetensorsInput))
        for name in definition.inputs
    ):
        inputs = generate()
        return inputs, inputs
    seeds: list[int] = []
    return (
        timing_input_stream(generate, device, record=seeds),
        timing_input_stream(generate, device, replay=iter(seeds)),
    )


# ── output generation ────────────────────────────────────────────────────────


def normalize_outputs(
    out: Any,
    *,
    device: torch.device,
    output_names: List[str],
    output_dtypes: Dict[str, torch.dtype],
) -> Dict[str, torch.Tensor]:
    def to_tensor(name: str, v: Any) -> torch.Tensor:
        if isinstance(v, torch.Tensor):
            return v.to(device) if v.device != device else v
        dtype = output_dtypes[name]
        return torch.tensor(v, dtype=dtype, device=device)

    if isinstance(out, dict):
        return {k: to_tensor(k, v) for k, v in out.items() if k in output_dtypes}

    if isinstance(out, torch.Tensor):
        if len(output_names) != 1:
            raise RuntimeError(
                "Single Tensor returned but multiple outputs are defined"
            )
        name = output_names[0]
        return {name: to_tensor(name, out)}

    if isinstance(out, (int, float, bool)):
        if len(output_names) != 1:
            raise RuntimeError("Scalar returned but multiple outputs are defined")
        name = output_names[0]
        return {name: to_tensor(name, out)}

    if isinstance(out, (tuple, list)):
        if len(out) != len(output_names):
            raise RuntimeError(
                f"Tuple/list has {len(out)} elements but {len(output_names)} outputs expected"
            )
        return {name: to_tensor(name, val) for name, val in zip(output_names, out)}

    raise RuntimeError(
        "Unexpected return type; must be Tensor, scalar, or dict[name -> Tensor/scalar]"
    )


def allocate_outputs(
    definition: Definition, resolved_axes: dict[str, int], device: str
) -> list[torch.Tensor]:
    """Allocate output tensors based on definition and resolved axis values.

    Parameters
    ----------
    definition : Definition
        The kernel definition specifying output tensor specs.
    resolved_axes : dict[str, int]
        Concrete values for all variable axes (as returned by
        ``definition.get_resolved_axes_values(workload.axes)``).
    device : str
        The device to allocate tensors on (e.g., "cuda:0").

    Returns
    -------
    list[torch.Tensor]
        List of allocated (uninitialized) output tensors in definition order.
    """
    output_shapes = list(definition.get_output_shapes(resolved_axes).values())

    dtypes = definition.torch_output_dtypes
    return [
        torch.zeros(shape, dtype=dtype, device=device)
        for shape, dtype in zip(output_shapes, dtypes)
    ]


# ── Memory pool allocator ───────────────────────────────────────────────────
class ShiftingMemoryPoolAllocator:
    """Pre-allocated memory pool that provides unique ``data_ptr`` per iteration.

    Allocates a buffer only slightly larger than the input tensors
    (overhead ≈ ``total_iterations × 2048`` bytes per tensor). Each call
    copies the current sample into an advancing offset of the pool without
    retaining a full input batch per iteration.

    For growing input spans, replacement pools at least double in capacity.
    Old pools remain live for pointer uniqueness; together they occupy less
    than twice the largest pool per input. Tensor dtype and device must stay fixed.

    The advance is a seeded random multiple of ``_POOL_ALIGNMENT``. This
    removes the fixed-stride signal that a submission could use to detect the
    timing loop while keeping runs reproducible.

    Stride patterns (including stride-0 broadcasts from ``expand()``) are
    preserved so that the pool only stores the physical storage footprint,
    not the full logical numel.

    Parameters
    ----------
    inputs : list[Any] or callable
        Fixed values or a factory supplying the current iteration's values.
    outputs : list[torch.Tensor]
        Output tensors for destination-passing-style kernels, as returned
        by :func:`allocate_outputs`.  These are zero-filled at each offset
        on every call.  Pass an empty list for non-DPS kernels.
    total_iterations : int
        Total number of :meth:`get_unique_args` calls expected
        (warmup + timed iterations).
    seed : int
        Seed for the per-iteration shift sequence.
    """

    # Tensor alignment in bytes – shift by a random multiple each iteration.
    _POOL_ALIGNMENT = 256
    _MAX_SHIFT_BYTES = 2048

    def __init__(
        self,
        inputs: List[Any] | Callable[[], List[Any]],
        outputs: List[torch.Tensor],
        total_iterations: int,
        seed: int = 0,
    ) -> None:
        self._call_idx = 0
        self._total_iterations = total_iterations
        self._inputs = inputs
        self._input_entries: List[Dict[str, Any]] = []
        self._output_entries: List[Dict[str, Any]] = []
        self._live_rebased_tensors: List[torch.Tensor] = []
        self._offset_blocks = self._compute_offset_blocks(total_iterations, seed)

        if not callable(inputs):
            self._input_entries = [
                self._make_pool_entry(inp) if isinstance(inp, torch.Tensor) else {}
                for inp in inputs
            ]

        for out in outputs:
            self._output_entries.append(self._make_pool_entry(out))

    @classmethod
    def _compute_offset_blocks(cls, total_iterations: int, seed: int) -> List[int]:
        """Return reproducible cumulative alignment-block offsets."""
        max_multiplier = cls._MAX_SHIFT_BYTES // cls._POOL_ALIGNMENT
        rng = random.Random(seed)
        offsets = [0]
        for _ in range(max(0, total_iterations - 1)):
            offsets.append(offsets[-1] + rng.randint(1, max_multiplier))
        return offsets

    def _make_pool_entry(
        self, tensor: torch.Tensor, min_numel: int = 0
    ) -> Dict[str, Any]:
        shape = tuple(tensor.shape)
        strides = tensor.stride()
        storage_span = _storage_span_from_first_element(tensor)
        elem_size = tensor.element_size()

        block_numel = max(1, self._POOL_ALIGNMENT // elem_size)

        padding = self._offset_blocks[-1] * block_numel
        pool_numel = max(storage_span + padding, min_numel)
        pool = torch.empty(pool_numel, dtype=tensor.dtype, device=tensor.device)

        return {
            "pool": pool,
            "capacity": pool_numel - padding,
            "shape": shape,
            "strides": strides,
            "storage_span": storage_span,
            "block_numel": block_numel,
        }

    def get_unique_args(self) -> List[Any]:
        """Copy source data into the next pool offset and return views.

        Returns inputs followed by zero-filled outputs (for DPS kernels).
        Each call advances the internal offset so every returned tensor has
        a unique ``data_ptr``.  Memory between consecutive calls overlaps —
        only the starting address differs.
        """
        if self._call_idx >= self._total_iterations:
            raise RuntimeError(
                f"ShiftingMemoryPoolAllocator exhausted: called {self._call_idx + 1} "
                f"times but was allocated for {self._total_iterations} iterations"
            )

        result: List[Any] = []
        idx = self._call_idx
        block_offset = self._offset_blocks[idx]

        inputs = self._inputs() if callable(self._inputs) else self._inputs
        if idx == 0 and callable(self._inputs):
            self._input_entries = [{} for _ in inputs]
        for pos, (inp, entry) in enumerate(
            zip(inputs, self._input_entries, strict=True)
        ):
            if not isinstance(inp, torch.Tensor):
                result.append(inp)
                continue

            span = _storage_span_from_first_element(inp)
            if not entry:
                entry = self._input_entries[pos] = self._make_pool_entry(inp)
            elif entry["pool"].dtype != inp.dtype or entry["pool"].device != inp.device:
                raise ValueError(f"Timing input {pos} changed dtype or device")
            elif span > entry["capacity"]:
                old_pool = entry["pool"]
                entry = self._input_entries[pos] = self._make_pool_entry(
                    inp, 2 * old_pool.numel()
                )
                self._live_rebased_tensors.append(old_pool)
            start = block_offset * entry["block_numel"]
            entry["pool"].narrow(0, start, span).copy_(
                inp.detach().as_strided((span,), (1,))
            )
            shifted = _rebase_tensor_storage(
                entry["pool"].as_strided(inp.shape, inp.stride(), start)
            )
            self._live_rebased_tensors.append(shifted)
            result.append(shifted)

        for entry in self._output_entries:
            start = block_offset * entry["block_numel"]
            entry["pool"].narrow(0, start, entry["storage_span"]).zero_()
            shifted = _rebase_tensor_storage(
                entry["pool"].as_strided(entry["shape"], entry["strides"], start)
            )
            self._live_rebased_tensors.append(shifted)
            result.append(shifted)

        self._call_idx += 1
        return result
