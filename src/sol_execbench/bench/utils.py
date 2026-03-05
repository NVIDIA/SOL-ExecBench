"""Utility functions for benchmark execution."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch

from ..data import (
    Correctness,
    Definition,
    Evaluation,
    EvaluationStatus,
    Performance,
    SupportedHardware,
    Workload,
)
from ..data.workload import CorrectnessSpec
from .compile.runnable import Runnable, RunnableInputs
from ..data.workload import RandomInput, ScalarInput, SafetensorsInput, CustomInput
from ..data.dtypes import dtype_str_to_torch_dtype
from ..utils import env_snapshot, flush_stdio_streams


def _cast_to_fp4x2(x: torch.Tensor) -> torch.Tensor:
    """Quantize a tensor to FP4 E2M1 and pack into uint8 (2 FP4 values per byte).

    Args:
        x: Input tensor of shape (..., cols) with values in range [-6, 6]

    Returns:
        uint8 tensor of shape (..., cols//2) with packed FP4 values
    """
    # FP4 E2M1 encoding
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

    # Pack two FP4 values into one byte along cols dimension (dim 1)
    # Input shape: (..., cols)
    # first value (even cols) in low nibble, second value (odd cols) in high nibble
    packed = result[..., ::2] + result[..., 1::2] * 16
    return packed.view(torch.float4_e2m1fn_x2)


def _rand_tensor(
    shape: list[int], dtype: torch.dtype, device: torch.device
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


def normalize_outputs(
    out: Any,
    *,
    device: torch.device,
    output_names: list[str],
    output_dtypes: dict[str, torch.dtype],
) -> dict[str, torch.Tensor]:
    def to_tensor(name: str, v: Any) -> torch.Tensor:
        if isinstance(v, torch.Tensor):
            return v.to(device) if v.device != device else v
        dtype = output_dtypes[name]
        # Python scalar -> 0-D tensor for comparison
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


def compute_error_stats(
    output: torch.Tensor, reference: torch.Tensor, correctness: CorrectnessSpec
) -> tuple[float, float, bool, float]:
    x = output.to(torch.float32)
    y = reference.to(torch.float32)

    eps = 1e-8
    abs_error = torch.abs(x - y)
    rel_error = abs_error / (torch.abs(y) + eps)

    total_elements = abs_error.numel()
    if total_elements == 0:
        return 0.0, 0.0, False, 1.0

    exceeds_tol_mask = (abs_error > correctness.max_atol) & (
        rel_error > correctness.max_rtol
    )
    exceeds_count = float(exceeds_tol_mask.sum().item())
    matched_ratio = 1.0 - (exceeds_count / float(total_elements))
    matched_ratio = max(0.0, min(1.0, matched_ratio))

    exceeds_tol = matched_ratio < correctness.required_matched_ratio

    max_abs = float(abs_error.max().item())
    max_rel = float(rel_error.max().item())

    return max_abs, max_rel, exceeds_tol, matched_ratio


def is_sampling_operation(definition: Definition) -> bool:
    return getattr(definition, "op_type", None) == "sampling"


def compute_frequency_distribution(
    runnable: Any,
    inputs: list[dict[str, Any]],
    device: str,
    definition: Definition,
    num_trials: int = 10000,
) -> torch.Tensor:
    inp = inputs[0]

    workload_batch_size = inp["probs"].shape[0] if inp["probs"].dim() > 1 else 1
    vocab_size = inp["probs"].shape[-1]
    counter = torch.zeros(vocab_size, dtype=torch.int64, device=torch.device(device))

    trials_needed = (num_trials + workload_batch_size - 1) // workload_batch_size
    total_samples_collected = 0

    for trial in range(trials_needed):
        with torch.no_grad():
            out = runnable(**inp)

        output_names = list(definition.outputs.keys())
        output_dtypes = {
            k: dtype_str_to_torch_dtype(v.dtype) for k, v in definition.outputs.items()
        }

        out_normalized = normalize_outputs(
            out,
            device=torch.device(device),
            output_names=output_names,
            output_dtypes=output_dtypes,
        )

        samples = out_normalized["samples"]

        if samples.dim() == 0:
            sample_idx = samples.item()
            counter[sample_idx] += 1
            total_samples_collected += 1
        else:  # Batch of samples
            for i in range(samples.numel()):
                sample_idx = samples.flatten()[i].item()
                counter[sample_idx] += 1
                total_samples_collected += 1

    frequency = counter.float() / total_samples_collected
    return frequency


def load_safetensors(
    definition: Definition, workload: Workload, trace_set_root: Optional[Path] = None
) -> dict[str, torch.Tensor]:
    try:
        import safetensors.torch as st
    except Exception as e:
        raise RuntimeError(
            "safetensors is not available in the current environment"
        ) from e

    shapes_list = list(definition.get_input_shapes(workload.axes).values())
    input_names = list(definition.inputs.keys())
    expected = dict(zip(input_names, shapes_list))

    safe_tensors: dict[str, torch.Tensor] = {}
    for name, input_spec in workload.inputs.items():
        if input_spec.type != "safetensors":
            continue

        path = input_spec.path
        if trace_set_root is not None and not Path(path).is_absolute():
            path = str(trace_set_root / path)

        tensors = st.load_file(path)
        if input_spec.tensor_key not in tensors:
            raise ValueError(f"Missing key '{input_spec.tensor_key}' in '{path}'")
        t = tensors[input_spec.tensor_key]
        # shape check
        if tuple(t.shape) != expected[name]:
            raise ValueError(f"'{name}' expected {expected[name]}, got {list(t.shape)}")
        # dtype check
        expect_dtype = dtype_str_to_torch_dtype(definition.inputs[name].dtype)
        if t.dtype != expect_dtype:
            raise ValueError(f"'{name}' expected {expect_dtype}, got {t.dtype}")

        try:
            t = t.contiguous().pin_memory()
        except Exception:
            t = t.contiguous()
        safe_tensors[name] = t
    return safe_tensors


def gen_inputs(
    definition: Definition,
    workload: Workload,
    runnable: Runnable,
    device: str,
    safe_tensors: Optional[dict[str, torch.Tensor]] = None,
) -> RunnableInputs:
    # check that entrypoint is set if any custom inputs are present
    if definition.custom_inputs_entrypoint is None and any(
        isinstance(input, CustomInput) for input in workload.inputs.values()
    ):
        raise ValueError(
            "Definition.custom_inputs_entrypoint must be set when custom inputs are present"
        )

    if definition.custom_inputs_entrypoint is None:
        return gen_explicit_inputs(definition, workload, device, safe_tensors)
    return gen_custom_inputs(definition, workload, runnable, device)


def gen_custom_inputs(
    definition: Definition,
    workload: Workload,
    runnable: Runnable,
    device: str,
) -> RunnableInputs:
    """Generate custom inputs for the workload."""
    axes_values = definition.get_resolved_axes_values(workload.axes)
    scalar_inputs = workload.get_scalar_inputs()
    axes_and_scalars = {**axes_values, **scalar_inputs}
    custom_inputs = runnable.gen_inputs(axes_and_scalars, torch.device(device))
    if not isinstance(custom_inputs, dict):
        raise ValueError(
            f"Custom inputs must be a dictionary. Got: {type(custom_inputs)}"
        )
    if len(custom_inputs) != len(definition.inputs):
        raise ValueError(
            f"Custom inputs count mismatch. Custom: {len(custom_inputs)}. Definition: {len(definition.inputs)}"
        )

    # reorder custom_inputs to match the definition.inputs order
    inputs = []
    for name in definition.inputs.keys():
        if name not in custom_inputs:
            raise ValueError(
                f"Received input with '{name}' that was not specified in the definition!"
            )
        inputs.append(custom_inputs[name])

    # verify each input's shape
    definition_shapes = definition.get_input_shapes(axes_values)
    assert len(definition_shapes) == len(custom_inputs) == len(definition.inputs)
    for name, shape in definition_shapes.items():
        if shape is None and not isinstance(custom_inputs[name], (int, float, bool)):
            raise ValueError(
                f"Received input with '{name}' that is not a scalar! Got: {type(custom_inputs[name])}"
            )
        if shape is not None and not isinstance(custom_inputs[name], torch.Tensor):
            raise ValueError(
                f"Received input with '{name}' that is not a tensor! Got: {type(custom_inputs[name])}"
            )
        if shape is not None and tuple(custom_inputs[name].shape) != tuple(shape):
            raise ValueError(
                f"Received input with '{name}' that has shape {custom_inputs[name].shape} but the the expected shape is {shape}"
            )

    return RunnableInputs(callable_inputs=inputs, resolved_axes=axes_values)


def gen_explicit_inputs(
    definition: Definition,
    workload: Workload,
    device: str,
    safe_tensors: Optional[dict[str, torch.Tensor]] = None,
) -> RunnableInputs:
    """Generate input tensors in definition order.

    Returns a list of input values (tensors or scalars) in the same order
    as definition.inputs.
    """
    full_axes_values = definition.get_resolved_axes_values(workload.axes)
    shapes = list(definition.get_input_shapes(workload.axes).values())
    dev = torch.device(device)

    inputs = []
    for idx, (name, spec) in enumerate(definition.inputs.items()):
        dtype = dtype_str_to_torch_dtype(spec.dtype)

        if name not in workload.inputs:
            continue

        if isinstance(workload.inputs[name], SafetensorsInput):
            if safe_tensors is None or name not in safe_tensors:
                raise RuntimeError(f"Missing required safetensors input '{name}'")
            t_cpu = safe_tensors[name]
            inputs.append(t_cpu.to(device=dev, non_blocking=True))
        elif isinstance(workload.inputs[name], ScalarInput):
            inputs.append(workload.inputs[name].value)
        elif isinstance(workload.inputs[name], RandomInput):
            shape = shapes[idx]
            if shape is None:
                value = _rand_tensor((), dtype, dev).item()
            else:
                value = _rand_tensor(shape, dtype, dev)

                if is_sampling_operation(definition) and name == "probs":
                    value = torch.softmax(
                        value, dim=-1
                    )  # convert logits to probs for sampling

            inputs.append(value)
        else:
            raise RuntimeError(f"Unsupported input type: {type(workload.inputs[name])}")
    return RunnableInputs(callable_inputs=inputs, resolved_axes=full_axes_values)


_MAX_EMBEDDED_LOG_BYTES = 5 * 1024 * 1024


def _read_log_file(
    log_path: Optional[str], *, limit: int = _MAX_EMBEDDED_LOG_BYTES
) -> Optional[str]:
    if not log_path:
        return None

    flush_stdio_streams()

    try:
        with open(log_path, "rb") as fh:
            data = fh.read(limit + 1)
    except FileNotFoundError:
        return None
    except OSError:
        return None

    truncated = len(data) > limit
    if truncated:
        data = data[:limit]

    text = data.decode("utf-8", errors="replace")
    if truncated:
        text += "\n\n[log truncated]\n"
    return text


def make_eval(
    status: EvaluationStatus,
    hardware: SupportedHardware,
    log_path: Optional[str],
    correctness: Optional[Correctness] = None,
    performance: Optional[Performance] = None,
    extra_msg: Optional[str] = None,
) -> Evaluation:
    log_text = _read_log_file(log_path) or ""
    if extra_msg:
        log_text = log_text + "\n" + extra_msg if log_text else extra_msg
    return Evaluation(
        status=status,
        log=log_text,
        environment=env_snapshot(hardware),
        timestamp=datetime.now().isoformat(),
        correctness=correctness,
        performance=performance,
    )
