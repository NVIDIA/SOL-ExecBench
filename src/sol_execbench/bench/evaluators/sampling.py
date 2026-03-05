"""Evaluator for sampling operations with statistical validation."""

import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

import torch
from typing_extensions import override

from ..config import BenchmarkConfig
from ..compile import RunnableInputs
from ..runner.runner import BaselineHandle, DeviceBaseline
from ..timing import time_runnable
from ..utils import (
    compute_error_stats,
    gen_inputs,
    load_safetensors,
    make_eval,
)
from ..compile import BuilderRegistry, Runnable
from ...data import (
    Correctness,
    Definition,
    Evaluation,
    EvaluationStatus,
    SupportedHardware,
    Workload,
)
from ...data.workload import CorrectnessSpec

from .default import DefaultEvaluator
from .reward_hack import RewardHackDetected
from .utils import allocate_outputs, normalize_result


def _get_input_by_name(definition: Definition, inputs: list[Any], name: str) -> Any:
    """Get input value by name from inputs list using definition order."""
    input_names = list(definition.inputs.keys())
    if name not in input_names:
        raise KeyError(f"Input '{name}' not found in definition")
    return inputs[input_names.index(name)]


def _get_inputs_by_names(
    definition: Definition, inputs: list[Any], names: list[str]
) -> dict[str, Any]:
    """Get multiple inputs by name as a dict for convenience."""
    input_names = list(definition.inputs.keys())
    return {
        name: inputs[input_names.index(name)] for name in names if name in input_names
    }


class SamplingEvaluator(DefaultEvaluator):
    @override
    @classmethod
    def can_evaluate(cls, definition: Definition) -> bool:
        return is_sampling_op(definition)

    @override
    @classmethod
    def build_baseline(
        cls,
        definition: Definition,
        workload: Workload,
        cfg: BenchmarkConfig,
        device: str,
        trace_set_root: Optional[Path] = None,
        execution_device: Any = None,
        hardware: SupportedHardware = SupportedHardware.LOCAL,
    ) -> DeviceBaseline:
        ref_runnable = BuilderRegistry.get_instance().build_reference(definition)
        loaded_safe_tensors = (
            load_safetensors(definition, workload, trace_set_root)
            if any(d.type == "safetensors" for d in workload.inputs.values())
            else {}
        )

        inputs: list[list[Any]] = []
        outputs: list[list[torch.Tensor]] = []

        inp = gen_inputs(
            definition,
            workload,
            ref_runnable,
            device=device,
            safe_tensors=loaded_safe_tensors,
        )
        inputs.append(inp)

        thresholding_method = _detect_thresholding_method(definition)
        probs = _get_input_by_name(definition, inp, "probs")
        params = _get_inputs_by_names(definition, inp, ["top_k", "top_p"])
        valid_mask = _compute_valid_sampling_mask(probs, thresholding_method, params)

        masked_probs = probs * valid_mask.float()
        expected_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)

        # Store expected_probs as first (and only) tensor in the list
        outputs.append([expected_probs])

        # Reference is always value-returning style
        if cfg.time_baseline:
            latencies: list[float] = []
            for inp in inputs:
                ms = time_runnable(
                    ref_runnable,
                    inp,
                    cfg.warmup_runs,
                    cfg.iterations,
                    device,
                    execution_device,
                )
                latencies.append(ms)

            mean_latency_ms = sum(latencies) / float(len(latencies))
        else:
            mean_latency_ms = 0.0

        handle = BaselineHandle(uuid.uuid4().hex)

        return DeviceBaseline(
            handle=handle,
            definition=definition,
            device=device,
            inputs=inputs,
            outputs=outputs,
            mean_latency_ms=mean_latency_ms,
            correctness=workload.correctness,
        )

    @override
    @classmethod
    def check_correctness(
        cls,
        definition: Definition,
        sol_runnable: Runnable,
        inputs: list[RunnableInputs],
        ref_outputs: list[list[torch.Tensor]],
        cfg: BenchmarkConfig,
        correctness: CorrectnessSpec,
        log_path: str,
        device: str,
        hardware: SupportedHardware = SupportedHardware.LOCAL,
    ) -> tuple[Optional[Correctness], Optional[Evaluation]]:
        # expected_probs is stored as the first tensor in ref_outputs[0]
        expected_probs = ref_outputs[0][0]
        vocab_size = expected_probs.shape[-1]

        inp = inputs[0]
        probs = _get_input_by_name(definition, inp, "probs")
        params = _get_inputs_by_names(definition, inp, ["top_k", "top_p"])

        # Compute valid sampling mask based on thresholding
        thresholding_method = _detect_thresholding_method(definition)
        valid_mask = _compute_valid_sampling_mask(probs, thresholding_method, params)

        # Validate correct sampling token set
        for _ in range(correctness.sampling_validation_trials):
            try:
                out = cls.run_solution(definition, sol_runnable, inp, device)
            except RewardHackDetected as e:
                print(f"Reward hack detected: {e}")
                return None, make_eval(
                    status=EvaluationStatus.RUNTIME_ERROR,
                    hardware=hardware,
                    log_path=log_path,
                )
            except Exception:
                traceback.print_exc()
                return None, make_eval(
                    status=EvaluationStatus.RUNTIME_ERROR,
                    hardware=hardware,
                    log_path=log_path,
                )

            # Get samples tensor (first output based on definition order)
            samples = out[0]

            # Check vocabulary range
            if (samples < 0).any() or (samples >= vocab_size).any():
                invalid_samples = samples[(samples < 0) | (samples >= vocab_size)]
                correctness = Correctness(
                    max_relative_error=float("inf"), max_absolute_error=float("inf")
                )
                message = f"Samples {invalid_samples.tolist()} out of vocabulary range [0, {vocab_size})"
                print(message, file=sys.stderr)
                return correctness, make_eval(
                    status=EvaluationStatus.INCORRECT_NUMERICAL,
                    device=device,
                    log_path=log_path,
                    correctness=correctness,
                )

            # Validate thresholding - check samples are within valid mask
            if samples.dim() == 0:
                samples_flat = samples.unsqueeze(0)
            else:
                samples_flat = samples.flatten()

            batch_size = valid_mask.shape[0]
            for i in range(len(samples_flat)):
                batch_idx = i % batch_size
                sample_idx = samples_flat[i].item()
                if not valid_mask[batch_idx, sample_idx]:
                    correctness = Correctness(
                        max_relative_error=float("inf"), max_absolute_error=float("inf")
                    )
                    message = f"Sample {sample_idx} is outside valid {thresholding_method} mask for batch {batch_idx}"
                    print(message, file=sys.stderr)
                    return correctness, make_eval(
                        status=EvaluationStatus.INCORRECT_NUMERICAL,
                        device=device,
                        log_path=log_path,
                        correctness=correctness,
                    )

        try:
            sol_freqs = _sample_token_distributions(
                sol_runnable, inp, device, definition, num_trials=500000
            )
            torch.cuda.synchronize(device)
        except Exception:
            traceback.print_exc()
            return None, make_eval(
                status=EvaluationStatus.RUNTIME_ERROR,
                hardware=hardware,
                log_path=log_path,
            )

        batch_size = expected_probs.shape[0]
        tvds = []
        max_abs_errors = []
        max_rel_errors = []

        for i in range(batch_size):
            tvd_i = 0.5 * torch.sum(torch.abs(sol_freqs[i] - expected_probs[i])).item()
            tvds.append(tvd_i)

            max_abs_i, max_rel_i, _, _ = compute_error_stats(
                sol_freqs[i], expected_probs[i], correctness
            )
            max_abs_errors.append(max_abs_i)
            max_rel_errors.append(max_rel_i)

        # Use the worst (max) TVD and errors across all batch elements
        max_tvd = max(tvds)
        max_abs = max(max_abs_errors)
        max_rel = max(max_rel_errors)

        numerical_incorrect = max_tvd > correctness.sampling_tvd_threshold
        correctness = Correctness(
            max_relative_error=max_rel,
            max_absolute_error=max_abs,
            extra={"tvd": max_tvd, "tvds_per_batch": tvds},
        )
        if numerical_incorrect:
            return correctness, make_eval(
                status=EvaluationStatus.INCORRECT_NUMERICAL,
                hardware=hardware,
                log_path=log_path,
                correctness=correctness,
            )

        return correctness, None


def is_sampling_op(definition: Definition) -> bool:
    return getattr(definition, "op_type", None) == "sampling"


def _detect_thresholding_method(definition: Definition) -> str:
    name = definition.name.lower()
    if "top_k_top_p" in name:
        return "top_k_top_p"
    elif "top_k" in name:
        return "top_k"
    elif "top_p" in name:
        return "top_p"
    else:
        return "none"  # no thresholding


def _compute_valid_sampling_mask(
    probs: torch.Tensor, method: str, params: dict[str, Any], eps: float = 5e-2
) -> torch.Tensor:
    """
    For tie-breaking in top_k (allows any token with prob >= k-th largest)
    and numerical precision in top_p (allows tokens within eps of nucleus boundary).
    """
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)

    batch_size, vocab_size = probs.shape
    device = probs.device

    if method == "none":
        return torch.ones((batch_size, vocab_size), dtype=torch.bool, device=device)

    mask = torch.ones((batch_size, vocab_size), dtype=torch.bool, device=device)

    if method in ["top_k", "top_k_top_p"]:
        if "top_k" not in params:
            raise ValueError(f"top_k parameter required for {method} but not found")

        top_k_param = params["top_k"]
        for i in range(batch_size):
            k = (
                int(top_k_param[i].item())
                if top_k_param.dim() > 0
                else int(top_k_param.item())
            )

            if 0 < k < vocab_size:
                sorted_probs, _ = torch.sort(probs[i], descending=True)
                # k-th largest value (0-indexed, so k-1)
                pivot = sorted_probs[k - 1]
                mask[i] = probs[i] >= pivot  # tie-breaking handling

    # Apply top_p mask with epsilon tolerance
    if method in ["top_p", "top_k_top_p"]:
        if "top_p" not in params:
            raise ValueError(f"top_p parameter required for {method} but not found")

        top_p_param = params["top_p"]
        for i in range(batch_size):
            p = (
                float(top_p_param[i].item())
                if top_p_param.dim() > 0
                else float(top_p_param.item())
            )

            if 0 < p < 1:
                sorted_probs, sorted_indices = torch.sort(probs[i], descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=0)

                # Find tokens in nucleus (cumsum <= p + eps for numerical tolerance)
                nucleus_mask = cumsum <= (p + eps)

                if not nucleus_mask.any():
                    nucleus_mask[0] = True

                # Map back to original indices
                p_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
                p_mask[sorted_indices[nucleus_mask]] = True

                mask[i] = mask[i] & p_mask

    return mask


def _sample_token_distributions(
    runnable: Runnable,
    inputs: list[Any],
    device: str,
    definition: Definition,
    num_trials: int = 500000,
) -> torch.Tensor:
    probs = _get_input_by_name(definition, inputs, "probs")
    original_batch_size = probs.shape[0] if probs.dim() > 1 else 1
    vocab_size = probs.shape[-1]
    is_dps = runnable.metadata.destination_passing_style

    # Repeat entire input batch to fill up to target_batch_size for efficient sampling
    target_batch_size = 10000
    repeat_count = target_batch_size // original_batch_size
    actual_batch_size = repeat_count * original_batch_size

    input_names = list(definition.inputs.keys())
    padded_inputs: list[Any] = []
    for name, value in zip(input_names, inputs):
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            if name == "probs":
                # For probs, repeat the entire batch
                if value.dim() == 1:
                    value = value.unsqueeze(0)
                # Repeat the entire batch repeat_count times
                padded_value = value.repeat(repeat_count, *([1] * (value.dim() - 1)))
            elif name in ["top_k", "top_p"]:
                # For sampling parameters, repeat the entire batch
                if value.dim() == 0:
                    padded_value = value.unsqueeze(0).repeat(actual_batch_size)
                else:
                    padded_value = value.repeat(repeat_count)
            else:
                # For other tensors, repeat entire batch along batch dimension
                if value.dim() == 0:
                    padded_value = value.unsqueeze(0).repeat(actual_batch_size)
                else:
                    padded_value = value.repeat(
                        repeat_count, *([1] * (value.dim() - 1))
                    )
            padded_inputs.append(padded_value)
        else:
            # For non-tensor inputs, keep as is
            padded_inputs.append(value)

    counters = torch.zeros(
        (original_batch_size, vocab_size),
        dtype=torch.int64,
        device=torch.device(device),
    )

    trials_needed = (num_trials + repeat_count - 1) // repeat_count
    total_samples_per_batch = 0

    for _ in range(trials_needed):
        if is_dps:
            out = allocate_outputs(definition, padded_inputs, device)
            with torch.no_grad():
                runnable(*padded_inputs, *out)
        else:
            with torch.no_grad():
                result = runnable(*padded_inputs)
            out = normalize_result(definition, result, device)

        # Get samples tensor (first output based on definition order)
        samples = out[0]

        if samples.dim() == 0:
            # Single sample - assign to first batch element
            sample_idx = samples.item()
            counters[0, sample_idx] += 1
            total_samples_per_batch += 1
        else:
            # slice and accumulate per original batch element
            samples_flat = samples.flatten()
            for i in range(samples_flat.numel()):
                batch_idx = i % original_batch_size
                sample_idx = samples_flat[i].item()
                counters[batch_idx, sample_idx] += 1
            total_samples_per_batch += repeat_count

    # [batch_size, vocab_size]
    frequencies = counters.float() / total_samples_per_batch
    return frequencies
