"""Default evaluator for general kernel correctness and performance."""

import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

import torch

from ..config import BenchmarkConfig
from .evaluator import Evaluator
from ..runner.runner import BaselineHandle, DeviceBaseline
from ..timing import time_runnable
from ..utils import (
    compute_error_stats,
    gen_inputs,
    load_safetensors,
    make_eval,
)
from ..compile import BuilderRegistry, Runnable, RunnableInputs
from ...data import (
    Correctness,
    Definition,
    Evaluation,
    EvaluationStatus,
    Performance,
    SupportedHardware,
    Workload,
)
from ...data.workload import CorrectnessSpec
from .utils import allocate_outputs, normalize_result


class DefaultEvaluator(Evaluator):
    @classmethod
    def can_evaluate(cls, definition: Definition) -> bool:
        return True

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
        # Reference is always value-returning style
        ref_runnable = BuilderRegistry.get_instance().build_reference(definition)
        loaded_safe_tensors = (
            load_safetensors(definition, workload, trace_set_root)
            if any(d.type == "safetensors" for d in workload.inputs.values())
            else {}
        )

        inputs: list[RunnableInputs] = []
        outputs: list[list[torch.Tensor]] = []

        for _ in range(cfg.num_trials):
            try:
                inp = gen_inputs(
                    definition,
                    workload,
                    ref_runnable,
                    device=device,
                    safe_tensors=loaded_safe_tensors,
                )
                inputs.append(inp)

                with torch.no_grad():
                    result = ref_runnable(*inp)
                torch.cuda.synchronize(device)
                outputs.append(normalize_result(definition, result, device))
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Reference code execution failed: {e}")

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
                    lock_clocks=cfg.lock_clocks,
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
        max_abs = 0.0
        max_rel = 0.0
        numerical_incorrect = False
        is_dps = sol_runnable.metadata.destination_passing_style

        for trial, inp in enumerate(inputs):
            try:
                if is_dps:
                    # DPS style: allocate outputs and call with them
                    out = allocate_outputs(definition, inp.resolved_axes, device)
                    with torch.no_grad():
                        sol_runnable(*inp, *out)
                    torch.cuda.synchronize(device)
                else:
                    # Value-returning style: call and normalize result
                    with torch.no_grad():
                        result = sol_runnable(*inp)
                    torch.cuda.synchronize(device)
                    out = normalize_result(definition, result, device)
            except Exception:
                traceback.print_exc()
                return None, make_eval(
                    status=EvaluationStatus.RUNTIME_ERROR,
                    hardware=hardware,
                    log_path=log_path,
                )

            ref_out = ref_outputs[trial]

            for sol_tensor, ref_tensor in zip(out, ref_out):
                # Shape validation
                if tuple(sol_tensor.shape) != tuple(ref_tensor.shape):
                    print(f"Incorrect shape: {sol_tensor.shape} != {ref_tensor.shape}")
                    return None, make_eval(
                        status=EvaluationStatus.INCORRECT_SHAPE,
                        hardware=hardware,
                        log_path=log_path,
                    )

                # Dtype validation
                if sol_tensor.dtype != ref_tensor.dtype:
                    print(f"Incorrect dtype: {sol_tensor.dtype} != {ref_tensor.dtype}")
                    return None, make_eval(
                        status=EvaluationStatus.INCORRECT_DTYPE,
                        hardware=hardware,
                        log_path=log_path,
                    )

                # Non-finite values check
                non_finite_err_val = None
                if torch.isinf(sol_tensor).any().item():
                    non_finite_err_val = float("inf")
                elif torch.isnan(sol_tensor).any().item():
                    non_finite_err_val = float("nan")

                if non_finite_err_val is not None:
                    correctness = Correctness(
                        max_relative_error=non_finite_err_val,
                        max_absolute_error=non_finite_err_val,
                    )
                    return correctness, make_eval(
                        status=EvaluationStatus.INCORRECT_NUMERICAL,
                        hardware=hardware,
                        log_path=log_path,
                        correctness=correctness,
                    )

                # Compute error statistics
                abs_err, rel_err, exceeds_tol, _ = compute_error_stats(
                    sol_tensor, ref_tensor, correctness
                )

                if exceeds_tol:
                    numerical_incorrect = True

                max_abs = max(max_abs, abs_err)
                max_rel = max(max_rel, rel_err)

        correctness = Correctness(
            max_relative_error=max_rel, max_absolute_error=max_abs
        )

        if numerical_incorrect:
            return correctness, make_eval(
                status=EvaluationStatus.INCORRECT_NUMERICAL,
                hardware=hardware,
                log_path=log_path,
                correctness=correctness,
            )

        return correctness, None

    @classmethod
    def eval_performance(
        cls,
        definition: Definition,
        sol_runnable: Runnable,
        inputs: list[RunnableInputs],
        ref_mean_latency_ms: float,
        cfg: BenchmarkConfig,
        log_path: str,
        device: str,
        execution_device: Any,
        hardware: SupportedHardware = SupportedHardware.LOCAL,
    ) -> tuple[Performance, Optional[Evaluation]]:
        sol_latencies: list[float] = []
        is_dps = sol_runnable.metadata.destination_passing_style

        try:
            for inp in inputs:
                if is_dps:
                    # DPS style: allocate outputs and include in args
                    output_tensors = allocate_outputs(
                        definition, inp.resolved_axes, device
                    )
                    args = list(inp) + output_tensors
                else:
                    # Value-returning style
                    args = list(inp)
                ms = time_runnable(
                    sol_runnable,
                    args,
                    cfg.warmup_runs,
                    cfg.iterations,
                    device,
                    execution_device,
                    lock_clocks=cfg.lock_clocks,
                )
                sol_latencies.append(ms)
        except Exception:
            traceback.print_exc()
            return None, make_eval(
                status=EvaluationStatus.RUNTIME_ERROR,
                hardware=hardware,
                log_path=log_path,
            )

        if not sol_latencies:
            print("Failed to collect solution latencies", file=sys.stderr)
            return None, make_eval(
                status=EvaluationStatus.RUNTIME_ERROR,
                hardware=hardware,
                log_path=log_path,
            )

        sol_mean_latency_ms = sum(sol_latencies) / float(len(sol_latencies))
        performance = Performance(
            latency_ms=sol_mean_latency_ms,
            reference_latency_ms=ref_mean_latency_ms,
            speedup_factor=(ref_mean_latency_ms / sol_mean_latency_ms),
        )

        return performance, None
