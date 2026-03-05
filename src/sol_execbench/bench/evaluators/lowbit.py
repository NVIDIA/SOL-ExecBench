"""Evaluator for low-bit quantized kernels with relaxed tolerances."""

from __future__ import annotations

import traceback
from typing import Optional

import torch
from typing_extensions import override

from ..config import BenchmarkConfig
from ..utils import compute_error_stats, make_eval
from ..compile import Runnable, RunnableInputs
from ...data import (
    Correctness,
    Definition,
    Evaluation,
    EvaluationStatus,
    SupportedHardware,
)
from ...data.workload import CorrectnessSpec

from .default import DefaultEvaluator
from .utils import allocate_outputs, normalize_result


class LowBitEvaluator(DefaultEvaluator):
    @override
    @classmethod
    def can_evaluate(cls, definition: Definition) -> bool:
        return "moe_fp8_block_scale" in definition.name

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
        max_abs = 0.0
        max_rel = 0.0
        numerical_incorrect = False
        min_matched_ratio = 1.0
        is_dps = sol_runnable.metadata.destination_passing_style

        for trial, inp in enumerate(inputs):
            try:
                if is_dps:
                    out = allocate_outputs(definition, inp.resolved_axes, device)
                    with torch.no_grad():
                        sol_runnable(*inp, *out)
                    torch.cuda.synchronize(device)
                else:
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
                if tuple(sol_tensor.shape) != tuple(ref_tensor.shape):
                    return None, make_eval(
                        status=EvaluationStatus.INCORRECT_SHAPE,
                        hardware=hardware,
                        log_path=log_path,
                    )

                if sol_tensor.dtype != ref_tensor.dtype:
                    return None, make_eval(
                        status=EvaluationStatus.INCORRECT_DTYPE,
                        hardware=hardware,
                        log_path=log_path,
                    )

                non_finite_err_val: Optional[float] = None
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

                abs_err, rel_err, exceeds_tol, matched_ratio = compute_error_stats(
                    sol_tensor, ref_tensor, correctness
                )

                if exceeds_tol:
                    numerical_incorrect = True

                min_matched_ratio = min(min_matched_ratio, matched_ratio)
                max_abs = max(max_abs, abs_err)
                max_rel = max(max_rel, rel_err)

        correctness = Correctness(
            max_relative_error=max_rel,
            max_absolute_error=max_abs,
            extra={"matched_ratio": min_matched_ratio},
        )

        if numerical_incorrect:
            return correctness, make_eval(
                status=EvaluationStatus.INCORRECT_NUMERICAL,
                hardware=hardware,
                log_path=log_path,
                correctness=correctness,
            )

        return correctness, None
