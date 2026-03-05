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

"""Abstract base class for kernel evaluators."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import torch

from ..config import BenchmarkConfig
from ..runner.runner import DeviceBaseline
from ..utils import make_eval
from ..compile import Runnable, RunnableInputs
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
from .reward_hack import (
    check_lazy_outputs,
    check_thread_injection,
)
from .utils import allocate_outputs, normalize_result


class Evaluator(ABC):
    @classmethod
    @abstractmethod
    def can_evaluate(cls, definition: Definition) -> bool: ...

    @classmethod
    def run_solution(
        cls,
        definition: Definition,
        sol_runnable: Runnable,
        inp: RunnableInputs,
        device: str,
    ) -> list[torch.Tensor]:
        """Run the solution function with reward-hack checks.

        Handles both DPS and value-returning calling conventions,
        and checks for thread injection and lazy outputs.

        Raises:
            RewardHackDetected: If a reward hack is detected.
            Exception: If the solution function fails.
        """
        is_dps = sol_runnable.metadata.destination_passing_style
        threads_before = threading.active_count()

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

        check_thread_injection(threads_before, threading.active_count())
        check_lazy_outputs(out)

        return out

    @classmethod
    @abstractmethod
    def build_baseline(
        cls,
        definition: Definition,
        workload: Workload,
        cfg: BenchmarkConfig,
        device: str,
        trace_set_root: Optional[Path] = None,
        execution_device: Any = None,
        hardware: SupportedHardware = SupportedHardware.LOCAL,
    ) -> DeviceBaseline: ...

    @classmethod
    @abstractmethod
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
    ) -> tuple[Optional[Correctness], Optional[Evaluation]]: ...

    @classmethod
    @abstractmethod
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
    ) -> tuple[Performance, Optional[Evaluation]]: ...

    @classmethod
    def evaluate(
        cls,
        definition: Definition,
        sol_runnable: Runnable,
        inputs: list[RunnableInputs],
        ref_outputs: list[list[torch.Tensor]],
        ref_mean_latency_ms: float,
        cfg: BenchmarkConfig,
        correctness: CorrectnessSpec,
        log_path: str,
        device: str,
        execution_device: Any,
        hardware: SupportedHardware = SupportedHardware.LOCAL,
    ) -> Evaluation:
        correctness, evaluation = cls.check_correctness(
            definition=definition,
            sol_runnable=sol_runnable,
            inputs=inputs,
            ref_outputs=ref_outputs,
            cfg=cfg,
            correctness=correctness,
            log_path=log_path,
            device=device,
            hardware=hardware,
        )
        if evaluation is not None:
            return evaluation

        performance, evaluation = cls.eval_performance(
            definition=definition,
            sol_runnable=sol_runnable,
            inputs=inputs,
            ref_mean_latency_ms=ref_mean_latency_ms,
            cfg=cfg,
            log_path=log_path,
            device=device,
            execution_device=execution_device,
            hardware=hardware,
        )

        if evaluation is not None:
            return evaluation

        return make_eval(
            status=EvaluationStatus.PASSED,
            hardware=hardware,
            log_path=log_path,
            correctness=correctness,
            performance=performance,
        )
