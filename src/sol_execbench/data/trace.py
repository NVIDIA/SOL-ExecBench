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

"""Strong-typed data definitions for traces and evaluations."""

import math
from enum import Enum
from typing import Any, Optional

from pydantic import ConfigDict, Field, field_validator, model_validator

from .base_model import BaseModelWithDocstrings, NonEmptyString
from .workload import Workload


class Correctness(BaseModelWithDocstrings):
    """Correctness metrics from numerical evaluation.

    Contains error measurements comparing the solution output against
    a reference implementation to assess numerical accuracy.
    """

    model_config = ConfigDict(ser_json_inf_nan="strings")

    max_relative_error: float = Field(default=0.0)
    """Maximum relative error observed across all output elements."""
    max_absolute_error: float = Field(default=0.0)
    """Maximum absolute error observed across all output elements."""
    extra: Optional[dict[str, Any]] = Field(default=None)
    """Extra metrics for correctness evaluation."""

    @field_validator("max_relative_error", "max_absolute_error")
    @classmethod
    def non_negative_or_nan(cls, v: float):
        if math.isnan(v):
            return v
        if v < 0:
            raise ValueError("must be non-negative or NaN")
        return v


class Performance(BaseModelWithDocstrings):
    """Performance metrics from timing evaluation.

    Contains timing measurements and performance comparisons from
    benchmarking the solution against reference implementations.
    """

    latency_ms: float = Field(default=0.0, ge=0.0)
    """Solution execution latency in milliseconds."""
    reference_latency_ms: float = Field(default=0.0, ge=0.0)
    """Reference implementation latency in milliseconds for comparison."""
    speedup_factor: float = Field(default=0.0, ge=0.0)
    """Performance speedup factor compared to reference (reference_time / solution_time)."""


class Environment(BaseModelWithDocstrings):
    """Environment information from evaluation execution.

    Records the hardware and software environment details from when
    the evaluation was performed, enabling reproducibility analysis.
    """

    hardware: NonEmptyString
    """Hardware identifier where the evaluation was performed (e.g., 'NVIDIA_H100')."""
    libs: dict[str, str] = Field(default_factory=dict)
    """Dictionary of library names to version strings used during evaluation."""


class EvaluationStatus(str, Enum):
    """Status codes for evaluation results.

    Enumeration of all possible outcomes when evaluating a solution
    against a workload, covering success and various failure modes.
    """

    PASSED = "PASSED"
    """Evaluation completed successfully with correct results."""
    INVALID_REFERENCE = "INVALID_REFERENCE"
    """Definition reference code failed to run."""
    INCORRECT_SHAPE = "INCORRECT_SHAPE"
    """Solution produced output with incorrect tensor shape."""
    INCORRECT_NUMERICAL = "INCORRECT_NUMERICAL"
    """Solution produced numerically incorrect results."""
    INCORRECT_DTYPE = "INCORRECT_DTYPE"
    """Solution produced output with incorrect data type."""
    RUNTIME_ERROR = "RUNTIME_ERROR"
    """Solution encountered a runtime error during execution."""
    COMPILE_ERROR = "COMPILE_ERROR"
    """Solution failed to compile or build successfully."""
    TIMEOUT = "TIMEOUT"
    """Evaluation did not complete within the configured timeout."""


class Evaluation(BaseModelWithDocstrings):
    """Complete evaluation result for a solution on a workload.

    Records the full outcome of benchmarking a solution implementation
    against a specific workload, including status, metrics, and environment.
    """

    status: EvaluationStatus
    """The overall evaluation status indicating success or failure mode."""
    environment: Environment
    """Environment details where the evaluation was performed."""
    timestamp: NonEmptyString
    """Timestamp when the evaluation was performed (ISO format recommended)."""
    log: str = ""
    """Captured stdout/stderr from the evaluation run."""
    correctness: Optional[Correctness] = None
    """Correctness metrics (present for PASSED and INCORRECT_NUMERICAL status)."""
    performance: Optional[Performance] = None
    """Performance metrics (present only for PASSED status)."""

    @model_validator(mode="after")
    def _validate_status_correctness_performance(self) -> "Evaluation":
        """Validate correctness and performance fields based on status.

        Ensures that correctness and performance metrics are present or absent
        based on the evaluation status, following the schema requirements.

        Raises
        ------
        ValueError
            If correctness/performance presence doesn't match status requirements.
        """
        if self.status == EvaluationStatus.PASSED:
            if self.correctness is None:
                raise ValueError(
                    f"Evaluation must include correctness when status is {self.status}"
                )
            if self.performance is None:
                raise ValueError(
                    f"Evaluation must include performance when status is {self.status}"
                )
        elif self.status == EvaluationStatus.INCORRECT_NUMERICAL:
            if self.correctness is None:
                raise ValueError(
                    f"Evaluation must include correctness when status is {self.status}"
                )
            if self.performance is not None:
                raise ValueError(
                    f"Evaluation must not include performance when status is {self.status}"
                )
        else:
            # For other error statuses, neither correctness nor performance should be present
            if self.correctness is not None:
                raise ValueError(
                    f"Evaluation must not include correctness when status is {self.status}"
                )
            if self.performance is not None:
                raise ValueError(
                    f"Evaluation must not include performance when status is {self.status}"
                )
        return self


class Trace(BaseModelWithDocstrings):
    """Complete trace linking a solution to a definition with evaluation results.

    A Trace represents the complete record of benchmarking a specific solution
    implementation against a specific computational workload definition. It includes
    the workload configuration and evaluation results.

    Special case: A "workload trace" contains only definition and workload fields
    (with solution and evaluation set to None), representing a workload configuration
    without an actual benchmark execution.
    """

    definition: NonEmptyString
    """Name of the Definition that specifies the computational workload."""
    workload: Workload
    """Concrete workload configuration with specific axis values and inputs."""
    solution: Optional[str] = None
    """Name of the Solution implementation (None for workload-only traces)."""
    evaluation: Optional[Evaluation] = None
    """Evaluation results from benchmarking (None for workload-only traces)."""

    def is_workload_trace(self) -> bool:
        """Check if this is a workload-only trace.

        Returns
        -------
        bool
            True if this is a workload trace without solution/evaluation data.
        """
        return self.solution is None and self.evaluation is None

    def is_successful(self) -> bool:
        """Check if the benchmark execution was successful.

        Returns
        -------
        bool
            True if this is a regular trace with successful evaluation status.
            False for workload traces or failed evaluations.
        """
        return (
            not self.is_workload_trace()
        ) and self.evaluation.status == EvaluationStatus.PASSED
