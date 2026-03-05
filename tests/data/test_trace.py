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

import json
import math
import sys

import pytest

from sol_execbench.data import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    RandomInput,
    SafetensorsInput,
    Trace,
    Workload,
)


def test_correctness_valid_and_invalid():
    Correctness(max_relative_error=0.0, max_absolute_error=0.0)
    Correctness(max_relative_error=1e-5, max_absolute_error=1e-8)
    # Negative values are rejected
    with pytest.raises(ValueError):
        Correctness(max_relative_error=-1.0)
    with pytest.raises(ValueError):
        Correctness(max_absolute_error=-0.1)


def test_correctness_with_inf_and_nan():
    c = Correctness(max_relative_error=float("inf"), max_absolute_error=float("nan"))
    assert math.isinf(c.max_relative_error)
    assert math.isnan(c.max_absolute_error)

    # Serialises inf/nan as strings per JSON spec
    payload = json.loads(c.model_dump_json())
    assert payload["max_relative_error"] == "Infinity"
    assert payload["max_absolute_error"] == "NaN"

    # Parses back from string representations
    parsed = Correctness.model_validate(
        {"max_relative_error": "infinity", "max_absolute_error": "nan"}
    )
    assert math.isinf(parsed.max_relative_error)
    assert math.isnan(parsed.max_absolute_error)


def test_correctness_extra_field():
    c = Correctness(extra={"cosine_similarity": 0.99})
    assert c.extra["cosine_similarity"] == pytest.approx(0.99)


def test_performance_valid_and_invalid():
    Performance(latency_ms=1.5, reference_latency_ms=2.0, speedup_factor=1.33)
    Performance()  # all default to 0.0
    # Negative latency is rejected
    with pytest.raises(ValueError):
        Performance(latency_ms=-1.0)
    with pytest.raises(ValueError):
        Performance(reference_latency_ms=-0.1)
    with pytest.raises(ValueError):
        Performance(speedup_factor=-1.0)


def test_environment_valid_and_invalid():
    env = Environment(hardware="B200")
    assert env.hardware == "B200"
    assert env.libs == {}

    env2 = Environment(hardware="NVIDIA_H100", libs={"torch": "2.4.0", "cuda": "12.4"})
    assert env2.libs["torch"] == "2.4.0"

    # Empty hardware string is rejected
    with pytest.raises(ValueError):
        Environment(hardware="")


def test_evaluation_passed_requires_correctness_and_performance():
    env = Environment(hardware="cuda")
    # Missing both
    with pytest.raises(ValueError):
        Evaluation(
            status=EvaluationStatus.PASSED, log="", environment=env, timestamp="t"
        )
    # Missing performance only
    with pytest.raises(ValueError):
        Evaluation(
            status=EvaluationStatus.PASSED,
            log="",
            environment=env,
            timestamp="t",
            correctness=Correctness(),
        )
    # Missing correctness only
    with pytest.raises(ValueError):
        Evaluation(
            status=EvaluationStatus.PASSED,
            log="",
            environment=env,
            timestamp="t",
            performance=Performance(),
        )
    # Valid: both present
    Evaluation(
        status=EvaluationStatus.PASSED,
        log="",
        environment=env,
        timestamp="t",
        correctness=Correctness(),
        performance=Performance(),
    )


def test_evaluation_incorrect_numerical_requires_correctness_only():
    env = Environment(hardware="cuda")
    # Valid: correctness present, no performance
    Evaluation(
        status=EvaluationStatus.INCORRECT_NUMERICAL,
        log="",
        environment=env,
        timestamp="t",
        correctness=Correctness(),
    )
    # Performance must not be present
    with pytest.raises(ValueError):
        Evaluation(
            status=EvaluationStatus.INCORRECT_NUMERICAL,
            log="",
            environment=env,
            timestamp="t",
            correctness=Correctness(),
            performance=Performance(),
        )
    # Correctness must be present
    with pytest.raises(ValueError):
        Evaluation(
            status=EvaluationStatus.INCORRECT_NUMERICAL,
            log="",
            environment=env,
            timestamp="t",
        )


def test_evaluation_error_statuses_forbid_metrics():
    env = Environment(hardware="cuda")
    error_statuses = [
        EvaluationStatus.INCORRECT_SHAPE,
        EvaluationStatus.INCORRECT_DTYPE,
        EvaluationStatus.RUNTIME_ERROR,
        EvaluationStatus.COMPILE_ERROR,
        EvaluationStatus.TIMEOUT,
    ]
    for status in error_statuses:
        # Valid: no metrics
        Evaluation(status=status, log="", environment=env, timestamp="t")
        # Correctness must not be present
        with pytest.raises(ValueError):
            Evaluation(
                status=status,
                log="",
                environment=env,
                timestamp="t",
                correctness=Correctness(),
            )
        # Performance must not be present
        with pytest.raises(ValueError):
            Evaluation(
                status=status,
                log="",
                environment=env,
                timestamp="t",
                performance=Performance(),
            )


def test_trace_workload_only():
    workload = Workload(axes={"M": 8}, inputs={"A": RandomInput()}, uuid="w1")
    t = Trace(definition="def1", workload=workload)
    assert t.is_workload_trace() is True
    assert t.is_successful() is False
    assert t.solution is None
    assert t.evaluation is None


def test_trace_successful():
    workload = Workload(
        axes={"M": 8},
        inputs={
            "A": RandomInput(),
            "B": SafetensorsInput(path="p.safetensors", tensor_key="k"),
        },
        uuid="w2",
    )
    evaluation = Evaluation(
        status=EvaluationStatus.PASSED,
        log="ok",
        environment=Environment(hardware="B200"),
        timestamp="2025-01-01T00:00:00",
        correctness=Correctness(max_relative_error=1e-6, max_absolute_error=1e-8),
        performance=Performance(
            latency_ms=1.0, reference_latency_ms=2.0, speedup_factor=2.0
        ),
    )
    t = Trace(
        definition="def1", workload=workload, solution="sol1", evaluation=evaluation
    )
    assert t.is_workload_trace() is False
    assert t.is_successful() is True
    assert t.solution == "sol1"


def test_trace_failed_is_not_successful():
    workload = Workload(axes={"M": 4}, inputs={}, uuid="w3")
    evaluation = Evaluation(
        status=EvaluationStatus.COMPILE_ERROR,
        log="build failed",
        environment=Environment(hardware="B200"),
        timestamp="t",
    )
    t = Trace(
        definition="def1",
        workload=workload,
        solution="sol_broken",
        evaluation=evaluation,
    )
    assert t.is_workload_trace() is False
    assert t.is_successful() is False


def test_trace_roundtrip_json():
    workload = Workload(axes={"N": 16}, inputs={"x": RandomInput()}, uuid="w_rt")
    evaluation = Evaluation(
        status=EvaluationStatus.PASSED,
        log="",
        environment=Environment(hardware="LOCAL"),
        timestamp="2025-06-01T12:00:00",
        correctness=Correctness(),
        performance=Performance(latency_ms=0.5),
    )
    t = Trace(
        definition="rmsnorm_h4096",
        workload=workload,
        solution="my_sol",
        evaluation=evaluation,
    )

    # Round-trip through JSON
    dumped = t.model_dump_json()
    restored = Trace.model_validate_json(dumped)

    assert restored.definition == t.definition
    assert restored.solution == t.solution
    assert restored.workload.uuid == t.workload.uuid
    assert restored.evaluation.status == t.evaluation.status


if __name__ == "__main__":
    pytest.main(sys.argv)
