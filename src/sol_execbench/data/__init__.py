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

"""Data layer with strongly-typed dataclasses for SolExecBench."""

from .definition import AxisConst, AxisSpec, AxisVar, Definition, TensorSpec
from .json_utils import (
    append_jsonl_file,
    load_json_file,
    load_jsonl_file,
    save_json_file,
    save_jsonl_file,
)
from .solution import (
    BuildSpec,
    CompileOptions,
    Solution,
    SourceFile,
    SupportedBindings,
    SupportedHardware,
    SupportedLanguages,
)
from .trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    Trace,
)
from .trace_set import TraceSet
from .workload import CustomInput, InputSpec, RandomInput, SafetensorsInput, ScalarInput, Workload

__all__ = [
    # Definition types
    "AxisConst",
    "AxisExpr",
    "AxisSpec",
    "AxisVar",
    "TensorSpec",
    "Definition",
    # Solution types
    "SourceFile",
    "BuildSpec",
    "CompileOptions",
    "SupportedBindings",
    "SupportedHardware",
    "SupportedLanguages",
    "Solution",
    # Workload types
    "CustomInput",
    "RandomInput",
    "ScalarInput",
    "SafetensorsInput",
    "InputSpec",
    "Workload",
    # Trace types
    "Correctness",
    "Performance",
    "Environment",
    "Evaluation",
    "EvaluationStatus",
    "Trace",
    # TraceSet
    "TraceSet",
    # JSON functions
    "save_json_file",
    "load_json_file",
    "save_jsonl_file",
    "load_jsonl_file",
    "append_jsonl_file",
]
