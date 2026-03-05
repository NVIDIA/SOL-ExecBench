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

from .bench import Benchmark, BenchmarkConfig
from .data import (
    AxisConst,
    AxisVar,
    BuildSpec,
    Correctness,
    Definition,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    RandomInput,
    SafetensorsInput,
    Solution,
    SourceFile,
    CompileOptions,
    SupportedHardware,
    SupportedLanguages,
    TensorSpec,
    Trace,
    TraceSet,
    Workload,
    load_json_file,
    load_jsonl_file,
)

try:
    from ._version import __version__, __version_tuple__
except Exception:
    __version__ = "0.0.0.dev0"
    __version_tuple__ = (0, 0, 0, "dev0")

__all__ = [
    # Benchmark
    "Benchmark",
    "BenchmarkConfig",
    # Data schema
    "Solution",
    # Definition types
    "Definition",
    "AxisConst",
    "AxisVar",
    "AxisExpr",
    "TensorSpec",
    # Workload types
    "Workload",
    "InputSpec",
    # Solution types
    "SourceFile",
    "BuildSpec",
    "CompileOptions",
    "SupportedHardware",
    "SupportedLanguages",
    # Trace types
    "Trace",
    "TraceSet",
    "RandomInput",
    "SafetensorsInput",
    "Correctness",
    "Performance",
    "Environment",
    "Evaluation",
    "EvaluationStatus",
    # programmatic API
    "DeviceConfig",
    "load_json_file",
    "load_jsonl_file",
]
