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

"""Evaluator registry."""

from __future__ import annotations

from typing import Type

from ...data import Definition

from .default import DefaultEvaluator
from .evaluator import Evaluator
from .lowbit import LowBitEvaluator
from .sampling import SamplingEvaluator

EvaluatorType = Type[Evaluator]

_CUSTOM_EVALUATORS: list[EvaluatorType] = [SamplingEvaluator, LowBitEvaluator]
_DEFAULT_EVALUATOR: EvaluatorType = DefaultEvaluator


def resolve_evaluator(definition: Definition) -> EvaluatorType:
    matches = [cls for cls in _CUSTOM_EVALUATORS if cls.can_evaluate(definition)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        return _DEFAULT_EVALUATOR
    raise ValueError(f"Multiple evaluator matches for definition '{definition.name}'")
