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
