"""Abstract base class and common types for benchmark runners."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..config import BenchmarkConfig
from ..compile import RunnableInputs
from ...data import Definition, Evaluation, Solution, Workload
from ...data.workload import CorrectnessSpec


class RunnerError(RuntimeError): ...


class RunnerFatalError(RunnerError): ...


class BaselineHandle(str):
    pass


@dataclass
class DeviceBaseline:
    handle: BaselineHandle
    definition: Definition
    device: str
    inputs: list[RunnableInputs]
    outputs: list[list[torch.Tensor]]
    mean_latency_ms: float
    correctness: CorrectnessSpec = field(default_factory=CorrectnessSpec)


class Runner(ABC):
    def __init__(self, logger: logging.Logger) -> None: ...

    @abstractmethod
    def run_workload(
        self,
        definition: Definition,
        workload: Workload,
        solutions: list[Solution],
        config: BenchmarkConfig,
        root: Path,
    ) -> dict[str, Evaluation]: ...

    @abstractmethod
    def close(self) -> None:
        """Release all resources and terminate worker processes."""
        ...
