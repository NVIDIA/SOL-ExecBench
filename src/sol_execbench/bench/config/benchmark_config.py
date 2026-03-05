"""Configuration for benchmark execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .env import get_sol_execbench_cache_path


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs.

    All fields have default values to make configuration optional.
    """

    definitions: Optional[list[str]] = field(default=None)
    solutions: Optional[list[str]] = field(default=None)

    warmup_runs: int = field(default=10)
    iterations: int = field(default=50)
    num_trials: int = field(default=3)
    log_dir: str = field(
        default_factory=lambda: str(get_sol_execbench_cache_path() / "logs")
    )
    use_isolated_runner: bool = field(default=False)
    timeout_seconds: int = field(default=300)
    time_baseline: bool = field(default=True)
    lock_clocks: bool = field(default=True)
    stream_injection_multiplier: float = field(default=3.0)

    allow_failed_baseline: bool = field(default=False)

    def __post_init__(self):
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be >= 0")
        if self.iterations <= 0:
            raise ValueError("iterations must be > 0")
        if self.num_trials <= 0:
            raise ValueError("num_trials must be > 0")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if not isinstance(self.timeout_seconds, int):
            raise ValueError("timeout_seconds must be an int")
        if self.definitions is not None and not isinstance(self.definitions, list):
            raise ValueError("definitions must be a list or None")
        if self.solutions is not None and not isinstance(self.solutions, list):
            raise ValueError("solutions must be a list or None")
