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

"""Main benchmark execution engine for sol_execbench kernel solutions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .config import BenchmarkConfig
from ..data import SupportedHardware
from .runner import IsolatedRunner
from ..data import Definition, Evaluation, Solution, Trace, Workload
from .execution_device import get_device_config

logger = logging.getLogger(__name__)


class Benchmark:
    """Main benchmark execution engine for kernel solutions.

    Orchestrates the execution of solutions against workloads using IsolatedRunner
    with GPU clock locking. Clock locking is managed at the timing level in each
    subprocess worker, ensuring consistent benchmarking conditions.

    Parameters
    ----------
    hardware : SupportedHardware
        Hardware type to benchmark against.
    config : BenchmarkConfig, optional
        Benchmark configuration. If None, uses default configuration.
    log_dir : Path, optional
        Directory for log files. If None, uses config.log_dir.

    Examples
    --------
    >>> from sol_execbench import SupportedHardware, BenchmarkConfig, Benchmark
    >>>
    >>> # Create benchmark with clock locking
    >>> config = BenchmarkConfig()
    >>> benchmark = Benchmark(SupportedHardware.B200, config=config)
    >>>
    >>> # Run evaluation (clocks locked automatically during timing)
    >>> results = benchmark.run_workload(definition, workload, [solution])
    """

    def __init__(
        self,
        hardware: SupportedHardware,
        config: Optional[BenchmarkConfig] = None,
    ):
        """Initialize the benchmark engine.

        Parameters
        ----------
        hardware : SupportedHardware
            Hardware type to benchmark against.
        config : BenchmarkConfig, optional
            Benchmark configuration. If None, uses default configuration.
        log_dir : Path, optional
            Directory for log files. If None, uses config.log_dir.
        """
        self._config = config or BenchmarkConfig()
        self._device_spec = get_device_config(hardware)

        # Create runner (always use isolated runner with clock locking)
        self._runner = IsolatedRunner(self._config.log_dir, self._device_spec)
        logger.info(f"Initialized IsolatedRunner with log_dir: {self._config.log_dir}")

    def run_workload(
        self,
        definition: Definition,
        workload: Workload,
        solutions: list[Solution],
        trace_set_root: Optional[Path] = None,
    ) -> dict[str, Evaluation]:
        """Run all solutions against a single workload.

        Clock locking is handled by the IsolatedRunner if device_config was provided.

        Parameters
        ----------
        definition : Definition
            Problem definition with input/output specs.
        workload : Workload
            Workload configuration with specific tensor shapes.
        solutions : list[Solution]
            List of solutions to evaluate.
        trace_set_root : Path, optional
            Root directory of trace set (for reference loading).

        Returns
        -------
        dict[str, Evaluation]
            Dictionary mapping solution names to evaluation results.

        Examples
        --------
        >>> benchmark = Benchmark(SupportedHardware.B200)
        >>> results = benchmark.run_workload(definition, workload, [solution])
        >>> print(results[solution.name].status)  # EvaluationStatus.PASSED
        """
        return self._runner.run_workload(
            definition=definition,
            workload=workload,
            solutions=solutions,
            config=self._config,
            root=trace_set_root,
        )

    def run_all(
        self,
        definitions: list[Definition],
        workloads_per_def: dict[str, list[Workload]],
        solutions_per_def: dict[str, list[Solution]],
        trace_set_root: Optional[Path] = None,
        resume: bool = False,
        existing_traces: Optional[dict[tuple[str, str, str], Trace]] = None,
    ) -> list[Trace]:
        """Run all solutions for all definitions and workloads.

        This method orchestrates the complete evaluation, iterating through all
        definitions, workloads, and solutions. It supports resume mode to skip
        already-evaluated solution-workload pairs.

        Parameters
        ----------
        definitions : list[Definition]
            List of problem definitions to evaluate.
        workloads_per_def : dict[str, list[Workload]]
            Dictionary mapping definition names to their workloads.
        solutions_per_def : dict[str, list[Solution]]
            Dictionary mapping definition names to their solutions.
        trace_set_root : Path, optional
            Root directory of trace set (for reference loading).
        resume : bool, optional
            If True, skip solution-workload pairs that already have traces.
        existing_traces : dict[tuple[str, str, str], Trace], optional
            Dictionary mapping (definition_name, solution_name, workload_id) to existing traces.
            Required if resume=True.

        Returns
        -------
        list[Trace]
            List of all evaluation traces (existing + newly created).

        Examples
        --------
        >>> benchmark = Benchmark(SupportedHardware.B200)
        >>> traces = benchmark.run_all(
        ...     definitions=[definition],
        ...     workloads_per_def={definition.name: [workload]},
        ...     solutions_per_def={definition.name: [solution]},
        ... )
        """
        all_traces: list[Trace] = []

        # Add existing traces if resuming
        if resume and existing_traces is not None:
            all_traces.extend(existing_traces.values())
            logger.info(f"Resuming with {len(existing_traces)} existing traces")

        # Iterate through definitions
        for definition in definitions:
            def_name = definition.name
            workloads = workloads_per_def.get(def_name, [])
            solutions = solutions_per_def.get(def_name, [])

            if not workloads:
                logger.warning(f"No workloads found for definition: {def_name}")
                continue

            if not solutions:
                logger.warning(f"No solutions found for definition: {def_name}")
                continue

            logger.info(f"Evaluating definition: {def_name}")
            logger.info(f"  - {len(workloads)} workloads")
            logger.info(f"  - {len(solutions)} solutions")

            # Iterate through workloads
            for workload in workloads:
                # Check which solutions need evaluation
                solutions_to_run = solutions
                if resume and existing_traces is not None:
                    solutions_to_run = [
                        sol
                        for sol in solutions
                        if (def_name, sol.name, workload.uuid) not in existing_traces
                    ]

                    if len(solutions_to_run) < len(solutions):
                        skipped = len(solutions) - len(solutions_to_run)
                        logger.info(
                            f"  - Skipping {skipped} already-evaluated solutions for workload {workload.uuid}"
                        )

                if not solutions_to_run:
                    logger.info(
                        f"  - All solutions already evaluated for workload {workload.uuid}"
                    )
                    continue

                logger.info(
                    f"  - Running workload: {workload.uuid} ({len(solutions_to_run)} solutions)"
                )

                # Run workload (with or without clock locking)
                try:
                    results = self.run_workload(
                        definition=definition,
                        workload=workload,
                        solutions=solutions_to_run,
                        trace_set_root=trace_set_root,
                    )

                    # Create traces from results
                    for solution in solutions_to_run:
                        evaluation = results.get(solution.name)
                        if evaluation is not None:
                            trace = Trace(
                                definition=def_name,
                                solution=solution.name,
                                workload=workload,
                                evaluation=evaluation,
                            )
                            all_traces.append(trace)
                            logger.info(
                                f"    ✓ {solution.name}: {evaluation.status.value} "
                            )
                        else:
                            logger.warning(
                                f"    ✗ {solution.name}: No evaluation result returned"
                            )

                except Exception as e:
                    logger.error(
                        f"  ✗ Failed to run workload {workload.uuid}: {e}",
                        exc_info=True,
                    )
                    continue

        logger.info(f"Completed evaluation: {len(all_traces)} total traces")
        return all_traces

    @property
    def config(self) -> BenchmarkConfig:
        """Get the benchmark configuration."""
        return self._config


__all__ = ["Benchmark"]
