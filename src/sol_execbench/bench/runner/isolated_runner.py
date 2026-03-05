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

"""Isolated runner that spawns a new process for each solution."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import torch
from torch import multiprocessing as mp

from ..execution_device import ExecutionDevice, list_cuda_devices
from ..config import BenchmarkConfig, DeviceConfig
from ..evaluators import resolve_evaluator
from ..utils import make_eval
from ..compile import BuilderRegistry, Runnable, RunnableInputs
from ...data import Definition, Evaluation, EvaluationStatus, Solution, Workload
from ...utils import redirect_stdio_to_file

from .runner import (
    BaselineHandle,
    DeviceBaseline,
    Runner,
    RunnerError,
    RunnerFatalError,
)

logger = logging.getLogger(__name__)


class SubprocessWorker:
    """Each instance binds to a CUDA device; the baseline resides in the main process; each Solution starts an independent Worker process for strong isolation."""

    def __init__(
        self,
        device: str,
        device_config: DeviceConfig,
        log_dir: str = "/tmp/sol_execbench",
    ) -> None:
        """Per device subprocess worker

        Parameters
        ----------
        device : str
            Device string (e.g. "cuda:0").
        log_dir : str, optional
            Directory for log files, by default "/tmp/sol_execbench".
        device_config : DeviceConfig
            GPU specification for clock locking (required).
        """
        self._device = device
        self._log_dir = log_dir
        self._device_spec = device_config
        self._baselines: dict[BaselineHandle, DeviceBaseline] = {}

    def run_ref(
        self,
        definition: Definition,
        workload: Workload,
        cfg: BenchmarkConfig,
        trace_set_root: Optional[Path] = None,
    ) -> BaselineHandle:
        execution_device = (
            ExecutionDevice(device_config=self._device_spec, device_str=self._device)
            if self._device_spec is not None
            else None
        )
        hardware = self._device_spec.hardware_type
        evaluator_cls = resolve_evaluator(definition)
        baseline = evaluator_cls.build_baseline(
            definition=definition,
            workload=workload,
            cfg=cfg,
            device=self._device,
            trace_set_root=trace_set_root,
            execution_device=execution_device,
            hardware=hardware,
        )
        self._baselines[baseline.handle] = baseline
        return baseline.handle

    def run_solution(
        self, solution: Solution, baseline: BaselineHandle, cfg: BenchmarkConfig
    ) -> Evaluation:
        """Run solution in an isolated subprocess.

        Parameters
        ----------
        solution : Solution
            Solution to evaluate.
        baseline : BaselineHandle
            Handle to baseline for comparison.
        cfg : BenchmarkConfig
            Benchmark configuration.

        Returns
        -------
        Evaluation
            Evaluation results with status, correctness, and performance metrics.
        """
        if baseline not in self._baselines:
            raise RunnerError(f"Baseline handle not found: {baseline}")
        bl = self._baselines[baseline]

        log_path = os.path.join(self._log_dir, f"{solution.name}_{time.time()}.log")
        # New process for each solution run
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)

        proc = ctx.Process(
            target=_solution_worker_main,
            args=(
                child_conn,
                self._device,
                bl.definition,
                solution,
                cfg,
                log_path,
                self._device_spec,
            ),
            daemon=True,
        )
        proc.start()

        evaluation: Optional[Evaluation] = None
        start_time = time.time()
        try:
            if parent_conn.poll(timeout=30.0):  # 30 seconds for startup
                msg = parent_conn.recv()
                if msg.get("cmd") != "READY":
                    raise RunnerFatalError(f"Worker failed to start, got: {msg}")
                parent_conn.send({"ok": True})
            else:
                evaluation = make_eval(
                    status=EvaluationStatus.TIMEOUT,
                    hardware=self._device_spec.hardware_type,
                    log_path=log_path,
                    extra_msg="Worker failed to start within 30 seconds",
                )
                return evaluation

            while True:
                # Check if we've exceeded total timeout
                elapsed = time.time() - start_time
                remaining_timeout = max(1.0, cfg.timeout_seconds - elapsed)

                if elapsed >= cfg.timeout_seconds:
                    evaluation = make_eval(
                        status=EvaluationStatus.TIMEOUT,
                        hardware=self._device_spec.hardware_type,
                        log_path=log_path,
                        extra_msg=f"Evaluation timeout after {cfg.timeout_seconds} seconds for solution {solution.name}",
                    )
                    break

                # Wait for message with remaining timeout
                if parent_conn.poll(timeout=remaining_timeout):
                    msg = parent_conn.recv()
                    cmd = msg.get("cmd")
                else:
                    # Timeout
                    evaluation = make_eval(
                        status=EvaluationStatus.TIMEOUT,
                        hardware=self._device_spec.hardware_type,
                        log_path=log_path,
                        extra_msg=f"Evaluation timeout after {cfg.timeout_seconds} seconds for solution {solution.name}",
                    )
                    break

                if cmd == "LOAN":
                    # Zero-effect copy via IPC handle
                    parent_conn.send(
                        {
                            "ok": True,
                            "inputs": bl.inputs,
                            "ref_outputs": bl.outputs,
                            "ref_mean_latency_ms": bl.mean_latency_ms,
                            "correctness": bl.correctness,
                        }
                    )

                elif cmd == "EVAL":
                    evaluation = msg["evaluation"]
                    break

                elif cmd == "ERROR":
                    error_msg = msg.get("msg", "Unknown error")
                    evaluation = make_eval(
                        status=EvaluationStatus.RUNTIME_ERROR,
                        hardware=self._device_spec.hardware_type,
                        log_path=log_path,
                        extra_msg=error_msg,
                    )
                    break

                else:
                    logger.warning("Unknown worker command: %s", cmd)
                    continue

        except EOFError as e:
            logger.error("Worker crashed (EOF) running %s: %s", solution.name, e)
        except Exception:
            logger.error("Unknown error running %s", solution.name, exc_info=True)
        finally:
            try:
                parent_conn.close()
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.terminate()
            except Exception:
                pass

        if evaluation is None:
            evaluation = make_eval(
                status=EvaluationStatus.RUNTIME_ERROR,
                hardware=self._device_spec.hardware_type,
                log_path=log_path,
                extra_msg="Worker process failed unexpectedly",
            )

        return evaluation

    def release(self, baseline: BaselineHandle) -> None:
        self._baselines.pop(baseline, None)

    def close(self) -> None:
        self._baselines.clear()


def _solution_worker_main(
    conn: mp.connection.Connection,
    device: str,
    definition: Definition,
    solution: Solution,
    cfg: BenchmarkConfig,
    log_path: str,
    device_config: Optional[DeviceConfig] = None,
) -> None:
    """Worker process: strong isolation for single Solution.

    Borrow/return trial data via Pipe and send Evaluation back to parent process.

    Parameters
    ----------
    conn : mp.connection.Connection
        Multiprocessing connection for communication with parent process.
    device : str
        Device string (e.g. "cuda:0").
    definition : Definition
        Operation definition.
    solution : Solution
        Solution to evaluate.
    cfg : BenchmarkConfig
        Benchmark configuration.
    log_path : str
        Path to log file.
    device_config : DeviceConfig, optional
        GPU specification for clock locking during benchmarking.
    """
    _ = redirect_stdio_to_file(log_path)

    # Create execution device for clock locking (required)
    if device_config is None:
        conn.send({"cmd": "ERROR", "msg": "device_config is required for benchmarking"})
        return

    try:
        execution_device = ExecutionDevice(
            device_config=device_config, device_str=device
        )
        if cfg.lock_clocks:
            logger.info(
                f"Clock locking enabled for benchmarking: {execution_device.default_clock_mhz} MHz"
            )
        else:
            logger.info("Clock locking disabled for benchmarking")
    except Exception as e:
        logger.error(f"Failed to create execution device: {e}")
        conn.send({"cmd": "ERROR", "msg": f"Failed to create execution device: {e}"})
        return

    try:
        torch.cuda.set_device(int(device.split(":")[1]))
        registry = BuilderRegistry.get_instance()

        # Handshake
        conn.send({"cmd": "READY"})
        init = conn.recv()
        if not init.get("ok", False):
            conn.send({"cmd": "ERROR", "msg": "Init not ok"})
            return

        # Build solution (no clock locking - just compilation)
        try:
            runnable_sol: Runnable = registry.build(definition, solution)
        except Exception as e:
            import traceback

            print(
                f"Build error: {type(e).__name__}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            )
            ev = make_eval(
                status=EvaluationStatus.COMPILE_ERROR,
                hardware=device_config.hardware_type,
                log_path=log_path,
            )
            conn.send({"cmd": "EVAL", "evaluation": ev})
            return

        # Request baseline data
        conn.send({"cmd": "LOAN"})
        loan = conn.recv()

        inputs_bl: RunnableInputs = loan["inputs"]
        ref_outputs_bl: list[list[torch.Tensor]] = loan["ref_outputs"]
        ref_mean_latency_ms = loan["ref_mean_latency_ms"]
        correctness = loan["correctness"]

        inputs = [inp.clone_tensors() for inp in inputs_bl]

        # Evaluate solution (clock locking happens inside timing functions)
        evaluator_cls = resolve_evaluator(definition)
        evaluation = evaluator_cls.evaluate(
            definition=definition,
            sol_runnable=runnable_sol,
            inputs=inputs,
            ref_outputs=ref_outputs_bl,
            ref_mean_latency_ms=ref_mean_latency_ms,
            cfg=cfg,
            correctness=correctness,
            log_path=log_path,
            device=device,
            execution_device=execution_device,
            hardware=device_config.hardware_type,
        )

        conn.send({"cmd": "EVAL", "evaluation": evaluation})

    except Exception as e:
        try:
            conn.send({"cmd": "ERROR", "msg": str(e)})
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


class IsolatedRunner(Runner):
    def __init__(self, log_dir: str, device_config: DeviceConfig) -> None:
        """Initialize the isolated runner with per device workers.

        Parameters
        ----------
        log_dir : str
            Directory for log files.
        device_config : DeviceConfig
            GPU specification for clock locking (required).
        """
        # Track retry attempts for each device
        self._device_retry_counts: dict[str, int] = {}
        self._worker_max_retries = 3
        self._device_spec = device_config

        # Initialize workers for all available CUDA devices
        self._available_devices = list_cuda_devices()
        self._workers = [
            SubprocessWorker(d, device_config, log_dir) for d in self._available_devices
        ]
        self._curr_worker_idx = 0
        self._log_dir = log_dir

        if len(self._workers) == 0:
            raise RuntimeError("No CUDA devices available")

        logger.info(
            f"Initialized benchmark multi-process on {len(self._available_devices)} CUDA devices "
            f"and {len(self._workers)} workers"
        )
        logger.info(
            f"Clock locking enabled: {device_config.locked_clock_speed} MHz ({device_config.sm_version})"
        )

    def _pick_workers(self, K: int) -> list[SubprocessWorker]:
        """Pick K workers in round-robin fashion.

        Parameters
        ----------
        K : int
            Number of workers to pick.

        Returns
        -------
        list[SubprocessWorker]
            List of selected workers.
        """
        if K <= 0 or not self._workers:
            return []
        D = len(self._workers)
        start = self._curr_worker_idx
        sel = [self._workers[(start + i) % D] for i in range(min(K, D))]
        self._curr_worker_idx = (start + K) % D
        return sel

    def _relaunch_worker(self, device: str) -> SubprocessWorker:
        """Relaunch a worker for the given device.

        Parameters
        ----------
        device : str
            Device string (e.g. "cuda:0").

        Returns
        -------
        SubprocessWorker
            New worker instance for the device.
        """
        logger.info(f"Relaunching worker for device {device}")
        return SubprocessWorker(device, self._log_dir)

    def _handle_failed_workers(self, failed_workers: list[SubprocessWorker]) -> None:
        """Handle failed workers by attempting to relaunch them or removing them.

        Parameters
        ----------
        failed_workers : List[SubprocessWorker]
            List of workers that have failed.
        """
        workers_to_remove = []
        workers_to_add = []

        for failed_worker in failed_workers:
            device = failed_worker._device
            retry_count = self._device_retry_counts.get(device, 0)

            if retry_count < self._worker_max_retries:
                self._device_retry_counts[device] = retry_count + 1
                try:
                    new_worker = self._relaunch_worker(device)
                    workers_to_add.append(new_worker)
                    logger.info(f"Successfully relaunched worker for device {device} ")
                except Exception:
                    logger.error(f"Failed to relaunch worker for device {device} ")
                    if retry_count + 1 >= self._worker_max_retries:
                        workers_to_remove.append(failed_worker)
                        logger.warning(
                            f"Removing device {device} after {self._worker_max_retries} failed attempts"
                        )
            else:
                workers_to_remove.append(failed_worker)
                logger.warning(
                    f"Removing device {device} after {self._worker_max_retries} failed attempts"
                )
        if workers_to_remove:
            self._workers = [r for r in self._workers if r not in workers_to_remove]

        self._workers.extend(workers_to_add)

        if self._workers:
            self._curr_worker_idx %= len(self._workers)

    def _has_healthy_workers(self) -> bool:
        """Check if there are any healthy workers available.

        Returns
        -------
        bool
            True if there are healthy workers, False otherwise.
        """
        return bool(self._workers)

    def run_workload(
        self,
        definition: Definition,
        workload: Workload,
        solutions: list[Solution],
        config: BenchmarkConfig,
        root: Path,
    ) -> dict[str, Evaluation]:
        """Run a workload with the given solutions and return evaluation results.

        Parameters
        ----------
        definition : Definition
            Operation definition.
        workload : Workload
            Workload specification.
        solutions : List[Solution]
            List of solutions to evaluate.
        config : BenchmarkConfig
            Benchmark configuration.
        root : Path
            Root path for the trace set.

        Returns
        -------
        dict[str, Evaluation]
            Dictionary mapping solution names to their evaluations.
        """
        if not solutions:
            return {}

        K = min(len(self._workers), len(solutions))
        selected = self._pick_workers(K)
        if not selected:
            raise RuntimeError("No healthy workers available")

        # Build baselines on each worker
        baselines: dict[SubprocessWorker, BaselineHandle] = {}
        failed_workers: list[SubprocessWorker] = []
        last_builder_error: Optional[str] = None

        with ThreadPoolExecutor(max_workers=K) as pool:
            baseline_futs = {
                pool.submit(r.run_ref, definition, workload, config, root): r
                for r in selected
            }
            for fut, r in baseline_futs.items():
                try:
                    h = fut.result()
                    baselines[r] = h
                except Exception as e:
                    last_builder_error = str(e)
                    failed_workers.append(r)
                    logger.error(
                        f"Runner {r._device} failed while running reference for "
                        f"def={definition.name} workload={workload.uuid}: {e}"
                    )

        if len(failed_workers) == len(selected) and config.allow_failed_baseline:
            # fail all solutions due to a failed baseline
            return {
                solution.name: make_eval(
                    status=EvaluationStatus.INVALID_REFERENCE,
                    hardware=self._device_spec.hardware_type,
                    log_path=None,
                    extra_msg=last_builder_error,
                )
                for solution in solutions
            }

        # Handle failed workers
        if failed_workers:
            self._handle_failed_workers(failed_workers)
            if not self._has_healthy_workers():
                raise RuntimeError("No healthy workers available")

        # Filter out workers that failed to build baselines
        selected = [r for r in selected if r in baselines]
        if not selected:
            raise RuntimeError("No healthy workers available after baseline setup")

        try:
            # Evaluate solutions round-robin across workers
            with ThreadPoolExecutor(max_workers=len(selected)) as pool:
                sol_futs: dict[str, Any] = {}
                for i, solution in enumerate(solutions):
                    r = selected[i % len(selected)]
                    sol_futs[solution.name] = pool.submit(
                        r.run_solution, solution, baselines[r], config
                    )

                results: dict[str, Evaluation] = {
                    name: fut.result() for name, fut in sol_futs.items()
                }
        finally:
            # Always release baselines, even if solution execution fails
            for r in selected:
                if r in baselines:
                    r.release(baselines[r])
            torch.cuda.empty_cache()

        return results

    def close(self) -> None:
        """Release all resources and terminate worker processes."""
        for worker in self._workers:
            worker.close()
        self._workers.clear()
