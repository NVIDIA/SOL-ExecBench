#!/usr/bin/env python3

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

"""SOL ExecBench evaluation driver.

Self-contained script written to the GPU staging directory.
Evaluates a user solution and outputs JSONL Trace objects to stdout.
All non-JSON output (library messages, Triton JIT logs) goes to stderr.
"""

from __future__ import annotations

import sys

if __name__ == "__main__" and not (
    sys.flags.ignore_environment and sys.flags.safe_path
):
    from os import execv

    execv(sys.executable, [sys.executable, "-E", "-P", *sys.orig_argv[1:]])

import atexit
import ctypes
import gc
import importlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import tempfile
import threading
import traceback
from contextlib import ExitStack, contextmanager
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from pathlib import Path
from tempfile import TemporaryFile
from typing import Optional

STAGING_DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    # ── Redirect stdout → stderr BEFORE importing torch/triton ──────────────────
    # Saves the original stdout fd so we can print JSON to it later.
    _real_stdout_fd = os.dup(1)
    _real_stdout = os.fdopen(_real_stdout_fd, "w", buffering=1)
    os.dup2(2, 1)  # fd 1 now points at stderr
    sys.stdout = open(1, "w", buffering=1, closefd=False)

    sys.path[:] = [
        path
        for path in [*os.environ.get("PYTHONPATH", "").split(os.pathsep), *sys.path]
        if path and not Path(path).resolve().is_relative_to(STAGING_DIR)
    ]

import torch  # noqa: E402 — must come after redirect
from torch.utils.serialization import config as serialization_config  # noqa: E402


# ── DPS helper ───────────────────────────────────────────────────────────────
def _call_and_collect_outputs(
    fn,
    inputs,
    dps,
    definition,
    resolved_axes,
    device,
    output_names,
    output_dtypes_torch,
):
    """Call fn in the correct calling convention and return a list of output tensors.

    DPS=True (destination-passing style):
        fn(*inputs, *pre_allocated_outputs)  → outputs written in-place
    DPS=False (return-value style):
        result = fn(*inputs)  → normalize_outputs(result) → list of tensors
    """
    if dps:
        outputs = allocate_outputs(definition, resolved_axes, device)
        fn(*inputs, *outputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        return outputs
    else:
        result = fn(*inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        out_dict = normalize_outputs(
            result,
            device=torch.device(device),
            output_names=output_names,
            output_dtypes=output_dtypes_torch,
        )
        return [out_dict[n] for n in output_names]


# ── Load problem ─────────────────────────────────────────────────────────────
def _load_problem() -> "tuple[dict, list[dict]]":
    """Load definition dict + workload dicts from staging directory."""
    local_def = STAGING_DIR / "definition.json"
    local_wkl = STAGING_DIR / "workload.jsonl"

    if not local_def.exists():
        raise RuntimeError(
            "definition.json not found in staging directory — "
            "client must supply definition and workloads inline"
        )

    definition = json.loads(local_def.read_text())
    workloads: list[dict] = []
    if local_wkl.exists():
        for line in local_wkl.read_text().splitlines():
            line = line.strip()
            if line:
                workloads.append(json.loads(line))
    return definition, workloads


# ── Imports from sol_execbench.core ─────────────────────────────────────────────
from sol_execbench.core.bench.clock_lock import are_clocks_locked  # noqa: E402
from sol_execbench.core.bench.config import BenchmarkConfig  # noqa: E402
from sol_execbench.core.bench.correctness import (  # noqa: E402
    compute_error_stats,
    set_seed,
)
from sol_execbench.core.bench.io import (  # noqa: E402
    allocate_outputs,
    gen_inputs,
    load_safetensors,
    normalize_outputs,
)
from sol_execbench.core.bench.reward_hack import (  # noqa: E402
    RewardHackDetected,
    check_eval_integrity,
    check_lazy_outputs,
    check_monkey_patch,
    check_thread_injection,
    snapshot_critical_functions,
)
from sol_execbench.core.bench.timing import time_runnable  # noqa: E402
from sol_execbench.core.bench.utils import (  # noqa: E402
    make_eval,
)
from sol_execbench.core import (  # noqa: E402
    Correctness,
    Definition,
    EvaluationStatus,
    Performance,
    Solution,
    SupportedLanguages,
    Trace,
    Workload,
)
from sol_execbench.core.data.dtypes import dtype_str_to_torch_dtype  # noqa: E402


def _send(channel, message):
    channel.send_bytes(json.dumps(message, allow_nan=False).encode())


def _receive(channel):
    return json.loads(channel.recv_bytes(maxlength=1 << 20))


def _chunks(tensor):
    data = memoryview(tensor.reshape(-1).view(torch.uint8).numpy())
    size = 4 << 20
    for offset in range(0, len(data), size):
        yield data[offset : offset + size]


def _send_outputs(channel, outputs):
    _send(
        channel, ["ok", [(list(tensor.shape), str(tensor.dtype)) for tensor in outputs]]
    )
    if _receive(channel):
        for tensor in outputs:
            for chunk in _chunks(tensor.resolve_conj().resolve_neg().contiguous()):
                channel.send_bytes(chunk)


class WorkerError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class SolutionWorker:
    def __init__(self, channel, archive):
        self.channel = channel
        self.archive = archive

    def reply(self):
        kind, value = _receive(self.channel)
        if kind == "hack":
            raise WorkerError(EvaluationStatus.REWARD_HACK, value)
        if kind == "error":
            raise WorkerError(EvaluationStatus.RUNTIME_ERROR, value)
        if kind != "ok":
            raise ValueError("Invalid solution worker reply")
        return value

    def set_inputs(self, inputs):
        if sys.platform == "linux":
            with serialization_config.patch({"save.compute_crc32": False}):
                torch.save(inputs, f"/proc/self/fd/{self.archive.fileno()}")
        else:
            self.archive.seek(0)
            self.archive.truncate()
            torch.save(inputs, self.archive)

    def run(self, axes, first, expected):
        _send(self.channel, ["run", axes, first])
        metadata = self.reply()
        shapes = [list(tensor.shape) for tensor in expected]
        dtypes = [str(tensor.dtype) for tensor in expected]
        if [shape for shape, dtype in metadata] != shapes:
            _send(self.channel, False)
            raise WorkerError(EvaluationStatus.INCORRECT_SHAPE, "Output shapes differ")
        if [dtype for shape, dtype in metadata] != dtypes:
            _send(self.channel, False)
            raise WorkerError(EvaluationStatus.INCORRECT_DTYPE, "Output dtypes differ")
        _send(self.channel, True)
        outputs = []
        for tensor in expected:
            host = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
            for chunk in _chunks(host):
                chunk[:] = self.channel.recv_bytes(maxlength=len(chunk))
            outputs.append(host.to(tensor.device))
        return outputs

    def time(self, axes):
        _send(self.channel, ["time", axes, False])
        latency = self.reply()
        if (
            type(latency) not in (int, float)
            or not math.isfinite(latency)
            or latency <= 0
        ):
            raise ValueError("Invalid solution latency")
        return latency


@contextmanager
def solution_worker(staging):
    with ExitStack() as stack:
        if sys.platform == "linux":
            archive = stack.enter_context(
                os.fdopen(os.memfd_create("sol-inputs"), "w+b")
            )
            reader = stack.enter_context(
                open(f"/proc/self/fd/{archive.fileno()}", "rb")
            )
        else:
            archive = reader = stack.enter_context(TemporaryFile())
        local, remote = Pipe()
        stack.enter_context(local)
        with remote:
            args = [
                sys.executable,
                "-E",
                "-P",
                __file__,
                "--worker",
                str(staging),
                str(remote.fileno()),
                str(reader.fileno()),
                str(os.getpid()),
            ]
            proc = subprocess.Popen(
                args,
                pass_fds=(remote.fileno(), reader.fileno()),
                env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
                stdin=subprocess.DEVNULL,
                stdout=sys.stderr,
            )
        try:
            yield SolutionWorker(local, archive)
        finally:
            local.close()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


# ── Resolve user function ─────────────────────────────────────────────────────
def _load_solution(staging, solution):
    sys.path.insert(0, str(staging))
    _entry_point = solution.spec.entry_point
    if "::" in _entry_point:
        _entry_module_or_file, _entry_func_name = _entry_point.rsplit("::", 1)
    else:
        _entry_module_or_file, _entry_func_name = _entry_point, "run"

    _CPP_LANGUAGES = {
        SupportedLanguages.CUDA_CPP,
        SupportedLanguages.CUTLASS,
        SupportedLanguages.CUDNN,
        SupportedLanguages.CUBLAS,
    }
    if any(lang in _CPP_LANGUAGES for lang in solution.spec.languages):
        _so_path = staging / "benchmark_kernel.so"
        if not _so_path.exists():
            raise RuntimeError(f"benchmark_kernel.so not found at {_so_path}")
        _spec_obj = importlib.util.spec_from_file_location("benchmark_kernel", _so_path)
        _user_mod = importlib.util.module_from_spec(_spec_obj)
        _spec_obj.loader.exec_module(_user_mod)
        user_fn = getattr(_user_mod, _entry_func_name)
    else:
        # Block torch.utils.cpp_extension.load/load_inline on the GPU server.
        # CUDA/C++ compilation must go through Phase 1 on the compile server
        # (use a C++ language like "cuda_cpp" in the solution spec).
        import torch.utils.cpp_extension as _cpp_ext

        def _blocked_cpp_ext_load(*args, **kwargs):
            raise RuntimeError(
                "torch.utils.cpp_extension.load() and load_inline() are not permitted "
                'on the GPU server. Use a C++ language (e.g. "cuda_cpp") in your '
                "solution spec to compile on the compile server."
            )

        _cpp_ext.load = _blocked_cpp_ext_load
        _cpp_ext.load_inline = _blocked_cpp_ext_load

        _mod_name = (
            _entry_module_or_file.removesuffix(".py")
            .replace("/", ".")
            .replace(os.sep, ".")
        )
        _user_mod = importlib.import_module(_mod_name)
        user_fn = getattr(_user_mod, _entry_func_name)
    return user_fn


def _make_integrity_guard():
    namespace = globals()
    integrity = snapshot_critical_functions(
        namespace,
        [
            "check_eval_integrity",
            "time_runnable",
            "_call_and_collect_outputs",
            "allocate_outputs",
            "normalize_outputs",
            "check_monkey_patch",
            "check_lazy_outputs",
            "check_thread_injection",
        ],
    )
    verify = check_eval_integrity

    @contextmanager
    def guard():
        verify(integrity, namespace)
        try:
            yield
        finally:
            verify(integrity, namespace)

    return guard


def _serve(staging, channel, archive):
    definition = Definition.model_validate_json(
        (staging / "definition.json").read_bytes()
    )
    solution = Solution.model_validate_json((staging / "solution.json").read_bytes())
    config_file = staging / "config.json"
    config = (
        BenchmarkConfig(**json.loads(config_file.read_text()))
        if config_file.exists()
        else BenchmarkConfig()
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    guard = _make_integrity_guard()
    try:
        with guard():
            function = _load_solution(staging, solution)
    except RewardHackDetected as error:
        _send(channel, ["hack", str(error)])
        return
    set_seed(config.seed)
    _send(channel, ["ok", None])
    names = list(definition.outputs)
    dtypes = dict(zip(names, definition.torch_output_dtypes, strict=True))
    dps = solution.spec.destination_passing_style
    threads = None
    while True:
        try:
            command, axes, first = _receive(channel)
        except EOFError:
            return
        try:
            if sys.platform == "linux" and device != "cpu":
                inputs = torch.load(
                    f"/proc/self/fd/{archive.fileno()}",
                    mmap=True,
                    weights_only=True,
                    map_location=device,
                )
            else:
                archive.seek(0)
                inputs = torch.load(archive, weights_only=True, map_location=device)
            with guard():
                if command == "time":
                    outputs = allocate_outputs(definition, axes, device) if dps else []
                    check_monkey_patch()
                    check_thread_injection(threads, threading.active_count())
                    latency = time_runnable(
                        function,
                        inputs,
                        outputs,
                        device,
                        warmup=config.warmup_runs,
                        rep=config.iterations,
                        seed=config.seed,
                    )
                    check_thread_injection(threads, threading.active_count())
                else:
                    outputs = _call_and_collect_outputs(
                        function, inputs, dps, definition, axes, device, names, dtypes
                    )
                    check_lazy_outputs(outputs)
                    if first:
                        threads = threading.active_count()
        except RewardHackDetected as error:
            _send(channel, ["hack", str(error)])
            return
        except Exception as error:
            _send(
                channel, ["error", "".join(traceback.format_exception(error))[-8192:]]
            )
            continue
        del inputs
        if command == "run":
            outputs = [tensor.detach().cpu() for tensor in outputs]
        else:
            del outputs
        if device != "cpu":
            torch.cuda.empty_cache()
        if command == "time":
            _send(channel, ["ok", latency])
        else:
            _send_outputs(channel, outputs)
            del outputs


# ── Output helper ─────────────────────────────────────────────────────────────
def _emit(trace: Trace) -> None:
    """Write one Trace as strictly valid JSON to the real stdout.

    Uses allow_nan=False so that any unexpected NaN/Inf in the trace dict
    raises ValueError immediately instead of producing the non-standard
    JavaScript literal ``NaN`` which strict JSON parsers silently reject.
    """
    print(
        json.dumps(trace.model_dump(mode="json"), allow_nan=False),
        file=_real_stdout,
        flush=True,
    )


def _emit_worker_failure(index, error, context):
    status = (
        error.status
        if isinstance(error, WorkerError)
        else EvaluationStatus.RUNTIME_ERROR
    )
    terminal = (
        not isinstance(error, WorkerError) or status == EvaluationStatus.REWARD_HACK
    )
    affected = workloads[index:] if terminal else [workloads[index]]
    message = str(error) or type(error).__name__
    if status == EvaluationStatus.RUNTIME_ERROR:
        message = f"{context}: {message}"
    for workload in affected:
        _emit(
            Trace(
                definition=definition.name,
                solution=_solution_name,
                workload=workload,
                evaluation=_make_eval(status, _device, None, extra_msg=message),
            )
        )
    if terminal:
        sys.exit(0)


def _make_eval(
    status, device, log_path, *, correctness=None, performance=None, extra_msg=None
):
    parts = [p for p in (_clock_status_msg, extra_msg) if p]
    return make_eval(
        status,
        device,
        log_path,
        correctness=correctness,
        performance=performance,
        extra_msg="\n".join(parts) or None,
    )


def _evaluate_workload(_index, _workload):
    # Free GPU memory held by previous workload's tensors.
    # PyTorch's CUDA caching allocator retains freed blocks.
    # Without this cleanup, memory accumulates across workloads causing OOM.
    _inputs = None
    _ref_outputs = None
    _user_outputs = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _resolved_axes = definition.get_resolved_axes_values(_workload.axes)

    # -- Safetensors inputs --
    _safe_tensors: dict = {}
    if any(v.type == "safetensors" for v in _workload.inputs.values()):
        try:
            _safe_tensors = load_safetensors(definition, _workload, _safetensors_roots)
        except Exception as _e:
            _emit(
                Trace(
                    definition=definition.name,
                    solution=_solution_name,
                    workload=_workload,
                    evaluation=_make_eval(
                        EvaluationStatus.RUNTIME_ERROR,
                        _device,
                        None,
                        extra_msg=f"Failed to load safetensors: {_e}",
                    ),
                )
            )
            return

    # -- Correctness checking (10 rounds with fresh inputs) --
    # Running multiple rounds catches non-deterministic bugs and input-dependent
    # correctness issues.
    _custom_inputs_fn = ref_namespace.get(definition.custom_inputs_entrypoint)
    _correctness_failed = False
    _correctness = Correctness()

    for _round in range(10):
        # -- Generate inputs --
        try:
            _inputs = gen_inputs(
                definition,
                _workload,
                device=_device,
                safe_tensors=_safe_tensors or None,
                custom_inputs_fn=_custom_inputs_fn,
            )
            _worker.set_inputs(_inputs)
        except Exception as _e:
            _emit(
                Trace(
                    definition=definition.name,
                    solution=_solution_name,
                    workload=_workload,
                    evaluation=_make_eval(
                        EvaluationStatus.RUNTIME_ERROR,
                        _device,
                        None,
                        extra_msg=f"gen_inputs failed: {_e}",
                    ),
                )
            )
            _correctness_failed = True
            break

        # -- Run reference (always return-value style) to get ground-truth --
        try:
            _ref_outputs = _call_and_collect_outputs(
                ref_fn,
                _inputs,
                False,
                definition,
                _resolved_axes,
                _device,
                _output_names,
                _output_dtypes_torch,
            )
        except Exception as _e:
            _emit(
                Trace(
                    definition=definition.name,
                    solution=_solution_name,
                    workload=_workload,
                    evaluation=_make_eval(
                        EvaluationStatus.INVALID_REFERENCE,
                        _device,
                        None,
                        extra_msg=f"Reference run() failed: {_e}\n{traceback.format_exc()}",
                    ),
                )
            )
            _correctness_failed = True
            break

        if _round != 9 or not bench_config.benchmark_reference:
            _inputs = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            _user_outputs = _worker.run(_resolved_axes, _round == 0, _ref_outputs)
        except Exception as _e:
            _emit_worker_failure(_index, _e, "User function failed")
            _correctness_failed = True
            break

        # -- Numerical correctness check (every round) --
        _numerically_wrong = False
        for _ref_out, _usr_out in zip(_ref_outputs, _user_outputs):
            _c, _exceeds = compute_error_stats(_usr_out, _ref_out, _workload.tolerance)
            if _c.max_absolute_error > _correctness.max_absolute_error:
                _correctness = _c
            if _c.has_nan:
                _correctness = _c
            elif _c.has_inf and not _correctness.has_nan:
                _correctness = _c
            if _exceeds:
                _numerically_wrong = True
                break

        _ref_outputs = _user_outputs = _ref_out = _usr_out = None
        if _numerically_wrong:
            _emit(
                Trace(
                    definition=definition.name,
                    solution=_solution_name,
                    workload=_workload,
                    evaluation=_make_eval(
                        EvaluationStatus.INCORRECT_NUMERICAL,
                        _device,
                        None,
                        correctness=_correctness,
                    ),
                )
            )
            _correctness_failed = True
            break

    if _correctness_failed:
        return

    # -- User latency measurement --
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _sol_latency_ms = _worker.time(_resolved_axes)
    except Exception as _e:
        _emit_worker_failure(_index, _e, "Timing failed")
        return

    # -- Reference latency (for speedup factor) —always return-value style --
    # Inputs are cloned (not regenerated) since the reference cannot cheat.
    _ref_latency_ms = 0.0
    if bench_config.benchmark_reference:
        try:
            _ref_latency_ms = time_runnable(
                ref_fn,
                _inputs,
                [],
                _device,
                warmup=bench_config.warmup_runs,
                rep=bench_config.iterations,
                seed=bench_config.seed,
            )
        except Exception:
            pass

    _speedup = _ref_latency_ms / _sol_latency_ms if _sol_latency_ms > 0 else 0.0

    # -- Emit PASSED trace --
    _emit(
        Trace(
            definition=definition.name,
            solution=_solution_name,
            workload=_workload,
            evaluation=_make_eval(
                EvaluationStatus.PASSED,
                _device,
                None,
                correctness=_correctness,
                performance=Performance(
                    latency_ms=_sol_latency_ms,
                    reference_latency_ms=_ref_latency_ms,
                    speedup_factor=_speedup,
                ),
            ),
        )
    )


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        staging, channel_fd, archive_fd, parent = sys.argv[2:]
        if sys.platform == "linux":
            if ctypes.CDLL(None).prctl(
                1, signal.SIGKILL, 0, 0, 0
            ) or os.getppid() != int(parent):
                sys.exit(1)
        with (
            Connection(int(channel_fd)) as channel,
            os.fdopen(int(archive_fd), "rb") as archive,
        ):
            _serve(Path(staging), channel, archive)
        sys.exit(0)

    _resources = ExitStack()
    atexit.register(_resources.close)

    definition_dict, _workload_dicts = _load_problem()

    # ── Load config ───────────────────────────────────────────────────────────────
    _config_path = STAGING_DIR / "config.json"
    bench_config = (
        BenchmarkConfig(**json.loads(_config_path.read_text()))
        if _config_path.exists()
        else BenchmarkConfig()
    )

    # ── Parse definition ──────────────────────────────────────────────────────────
    definition = Definition(**definition_dict)
    workloads = [Workload(**w) for w in _workload_dicts]

    # ── Parse solution ────────────────────────────────────────────────────────────
    _solution = Solution(**json.loads((STAGING_DIR / "solution.json").read_text()))
    _solution_name = _solution.name

    # ── Exec reference code ───────────────────────────────────────────────────────
    # Write to a real file so decorators that call inspect.getsourcelines()
    # (e.g. @triton.jit) can read the source back.
    _ref_dir = _resources.enter_context(tempfile.TemporaryDirectory())
    _ref_file = Path(_ref_dir) / "_reference.py"
    _ref_file.write_text(definition.reference)
    try:
        _ref_spec = importlib.util.spec_from_file_location("_reference", _ref_file)
        _ref_module = importlib.util.module_from_spec(_ref_spec)
        _ref_spec.loader.exec_module(_ref_module)
    except Exception as _ref_err:
        raise RuntimeError(f"Failed to exec reference code: {_ref_err}") from _ref_err

    ref_namespace = vars(_ref_module)
    ref_fn = ref_namespace.get("run")
    if ref_fn is None:
        raise RuntimeError("Reference code does not define a top-level 'run' function")

    # ── Device ────────────────────────────────────────────────────────────────────
    _device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ── Safetensors blob roots ────────────────────────────────────────────────────
    # Priority: 1) staging dir (client-inlined blobs), 2) flashinfer-trace directory.
    _safetensors_roots = [STAGING_DIR]
    _benchmark_dir = os.environ.get("FLASHINFER_TRACE_DIR", None)
    if _benchmark_dir:
        _safetensors_roots.append(Path(_benchmark_dir))

    # ── Pre-compute output metadata (used by normalize_outputs for DPS=false) ─────
    _output_names = list(definition.outputs.keys())
    _output_dtypes_torch = {
        k: dtype_str_to_torch_dtype(v.dtype) for k, v in definition.outputs.items()
    }

    # ── Evaluate each workload ────────────────────────────────────────────────────
    _device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""

    # Check whether clocks are locked (set by Docker entrypoint / server startup).
    _clocks_locked = are_clocks_locked()

    # Only include clock status in trace logs when the user explicitly requested
    # clock locking (lock_clocks=True).  When lock_clocks=False (default), the
    # status is noise and misleads users into thinking failures are clock-related.
    _clock_status_msg: Optional[str] = None
    if bench_config.lock_clocks:
        _clock_status_msg = (
            "Clocks locked: yes" if _clocks_locked else "Clocks locked: no"
        )

    _worker = _resources.enter_context(solution_worker(STAGING_DIR))
    try:
        _worker.reply()
    except WorkerError as _error:
        _emit_worker_failure(0, _error, "Solution import failed")

    # ── Validate clock lock requirement ───────────────────────────────────────────
    # If lock_clocks=True but clocks are not locked, reject all workloads.
    if bench_config.lock_clocks and not _clocks_locked:
        _reject_msg = "lock_clocks=True but GPU clocks are not locked on this server"
        for _wl in workloads:
            _emit(
                Trace(
                    definition=definition.name,
                    solution=_solution_name,
                    workload=_wl,
                    evaluation=_make_eval(
                        EvaluationStatus.RUNTIME_ERROR,
                        _device,
                        None,
                        extra_msg=_reject_msg,
                    ),
                )
            )
        sys.exit(0)

    set_seed(bench_config.seed)
    for _index, _workload in enumerate(workloads):
        _evaluate_workload(_index, _workload)
