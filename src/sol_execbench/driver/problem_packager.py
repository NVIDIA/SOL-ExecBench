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

"""Package a problem (definition + workloads + solution) into a staging directory.

Produces shell commands that the CLI can run directly via subprocess to compile
C++/CUDA solutions and evaluate them on the GPU.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from ..core import (
    BenchmarkConfig,
    Definition,
    Solution,
    SupportedHardware,
    SupportedLanguages,
    Trace,
    Workload,
)

_TEMPLATES_DIR = Path(__file__).parent / "templates"

_CPP_LANGUAGES = {
    SupportedLanguages.CUDA_CPP,
    SupportedLanguages.CUTLASS,
    SupportedLanguages.CUDNN,
    SupportedLanguages.CUBLAS,
}

_BLACKWELL_HARDWARE = {SupportedHardware.B200}


def _get_local_sm() -> str | None:
    """Detect the local GPU's SM version via nvidia-smi."""
    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=compute_cap",
                    "--format=csv,noheader",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .splitlines()[0]
            .strip()
        )
        return f"sm_{out.replace('.', '')}"
    except Exception:
        return None


def _sm_to_gencode(sm: str) -> str:
    """Convert an SM version string (e.g. 'sm_90', 'sm_100a') to a gencode flag."""
    arch = sm.removeprefix("sm_")
    return f"-gencode=arch=compute_{arch},code={sm}"


def _canonical_name(name: str) -> str:
    """PEP 503 normalized distribution name (lowercase, runs of -_. -> -)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(req: str) -> str:
    """Extract the canonical distribution name from a pip requirement string."""
    m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req)
    return _canonical_name(m.group(1)) if m else ""


class ProblemPackager:
    """Stage files for compilation and execution, returning commands for the CLI."""

    def __init__(
        self,
        definition: Definition,
        workloads: list[Workload],
        solution: Solution,
        config: BenchmarkConfig,
        output_dir: Path,
        keep_output_dir: bool = False,
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_output_dir = keep_output_dir

        self.definition = definition
        self.workloads = workloads
        self.solution = solution
        self.config = config

        # Write problem files to staging directory up front.
        (self.output_dir / "definition.json").write_text(definition.model_dump_json())
        (self.output_dir / "workload.jsonl").write_text(
            "\n".join(w.model_dump_json() for w in workloads)
        )
        (self.output_dir / "solution.json").write_text(solution.model_dump_json())
        (self.output_dir / "config.json").write_text(
            json.dumps(dataclasses.asdict(config))
        )
        self._write_sources()

    def __del__(self):
        if not self.keep_output_dir:
            shutil.rmtree(self.output_dir, ignore_errors=True)

    @property
    def _is_cpp(self) -> bool:
        return any(lang in _CPP_LANGUAGES for lang in self.solution.spec.languages)

    def _inject_gencode_flags(self, sol_dict: dict) -> dict:
        """Auto-inject -gencode flags when no explicit arch flag is set.

        Blackwell targets get sm_100a (required for tcgen05/TMEM instructions).
        LOCAL target detects the compile machine's GPU.
        """
        spec = sol_dict["spec"]
        compile_options = dict(spec.get("compile_options") or {})
        cuda_cflags = list(compile_options.get("cuda_cflags", []))

        if any("-gencode" in f or "-arch" in f for f in cuda_cflags):
            return sol_dict

        gencode_sms: list[str] = []
        target_hw = {h.upper() for h in spec.get("target_hardware", [])}

        if any(h == hw.value for h in target_hw for hw in _BLACKWELL_HARDWARE):
            gencode_sms.append("sm_100a")

        if SupportedHardware.LOCAL.value in target_hw:
            local_sm = _get_local_sm()
            if local_sm:
                gencode_sms.append(local_sm)

        # Deduplicate preserving order
        seen: set[str] = set()
        unique = [s for s in gencode_sms if not (s in seen or seen.add(s))]
        if unique:
            compile_options["cuda_cflags"] = [
                _sm_to_gencode(sm) for sm in unique
            ] + cuda_cflags
            spec["compile_options"] = compile_options
            sol_dict["spec"] = spec

        return sol_dict

    def _write_sources(self) -> None:
        """Write solution source files to the staging directory."""
        for src in self.solution.sources:
            dest = self.output_dir / src.path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(src.content)

    def install_deps(self) -> tuple[list[str], str] | None:
        """Return (command, target_dir) to install pip_packages, or None if empty.

        Installs into an isolated target_dir the CLI prepends to PYTHONPATH. With
        ``UV_FIND_LINKS`` set, locks the install to that offline wheelhouse
        (``--no-index --offline --no-deps``); otherwise resolves from the index.
        """
        pkgs = self.solution.spec.pip_packages
        if not pkgs:
            return None
        target = self.output_dir / "_pip_deps"
        cmd = [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--target",
            str(target),
        ]
        wheelhouse = os.environ.get("UV_FIND_LINKS")
        if wheelhouse:
            cmd += [
                "--no-index",
                "--find-links",
                wheelhouse,
                "--offline",
                "--no-deps",
            ]
        cmd += list(pkgs)
        return cmd, str(target)

    def check_pip_packages_allowed(self) -> tuple[list[str], list[str]]:
        """Return (disallowed_requirements, available_wheels) for the wheelhouse.

        In wheelhouse mode (``UV_FIND_LINKS`` set) a pip_packages entry is allowed
        only if a wheel for its distribution is present in the wheelhouse — that dir
        is the curated allowlist. Returns the disallowed entries plus the available
        wheel filenames (for a clear error). Outside wheelhouse mode returns
        ([], []): there is no allowlist to enforce (index resolution, dev only).
        """
        wheelhouse = os.environ.get("UV_FIND_LINKS")
        pkgs = self.solution.spec.pip_packages
        if not wheelhouse or not pkgs:
            return [], []
        wh = Path(wheelhouse)
        available = sorted(p.name for p in wh.glob("*.whl")) if wh.is_dir() else []
        allowed = {_canonical_name(f.split("-", 1)[0]) for f in available}
        disallowed = [p for p in pkgs if _requirement_name(p) not in allowed]
        return disallowed, available

    def compile(self) -> tuple[list[str], str]:
        """Stage compilation files and return (command, artifact_path).

        Writes build_ext.py, solution.json, and C++/CUDA source files to
        output_dir. Injects gencode flags for the target hardware.

        The CLI should run the command in output_dir.
        After success, the artifact (benchmark_kernel.so) will be at artifact_path.
        """
        assert self._is_cpp, (
            f"compile() only handles C++/CUDA solutions, "
            f"got languages={self.solution.spec.languages}"
        )

        sol_dict = json.loads(self.solution.model_dump_json())
        sol_dict = self._inject_gencode_flags(sol_dict)

        # Overwrite solution.json with injected gencode flags.
        (self.output_dir / "solution.json").write_text(json.dumps(sol_dict))
        (self.output_dir / "build_ext.py").write_text(
            (_TEMPLATES_DIR / "build_ext.py").read_text()
        )

        cmd = ["python", "build_ext.py"]
        artifact_path = str(self.output_dir / "benchmark_kernel.so")

        return cmd, artifact_path

    def execute(self) -> list[str]:
        """Stage execution files and return the command to run.

        Writes eval_driver.py, definition.json, workload.jsonl, solution.json
        to output_dir. For Python solutions, also writes source files. For C++
        solutions, expects benchmark_kernel.so to already exist in output_dir
        (produced by a prior compile() call).

        The CLI should run the command in output_dir.
        Trace JSON will be emitted on stdout (one JSON object per line).
        """
        if self._is_cpp:
            so_path = self.output_dir / "benchmark_kernel.so"
            if not so_path.exists():
                raise FileNotFoundError(
                    f"benchmark_kernel.so not found at {so_path} — "
                    "run compile() first for C++/CUDA solutions"
                )

        (self.output_dir / "eval_driver.py").write_text(
            (_TEMPLATES_DIR / "eval_driver.py").read_text()
        )

        return ["python", "eval_driver.py"]

    def convert_stdout_to_traces(self, stdout: str) -> list[Trace]:
        """Parse JSONL stdout from eval_driver.py into Trace objects.

        Each line starting with '{' is parsed as a Trace JSON object.
        Non-JSON lines (library noise redirected to stderr) are skipped.
        """
        traces = []
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                traces.append(Trace(**json.loads(line)))
        return traces
