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

"""Problem loader for the evaluate subcommand."""

from __future__ import annotations

from pathlib import Path

from .data import (
    Definition,
    Workload,
    load_json_file,
    load_jsonl_file,
)


class ProblemLoader:
    """Loads problems from the driver format."""

    @staticmethod
    def load_problem(
        problem_dir: Path,
    ) -> tuple[Definition, list[Workload], str | None]:
        """Load a problem from a directory containing definition.json and workload.jsonl.

        Parameters
        ----------
        problem_dir : Path
            Path to the problem directory.

        Returns
        -------
        tuple[Definition, list[Workload], str | None]
            A tuple containing the definition, list of workloads, and reference implementation code (None if not present).

        Raises
        ------
        FileNotFoundError
            If required files are missing (definition.json or workload.jsonl).
        ValueError
            If the files cannot be parsed.
        """
        definition_path = problem_dir / "definition.json"
        workload_path = problem_dir / "workload.jsonl"
        reference_path = problem_dir / "reference.py"

        if not definition_path.exists():
            raise FileNotFoundError(f"definition.json not found in {problem_dir}")
        if not workload_path.exists():
            raise FileNotFoundError(f"workload.jsonl not found in {problem_dir}")

        definition = load_json_file(Definition, definition_path)
        workloads = load_jsonl_file(Workload, workload_path)
        reference_code = reference_path.read_text() if reference_path.exists() else None

        return definition, workloads, reference_code

    @staticmethod
    def list_problems(drivers_dir: Path) -> list[str]:
        """List all available problems in the drivers_v2 directory.

        Parameters
        ----------
        drivers_dir : Path
            Path to the drivers_v2 directory.

        Returns
        -------
        list[str]
            List of problem names (directory names).
        """
        if not drivers_dir.exists():
            return []

        problems = [
            item.name
            for item in drivers_dir.iterdir()
            if item.is_dir()
            and (item / "definition.json").exists()
            and (item / "workload.jsonl").exists()
        ]
        return sorted(problems)
