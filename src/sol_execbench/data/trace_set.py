"""TraceSet as a pure data warehouse for definitions, solutions, and traces."""

from __future__ import annotations

import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import safetensors.torch
import torch

from ..bench.config.env import get_sol_execbench_dataset_path

from .definition import Definition
from .json_utils import append_jsonl_file, load_json_file, load_jsonl_file
from .solution import Solution
from .trace import EvaluationStatus, Trace


@dataclass
class TraceSet:
    """Stores a FlashInfer Trace dataset containing definitions, solutions, workloads, and traces.

    TraceSet serves as a centralized data warehouse for managing FlashInfer benchmark data.
    It provides efficient lookup and filtering capabilities for definitions, solutions, and
    execution traces organized by definition names.

    The data structure is optimized for fast lookups with dictionary-based storage where
    keys are definition names and values are lists of associated objects.
    """

    root: Optional[Path] = None
    """The root path of the TraceSet. If None, all add() or get() operations will be performed
    in-memory."""
    definitions: dict[str, Definition] = field(default_factory=dict)
    """The definitions in the database. Map from definition name to Definition object."""
    solutions: dict[str, list[Solution]] = field(default_factory=dict)
    """The solutions in the database. Map from definition name to all the solutions for that
    definition."""
    workloads: dict[str, list[Trace]] = field(default_factory=dict)
    """The workload traces in the database. Map from definition name to all workload traces for that
    definition."""
    traces: dict[str, list[Trace]] = field(default_factory=dict)
    """The traces in the database. Map from definition name to all traces for that definition."""
    _solution_by_name: dict[str, Solution] = field(
        default_factory=dict, init=False, repr=False
    )
    """Fast lookup index: solution name -> Solution object. Automatically maintained."""

    def __post_init__(self):
        """Initialize the _solution_by_name index from existing solutions."""
        for solutions_list in self.solutions.values():
            for solution in solutions_list:
                if solution.name in self._solution_by_name:
                    raise ValueError(f"Duplicate solution name found: {solution.name}")
                self._solution_by_name[solution.name] = solution

    @property
    def definitions_path(self) -> Path:
        if self.root is None:
            raise ValueError("Root path is not set")
        return self.root / "definitions"

    @property
    def solutions_path(self) -> Path:
        if self.root is None:
            raise ValueError("Root path is not set")
        return self.root / "solutions"

    @property
    def workloads_path(self) -> Path:
        if self.root is None:
            raise ValueError("Root path is not set")
        return self.root / "workloads"

    @property
    def traces_path(self) -> Path:
        if self.root is None:
            raise ValueError("Root path is not set")
        return self.root / "traces"

    @property
    def blob_path(self) -> Path:
        if self.root is None:
            raise ValueError("Root path is not set")
        return self.root / "blob"

    @property
    def blob_workloads_path(self) -> Path:
        if self.root is None:
            raise ValueError("Root path is not set")
        return self.root / "blob" / "workloads"

    @classmethod
    def from_path(cls: type[TraceSet], path: Optional[str] = None) -> TraceSet:
        """Load a TraceSet from a directory structure.

        Loads a complete TraceSet by scanning the directory structure for:
        - definitions/: JSON files containing Definition objects
        - solutions/: JSON files containing Solution objects
        - workloads/: JSONL files containing Trace objects (workload traces)
        - traces/: JSONL files containing Trace objects (execution traces)

        Parameters
        ----------
        path : Optional[str], optional
            Root directory path containing the dataset structure. If None,
            the global environment variable SOLEXECBENCH_DATASET_PATH is used. If
            SOLEXECBENCH_DATASET_PATH is not set, ~/.cache/sol_execbench/dataset is used.

        Returns
        -------
        TraceSet
            A new TraceSet instance populated with data from the directory.

        Raises
        ------
        ValueError
            If duplicate definition names or solution names are found.
        FileNotFoundError
            If the specified path doesn't exist.
        """
        path = Path(path) if path else get_sol_execbench_dataset_path()

        # Create the path if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)

        trace_set = cls(root=path)

        # Load json files from all subdirectories
        for p in sorted((trace_set.definitions_path.rglob("*.json"))):
            d = load_json_file(Definition, p)
            if d.name in trace_set.definitions:
                raise ValueError(f"Duplicate definition name: {d.name}")
            trace_set.definitions[d.name] = d

        seen_solutions = set()
        for p in sorted((trace_set.solutions_path.rglob("*.json"))):
            s = load_json_file(Solution, p)
            if s.name in seen_solutions:
                raise ValueError(f"Duplicate solution name: {s.name}")
            seen_solutions.add(s.name)
            trace_set.solutions.setdefault(s.definition, []).append(s)
            trace_set._solution_by_name[s.name] = s

        for p in sorted((trace_set.workloads_path.rglob("*.jsonl"))):
            for t in load_jsonl_file(Trace, p):
                assert t.is_workload_trace()
                trace_set.workloads.setdefault(t.definition, []).append(t)

        for p in sorted((trace_set.traces_path.rglob("*.jsonl"))):
            for t in load_jsonl_file(Trace, p):
                assert not t.is_workload_trace()
                trace_set.traces.setdefault(t.definition, []).append(t)

        return trace_set

    def to_dict(self) -> dict[str, Any]:
        """Convert the TraceSet to a Python dict.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the TraceSet.
        """
        return {
            "definitions": {
                name: definition.model_dump(mode="json")
                for name, definition in self.definitions.items()
            },
            "solutions": {
                name: [solution.model_dump(mode="json") for solution in solutions]
                for name, solutions in self.solutions.items()
            },
            "workload": {
                name: [workload.model_dump(mode="json") for workload in workloads]
                for name, workloads in self.workloads.items()
            },
            "traces": {
                name: [trace.model_dump(mode="json") for trace in traces]
                for name, traces in self.traces.items()
            },
        }

    def get_solution(self, name: str) -> Optional[Solution]:
        """Get a solution by name from all loaded solutions.

        Uses an O(1) index lookup for fast retrieval. Since solution names are unique
        across the entire dataset, this returns at most one solution.

        Parameters
        ----------
        name : str
            The name of the solution to retrieve.

        Returns
        -------
        Optional[Solution]
            The solution with the given name, or None if not found.
        """
        return self._solution_by_name.get(name)

    def filter_traces(
        self, def_name: str, atol: float = 1e-2, rtol: float = 1e-2
    ) -> list[Trace]:
        """Filter traces for a definition based on error bounds.

        Returns only successful traces that meet the specified absolute and relative
        error tolerance criteria. This is useful for finding high-quality implementations
        that produce numerically accurate results.

        Parameters
        ----------
        def_name : str
            Name of the definition to filter traces for.
        atol : float, optional
            Maximum absolute error tolerance, by default 1e-2.
        rtol : float, optional
            Maximum relative error tolerance, by default 1e-2.

        Returns
        -------
        list[Trace]
            List of traces that passed evaluation and meet error criteria.
            Empty list if no traces match the criteria.
        """
        return [
            t
            for t in self.traces.get(def_name, [])
            if t.evaluation
            and t.evaluation.status == EvaluationStatus.PASSED
            and t.evaluation.correctness
            and t.evaluation.correctness.max_absolute_error <= atol
            and t.evaluation.correctness.max_relative_error <= rtol
        ]

    def get_best_trace(
        self,
        def_name: str,
        axes: Optional[dict[str, int]] = None,
        max_abs_error: float = 1e-2,
        max_rel_error: float = 1e-2,
    ) -> Optional[Trace]:
        """Get the best performing trace for a definition based on speedup factor.

        Finds the trace with the highest speedup factor among those that meet the
        specified criteria including axis constraints and error tolerances.

        Parameters
        ----------
        def_name : str
            Name of the definition to find the best trace for.
        axes : Optional[dict[str, int]], optional
            Dictionary of axis name to value pairs for exact matching.
            Only traces with exactly matching axis values will be considered.
            If None, all axis configurations are considered.
        max_abs_error : float, optional
            Maximum absolute error tolerance, by default 1e-2.
        max_rel_error : float, optional
            Maximum relative error tolerance, by default 1e-2.

        Returns
        -------
        Optional[Trace]
            The best performing trace meeting all criteria, or None if no traces
            match the requirements.
        """
        candidates = self.traces.get(def_name, [])

        # Axes exact match
        # TODO(shanli): advanced input filtering
        if axes:
            candidates = [
                t
                for t in candidates
                if all(t.workload.axes.get(ax) == val for ax, val in axes.items())
            ]

        # Filter by successful status
        candidates = [
            t
            for t in candidates
            if t.evaluation and t.evaluation.status == EvaluationStatus.PASSED
        ]

        # Filter by error bounds
        candidates = [
            t
            for t in candidates
            if t.evaluation.correctness.max_absolute_error <= max_abs_error
            and t.evaluation.correctness.max_relative_error <= max_rel_error
        ]

        # Return the one with best speedup
        if candidates:
            return max(
                candidates,
                key=lambda t: t.evaluation.performance.speedup_factor
                if t.evaluation
                else 0,
            )

        return None

    def summary(self) -> dict[str, any]:
        """Get a comprehensive summary of all traces in the TraceSet.

        Computes aggregate statistics across all execution traces including success rates,
        latency statistics, and overall dataset size metrics.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the following keys:
            - total: Total number of traces
            - passed: Number of traces with successful evaluation
            - failed: Number of traces with failed evaluation
            - min_latency_ms: Minimum latency among successful traces (None if no successful traces)
            - max_latency_ms: Maximum latency among successful traces (None if no successful traces)
            - avg_latency_ms: Average latency among successful traces (None if no successful traces)
        """
        all_traces = [t for traces in self.traces.values() for t in traces]

        if not all_traces:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "min_latency_ms": None,
                "max_latency_ms": None,
                "avg_latency_ms": None,
            }

        passed = [
            t
            for t in all_traces
            if t.evaluation and t.evaluation.status == EvaluationStatus.PASSED
        ]

        latencies = [t.evaluation.performance.latency_ms for t in passed]

        min_latency = min(latencies) if latencies else None
        max_latency = max(latencies) if latencies else None
        avg_latency = sum(latencies) / len(latencies) if latencies else None

        return {
            "total": len(all_traces),
            "passed": len(passed),
            "failed": len(all_traces) - len(passed),
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "avg_latency_ms": avg_latency,
        }

    def backup_traces(self) -> None:
        """Backup the traces directory to a new directory. This is useful when we want to keep the
        old traces for reference.
        """
        if self.root is None:
            raise ValueError("Root path is not set")
        traces_path = self.traces_path
        backup_path = (
            self.root / f"traces_bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Move traces directory to backup
        if traces_path.exists():
            shutil.move(str(traces_path), str(backup_path))
        else:
            backup_path.mkdir(parents=True, exist_ok=True)

        # Create new traces directory
        traces_path.mkdir(parents=True, exist_ok=True)

    def add_traces(self, traces: list[Trace]) -> None:
        """Add traces to the TraceSet, and store the traces to disk.

        Parameters
        ----------
        traces : list[Trace]
            The traces to add to the TraceSet.
        """
        buckets: dict[Path, list[Trace]] = defaultdict(list)
        for trace in traces:
            # Add to in-memory database
            if trace.definition not in self.definitions:
                raise ValueError(
                    f"Add trace failed: Definition {trace.definition} not found in TraceSet"
                )
            self.traces.setdefault(trace.definition, []).append(trace)

            # Add to disk bucket
            definition = self.definitions[trace.definition]
            path = self.traces_path / definition.op_type / f"{definition.name}.jsonl"
            buckets[path].append(trace)

        # Write to disk
        if self.root is not None:
            for path, traces in buckets.items():
                append_jsonl_file(traces, path)

    def add_workload_blob_tensor(
        self, def_name: str, workload_uuid: str, tensors: dict[str, torch.Tensor]
    ) -> str:
        """Store a dict of workload blob tensors to a safetensors file, and return the saved file
        path relative to the TraceSet root.

        Parameters
        ----------
        def_name : str
            The def name of the tensor.
        workload_uuid : str
            The workload uuid of the tensor.
        tensors : dict[str, torch.Tensor]
            The dict of tensors to store.

        Returns
        -------
        str
            The file path of the saved tensor.

        Raises
        ------
        ValueError
            If the root path is not set, or the constructed tensor path already exists.

        OSError
            If writing to disk fails.
        """
        if self.root is None:
            raise ValueError("Root path is not set")
        op_type = self.definitions[def_name].op_type
        file_path = (
            self.blob_workloads_path
            / op_type
            / def_name
            / f"{def_name}_{workload_uuid}.safetensors"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Throw error if the tensor path exists
        if file_path.exists():
            raise ValueError(f"Tensor save path already exists: {file_path}")
        cpu_tensors = {k: (v.cpu() if v.is_cuda else v) for k, v in tensors.items()}
        safetensors.torch.save_file(cpu_tensors, file_path)
        return str(file_path.relative_to(self.root))

    def get_workload_blob_tensor(self, path_str: str) -> dict[str, torch.Tensor]:
        """Get a workload blob tensor from disk to CPU.

        Parameters
        ----------
        path_str : str
            The file path of the tensor relative to the TraceSet root.

        Returns
        -------
        dict[str, torch.Tensor]
            The dict of tensors from the file.
        """
        if self.root is None:
            raise ValueError("Root path is not set")
        file_path = self.root / path_str
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")
        return safetensors.torch.load_file(file_path, device="cpu")

    def add_workload_traces(self, traces: list[Trace]) -> None:
        """Add workload traces to the TraceSet, and store the traces to disk.

        Parameters
        ----------
        traces : list[Trace]
            The traces to add to the TraceSet.
        """
        buckets: dict[Path, list[Trace]] = defaultdict(list)
        for trace in traces:
            # Add to in-memory database
            if trace.definition not in self.definitions:
                raise ValueError(
                    f"Add workload trace failed: Definition {trace.definition} "
                    "not found in TraceSet"
                )
            self.workloads.setdefault(trace.definition, []).append(trace)

            # Add to disk bucket
            definition = self.definitions[trace.definition]
            path = self.workloads_path / definition.op_type / f"{definition.name}.jsonl"
            buckets[path].append(trace)

        # Write to disk
        if self.root is not None:
            for path, traces in buckets.items():
                append_jsonl_file(traces, path)
