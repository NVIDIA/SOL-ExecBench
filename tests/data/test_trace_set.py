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

import sys
from pathlib import Path

import pytest
import torch

from sol_execbench.data import (
    AxisConst,
    AxisVar,
    BuildSpec,
    Correctness,
    Definition,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    RandomInput,
    Solution,
    SourceFile,
    SupportedHardware,
    SupportedLanguages,
    TensorSpec,
    Trace,
    TraceSet,
    Workload,
    save_json_file,
    save_jsonl_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REF = "def run(a):\n    return a\n"


def _make_definition(name: str = "d1", op_type: str = "op") -> Definition:
    return Definition(
        name=name,
        op_type=op_type,
        axes={"M": AxisVar(), "N": AxisConst(value=2)},
        inputs={"a": TensorSpec(shape=["M", "N"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference=_REF,
    )


def _make_solution(name: str, def_name: str = "d1") -> Solution:
    return Solution(
        name=name,
        definition=def_name,
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=[SupportedHardware.LOCAL],
            entry_point="main.py::run",
        ),
        sources=[SourceFile(path="main.py", content="def run():\n    pass\n")],
    )


def _passed_eval(latency_ms: float = 1.0, speedup: float = 2.0) -> Evaluation:
    return Evaluation(
        status=EvaluationStatus.PASSED,
        log="",
        environment=Environment(hardware="LOCAL"),
        timestamp="t",
        correctness=Correctness(max_relative_error=0.0, max_absolute_error=0.0),
        performance=Performance(
            latency_ms=latency_ms,
            reference_latency_ms=latency_ms * speedup,
            speedup_factor=speedup,
        ),
    )


def _error_eval(
    status: EvaluationStatus = EvaluationStatus.RUNTIME_ERROR,
) -> Evaluation:
    return Evaluation(
        status=status,
        log="err",
        environment=Environment(hardware="LOCAL"),
        timestamp="t",
    )


def _workload(m: int = 4, uuid: str = "w") -> Workload:
    return Workload(axes={"M": m}, inputs={"A": RandomInput()}, uuid=uuid)


# ---------------------------------------------------------------------------
# In-memory TraceSet construction
# ---------------------------------------------------------------------------


def test_empty_trace_set():
    ts = TraceSet()
    assert ts.root is None
    assert ts.definitions == {}
    assert ts.solutions == {}
    assert ts.workloads == {}
    assert ts.traces == {}


def test_get_solution_found_and_missing():
    sol = _make_solution("s1")
    ts = TraceSet(solutions={"d1": [sol]})
    assert ts.get_solution("s1") is sol
    assert ts.get_solution("nonexistent") is None


def test_duplicate_solution_name_raises():
    sol = _make_solution("same")
    with pytest.raises(ValueError, match="Duplicate solution name"):
        TraceSet(solutions={"d1": [sol, sol]})


def test_root_path_properties_require_root():
    ts = TraceSet()
    with pytest.raises(ValueError):
        _ = ts.definitions_path
    with pytest.raises(ValueError):
        _ = ts.solutions_path
    with pytest.raises(ValueError):
        _ = ts.workloads_path
    with pytest.raises(ValueError):
        _ = ts.traces_path
    with pytest.raises(ValueError):
        _ = ts.blob_path


# ---------------------------------------------------------------------------
# from_path: load from directory structure
# ---------------------------------------------------------------------------


def test_trace_set_from_path(tmp_path: Path):
    (tmp_path / "definitions").mkdir()
    (tmp_path / "solutions").mkdir()
    (tmp_path / "workloads").mkdir()
    (tmp_path / "traces").mkdir()

    definition = _make_definition()
    save_json_file(definition, tmp_path / "definitions" / "d1.json")

    sol1 = _make_solution("s1")
    sol2 = _make_solution("s2")
    save_json_file(sol1, tmp_path / "solutions" / "s1.json")
    save_json_file(sol2, tmp_path / "solutions" / "s2.json")

    trace_pass = Trace(
        definition="d1",
        workload=_workload(4, "tw1"),
        solution="s1",
        evaluation=_passed_eval(1.0),
    )
    trace_fail = Trace(
        definition="d1",
        workload=_workload(4, "tw2"),
        solution="s2",
        evaluation=_error_eval(),
    )
    trace_workload = Trace(definition="d1", workload=_workload(8, "tw3"))

    save_jsonl_file([trace_workload], tmp_path / "workloads" / "d1.jsonl")
    save_jsonl_file([trace_pass, trace_fail], tmp_path / "traces" / "d1.jsonl")

    ts = TraceSet.from_path(str(tmp_path))

    assert ts.definitions["d1"].name == "d1"
    assert ts.get_solution("s1").name == "s1"
    assert ts.get_solution("s2").name == "s2"
    assert len(ts.workloads["d1"]) == 1
    assert len(ts.traces["d1"]) == 2


def test_from_path_empty_directory(tmp_path: Path):
    # Should not raise; everything is just empty
    ts = TraceSet.from_path(str(tmp_path))
    assert ts.definitions == {}
    assert ts.traces == {}


def test_from_path_duplicate_definition_raises(tmp_path: Path):
    (tmp_path / "definitions").mkdir()
    d = _make_definition("dup")
    save_json_file(d, tmp_path / "definitions" / "a.json")
    save_json_file(d, tmp_path / "definitions" / "b.json")  # same name
    with pytest.raises(ValueError, match="Duplicate definition"):
        TraceSet.from_path(str(tmp_path))


def test_from_path_duplicate_solution_raises(tmp_path: Path):
    (tmp_path / "solutions").mkdir()
    s = _make_solution("dup_sol")
    save_json_file(s, tmp_path / "solutions" / "a.json")
    save_json_file(s, tmp_path / "solutions" / "b.json")  # same name
    with pytest.raises(ValueError, match="Duplicate solution"):
        TraceSet.from_path(str(tmp_path))


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


def test_summary_empty():
    ts = TraceSet()
    s = ts.summary()
    assert s["total"] == 0
    assert s["passed"] == 0
    assert s["failed"] == 0
    assert s["min_latency_ms"] is None
    assert s["avg_latency_ms"] is None
    assert s["max_latency_ms"] is None


def test_summary_with_traces():
    t_pass1 = Trace(
        definition="d1",
        workload=_workload(uuid="u1"),
        solution="s1",
        evaluation=_passed_eval(1.0),
    )
    t_pass2 = Trace(
        definition="d1",
        workload=_workload(uuid="u2"),
        solution="s2",
        evaluation=_passed_eval(3.0),
    )
    t_fail = Trace(
        definition="d1",
        workload=_workload(uuid="u3"),
        solution="s3",
        evaluation=_error_eval(),
    )
    ts = TraceSet(traces={"d1": [t_pass1, t_pass2, t_fail]})

    s = ts.summary()
    assert s["total"] == 3
    assert s["passed"] == 2
    assert s["failed"] == 1
    assert s["min_latency_ms"] == pytest.approx(1.0)
    assert s["max_latency_ms"] == pytest.approx(3.0)
    assert s["avg_latency_ms"] == pytest.approx(2.0)


def test_summary_no_passed_traces():
    t_fail = Trace(
        definition="d1",
        workload=_workload(uuid="u1"),
        solution="s1",
        evaluation=_error_eval(),
    )
    ts = TraceSet(traces={"d1": [t_fail]})

    s = ts.summary()
    assert s["total"] == 1
    assert s["passed"] == 0
    assert s["failed"] == 1
    assert s["min_latency_ms"] is None


# ---------------------------------------------------------------------------
# filter_traces()
# ---------------------------------------------------------------------------


def test_filter_traces_by_error_bounds():
    good = Evaluation(
        status=EvaluationStatus.PASSED,
        log="",
        environment=Environment(hardware="h"),
        timestamp="t",
        correctness=Correctness(max_relative_error=1e-6, max_absolute_error=1e-8),
        performance=Performance(latency_ms=1.0),
    )
    bad = Evaluation(
        status=EvaluationStatus.PASSED,
        log="",
        environment=Environment(hardware="h"),
        timestamp="t",
        correctness=Correctness(max_relative_error=0.5, max_absolute_error=0.5),
        performance=Performance(latency_ms=0.5),
    )
    t_good = Trace(
        definition="d1", workload=_workload(uuid="u1"), solution="s1", evaluation=good
    )
    t_bad = Trace(
        definition="d1", workload=_workload(uuid="u2"), solution="s2", evaluation=bad
    )
    ts = TraceSet(traces={"d1": [t_good, t_bad]})

    results = ts.filter_traces("d1", atol=1e-2, rtol=1e-2)
    assert len(results) == 1
    assert results[0].solution == "s1"


def test_filter_traces_unknown_definition():
    ts = TraceSet()
    assert ts.filter_traces("nonexistent") == []


# ---------------------------------------------------------------------------
# get_best_trace()
# ---------------------------------------------------------------------------


def test_get_best_trace_picks_highest_speedup():
    fast = Trace(
        definition="d1",
        workload=_workload(4, "u1"),
        solution="fast",
        evaluation=_passed_eval(latency_ms=1.0, speedup=5.0),
    )
    slow = Trace(
        definition="d1",
        workload=_workload(4, "u2"),
        solution="slow",
        evaluation=_passed_eval(latency_ms=2.0, speedup=2.0),
    )
    ts = TraceSet(traces={"d1": [fast, slow]})

    best = ts.get_best_trace("d1", axes={"M": 4})
    assert best is not None
    assert best.solution == "fast"


def test_get_best_trace_filters_by_axes():
    t_m4 = Trace(
        definition="d1",
        workload=_workload(4, "u1"),
        solution="s_m4",
        evaluation=_passed_eval(speedup=3.0),
    )
    t_m8 = Trace(
        definition="d1",
        workload=_workload(8, "u2"),
        solution="s_m8",
        evaluation=_passed_eval(speedup=10.0),
    )
    ts = TraceSet(traces={"d1": [t_m4, t_m8]})

    # Only M=4 is considered
    best = ts.get_best_trace("d1", axes={"M": 4})
    assert best is not None
    assert best.solution == "s_m4"


def test_get_best_trace_excludes_failed():
    t_fail = Trace(
        definition="d1",
        workload=_workload(uuid="u1"),
        solution="s1",
        evaluation=_error_eval(),
    )
    ts = TraceSet(traces={"d1": [t_fail]})
    assert ts.get_best_trace("d1") is None


def test_get_best_trace_excludes_high_error():
    high_err = Evaluation(
        status=EvaluationStatus.PASSED,
        log="",
        environment=Environment(hardware="h"),
        timestamp="t",
        correctness=Correctness(max_relative_error=0.9, max_absolute_error=0.9),
        performance=Performance(latency_ms=0.1, speedup_factor=100.0),
    )
    t = Trace(
        definition="d1",
        workload=_workload(uuid="u1"),
        solution="s1",
        evaluation=high_err,
    )
    ts = TraceSet(traces={"d1": [t]})
    assert ts.get_best_trace("d1", max_abs_error=1e-2, max_rel_error=1e-2) is None


def test_get_best_trace_unknown_definition():
    ts = TraceSet()
    assert ts.get_best_trace("nonexistent") is None


# ---------------------------------------------------------------------------
# add_traces() and add_workload_traces()
# ---------------------------------------------------------------------------


def test_add_traces_in_memory(tmp_path: Path):
    definition = _make_definition()
    ts = TraceSet(root=tmp_path, definitions={"d1": definition})

    t = Trace(
        definition="d1",
        workload=_workload(uuid="u1"),
        solution="s1",
        evaluation=_passed_eval(),
    )
    ts.add_traces([t])

    assert len(ts.traces["d1"]) == 1
    assert ts.traces["d1"][0].solution == "s1"


def test_add_traces_persists_to_disk(tmp_path: Path):
    definition = _make_definition()
    ts = TraceSet(root=tmp_path, definitions={"d1": definition})

    t = Trace(
        definition="d1",
        workload=_workload(uuid="u1"),
        solution="s1",
        evaluation=_passed_eval(),
    )
    ts.add_traces([t])

    # The JSONL file should exist under traces/<op_type>/d1.jsonl
    jsonl_path = tmp_path / "traces" / "op" / "d1.jsonl"
    assert jsonl_path.exists()

    # Reload and verify
    ts2 = TraceSet.from_path(str(tmp_path))
    assert len(ts2.traces["d1"]) == 1


def test_add_traces_unknown_definition_raises():
    ts = TraceSet()
    t = Trace(
        definition="missing",
        workload=_workload(uuid="u1"),
        solution="s1",
        evaluation=_passed_eval(),
    )
    with pytest.raises(ValueError, match="Definition missing not found"):
        ts.add_traces([t])


def test_add_workload_traces_in_memory(tmp_path: Path):
    definition = _make_definition()
    ts = TraceSet(root=tmp_path, definitions={"d1": definition})

    wt = Trace(definition="d1", workload=_workload(uuid="wt1"))
    ts.add_workload_traces([wt])

    assert len(ts.workloads["d1"]) == 1
    assert ts.workloads["d1"][0].is_workload_trace()


def test_add_workload_traces_persists_to_disk(tmp_path: Path):
    definition = _make_definition()
    ts = TraceSet(root=tmp_path, definitions={"d1": definition})

    wt = Trace(definition="d1", workload=_workload(uuid="wt1"))
    ts.add_workload_traces([wt])

    jsonl_path = tmp_path / "workloads" / "op" / "d1.jsonl"
    assert jsonl_path.exists()


def test_add_workload_traces_unknown_definition_raises():
    ts = TraceSet()
    wt = Trace(definition="missing", workload=_workload(uuid="wt1"))
    with pytest.raises(ValueError, match="Definition missing not found"):
        ts.add_workload_traces([wt])


# ---------------------------------------------------------------------------
# backup_traces()
# ---------------------------------------------------------------------------


def test_backup_traces_moves_traces_dir(tmp_path: Path):
    ts = TraceSet(root=tmp_path)
    traces_path = tmp_path / "traces"
    traces_path.mkdir()
    (traces_path / "some.jsonl").write_text('{"dummy": 1}\n')

    ts.backup_traces()

    # Original traces dir is now empty (recreated)
    assert traces_path.exists()
    assert not list(traces_path.iterdir())

    # Backup directory exists with original content
    backups = [p for p in tmp_path.iterdir() if p.name.startswith("traces_bak_")]
    assert len(backups) == 1
    assert (backups[0] / "some.jsonl").exists()


def test_backup_traces_no_root_raises():
    ts = TraceSet()
    with pytest.raises(ValueError):
        ts.backup_traces()


# ---------------------------------------------------------------------------
# add_workload_blob_tensor() and get_workload_blob_tensor()
# ---------------------------------------------------------------------------


def test_add_and_get_blob_tensor(tmp_path: Path):
    definition = _make_definition(op_type="rmsnorm")
    ts = TraceSet(root=tmp_path, definitions={"d1": definition})

    tensors = {"x": torch.ones(4, 8), "w": torch.zeros(8)}
    rel_path = ts.add_workload_blob_tensor("d1", "uuid-abc", tensors)

    assert rel_path.endswith(".safetensors")

    loaded = ts.get_workload_blob_tensor(rel_path)
    assert "x" in loaded
    assert "w" in loaded
    assert loaded["x"].shape == (4, 8)
    assert loaded["w"].shape == (8,)


def test_add_blob_tensor_duplicate_raises(tmp_path: Path):
    definition = _make_definition(op_type="rmsnorm")
    ts = TraceSet(root=tmp_path, definitions={"d1": definition})

    tensors = {"x": torch.ones(2)}
    ts.add_workload_blob_tensor("d1", "same-uuid", tensors)

    with pytest.raises(ValueError, match="already exists"):
        ts.add_workload_blob_tensor("d1", "same-uuid", tensors)


def test_get_blob_tensor_missing_raises(tmp_path: Path):
    ts = TraceSet(root=tmp_path)
    with pytest.raises(ValueError, match="File not found"):
        ts.get_workload_blob_tensor("blob/workloads/nonexistent.safetensors")


def test_add_blob_tensor_no_root_raises():
    ts = TraceSet()
    with pytest.raises(ValueError):
        ts.add_workload_blob_tensor("d1", "u1", {})


# ---------------------------------------------------------------------------
# to_dict()
# ---------------------------------------------------------------------------


def test_to_dict_structure():
    definition = _make_definition()
    sol = _make_solution("s1")
    t = Trace(
        definition="d1",
        workload=_workload(uuid="u1"),
        solution="s1",
        evaluation=_passed_eval(),
    )
    ts = TraceSet(
        definitions={"d1": definition},
        solutions={"d1": [sol]},
        traces={"d1": [t]},
    )
    d = ts.to_dict()
    assert "definitions" in d
    assert "solutions" in d
    assert "traces" in d
    assert "d1" in d["definitions"]
    assert len(d["solutions"]["d1"]) == 1
    assert len(d["traces"]["d1"]) == 1


if __name__ == "__main__":
    pytest.main(sys.argv)
