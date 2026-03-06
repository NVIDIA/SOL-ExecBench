"""Run every problem through ``scripts/run_problem.py`` and record results.

Each problem directory is executed as a subprocess so that failures (OOM,
segfaults, assertion errors) in one problem cannot affect the rest.

The CSV is written incrementally (after each problem) so partial results
survive crashes.  On resume, problems that already have rows in the CSV
are skipped.

Output CSV columns:

    definition_hash, problem, category, workload_uuid, status

where *definition_hash* is the SHA-256 hex digest of the problem's
``definition.json`` at verification time, and *status* is ``PASSED``,
``FAILED``, or ``ERROR`` (the subprocess crashed before producing
per-workload results).  On resume, problems whose definition hash has
changed since the last run are automatically re-verified, as are
problems whose workload UUIDs have changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Resolve the run_problem.py script relative to the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUN_PROBLEM_SCRIPT = _REPO_ROOT / "scripts" / "run_problem.py"

_RESULT_RE = re.compile(r"^RESULT:\s+(\S+)\s+(PASSED|FAILED)$", re.MULTILINE)

_CSV_HEADER = ["definition_hash", "problem", "category", "workload_uuid", "status"]


@dataclass
class _WorkloadResult:
    definition_hash: str
    problem: str
    category: str
    workload_uuid: str
    status: str  # PASSED | FAILED | ERROR


def _definition_hash(problem_dir: Path) -> str:
    """Return the SHA-256 hex digest of the problem's definition.json."""
    defn = problem_dir / "definition.json"
    if not defn.exists():
        return ""
    return hashlib.sha256(
        json.dumps(json.loads(defn.read_text()), sort_keys=True).encode()
    ).hexdigest()


def _discover_problems(benchmark_dir: Path) -> list[tuple[Path, str]]:
    """Return ``(problem_dir, category)`` pairs for every problem in the benchmark."""
    problems: list[tuple[Path, str]] = []
    for defn in sorted(benchmark_dir.rglob("definition.json")):
        problem_dir = defn.parent
        rel = problem_dir.relative_to(benchmark_dir)
        if not (problem_dir / "workload.jsonl").exists():
            continue
        if not (problem_dir / "reference.py").exists():
            continue
        # Category is the first component if there's a subfolder (L1/L2/Quant/FlashInfer-Bench).
        category = rel.parts[0] if len(rel.parts) > 1 else ""
        problems.append((problem_dir, category))
    return problems


def _workload_uuids(problem_dir: Path) -> set[str]:
    """Return the set of workload UUIDs currently on disk for *problem_dir*."""
    wl = problem_dir / "workload.jsonl"
    if not wl.exists():
        return set()
    uuids: set[str] = set()
    for line in wl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        uid = obj.get("uuid")
        if uid:
            uuids.add(uid)
    return uuids


def _load_completed(csv_path: Path) -> dict[str, str]:
    """Return ``{problem_name: definition_hash}`` for problems in *csv_path*."""
    if not csv_path.exists():
        return {}
    completed: dict[str, str] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed[row["problem"]] = row["definition_hash"]
    return completed


def _load_completed_uuids(csv_path: Path) -> dict[str, set[str]]:
    """Return ``{problem_name: {uuid, ...}}`` for problems in *csv_path*."""
    if not csv_path.exists():
        return {}
    result: dict[str, set[str]] = defaultdict(set)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["workload_uuid"]
            if uid != "*":  # skip ERROR sentinel rows
                result[row["problem"]].add(uid)
    return result


def _load_failed(csv_path: Path) -> set[str]:
    """Return the set of problem names with FAILED or ERROR status in *csv_path*."""
    if not csv_path.exists():
        return set()
    failed: set[str] = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] in ("FAILED", "ERROR"):
                failed.add(row["problem"])
    return failed


def _remove_problems_from_csv(csv_path: Path, problems: set[str]) -> None:
    """Rewrite *csv_path* without rows belonging to *problems*."""
    if not csv_path.exists() or not problems:
        return
    rows: list[list[str]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row["problem"] not in problems:
                rows.append(row)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_one(problem_dir: Path) -> tuple[list[tuple[str, str]], str | None]:
    """Run ``run_problem.py`` on *problem_dir* and parse per-workload results.

    Returns ``([(uuid, status), ...], error_message_or_None)``.
    """
    cmd = [
        sys.executable,
        str(_RUN_PROBLEM_SCRIPT),
        str(problem_dir),
        "--num-workloads",
        "0",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        output = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        return [], "timeout (600s)"
    except Exception as e:
        return [], str(e)

    results = _RESULT_RE.findall(output)
    error = None if proc.returncode == 0 else output[-2000:]
    return results, error


def _append_rows(csv_path: Path, rows: list[_WorkloadResult]) -> None:
    """Append *rows* to the CSV, writing the header if the file is new."""
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(_CSV_HEADER)
        for r in rows:
            writer.writerow(
                [r.definition_hash, r.problem, r.category, r.workload_uuid, r.status]
            )


def verify_all_workloads(
    benchmark_dir: Path,
    *,
    rerun_failed: bool = False,
    subset: str | None = None,
) -> None:
    """Execute every problem in *benchmark_dir* and write a summary CSV.

    If *subset* is given, only problems whose category matches *subset* are
    verified, and results are written to ``verification_{subset}.csv``.
    """
    problems = _discover_problems(benchmark_dir)
    if subset:
        problems = [(d, c) for d, c in problems if c == subset]
    if not problems:
        print("No problems found" + (f" for subset '{subset}'" if subset else ""))
        return

    csv_name = f"verification_{subset}.csv" if subset else "verification.csv"
    csv_path = benchmark_dir / csv_name
    completed = _load_completed(csv_path)  # {problem_name: definition_hash}

    if rerun_failed:
        failed = _load_failed(csv_path)
        rerun_names = set(completed) & failed
        if not rerun_names:
            print("No failed or errored problems to re-run")
            return
        _remove_problems_from_csv(csv_path, rerun_names)
        completed = _load_completed(csv_path)
        to_run = [(d, c) for d, c in problems if d.name in rerun_names]
        skipped = len(problems) - len(to_run)
        print(f"Found {len(problems)} problems total")
        print(f"Re-running {len(to_run)} previously failed/errored problems\n")
    else:
        # Detect problems whose definition.json or workload UUIDs have changed.
        stale_def: set[str] = set()
        stale_wl: set[str] = set()
        completed_uuids = _load_completed_uuids(csv_path)
        for problem_dir, _cat in problems:
            name = problem_dir.name
            if name not in completed:
                continue
            if completed[name] != _definition_hash(problem_dir):
                stale_def.add(name)
            elif _workload_uuids(problem_dir) != completed_uuids.get(name, set()):
                stale_wl.add(name)
        stale = stale_def | stale_wl
        if stale:
            _remove_problems_from_csv(csv_path, stale)
            completed = _load_completed(csv_path)

        to_run = [(d, c) for d, c in problems if d.name not in completed]
        skipped = len(problems) - len(to_run)
        print(f"Found {len(problems)} problems total")
        if stale_def:
            print(
                f"{len(stale_def)} problem(s) have changed definition.json -- will re-verify"
            )
        if stale_wl:
            print(
                f"{len(stale_wl)} problem(s) have changed workloads -- will re-verify"
            )
        if skipped:
            print(f"Skipping {skipped} already-verified problems")
        if not to_run:
            print("All problems already verified")
            return
        print(f"Verifying {len(to_run)} problems\n")

    total_passed = 0
    total_failed = 0
    total_errors = 0

    for i, (problem_dir, category) in enumerate(to_run, 1):
        problem_name = problem_dir.name
        label = f"{category}/{problem_name}" if category else problem_name
        print(f"[{i}/{len(to_run)}] {label} ... ", end="", flush=True)

        workload_results, error = _run_one(problem_dir)
        rows: list[_WorkloadResult] = []
        defn_hash = _definition_hash(problem_dir)

        if workload_results:
            passed = sum(1 for _, s in workload_results if s == "PASSED")
            failed = sum(1 for _, s in workload_results if s == "FAILED")
            total_passed += passed
            total_failed += failed
            for uuid, status in workload_results:
                rows.append(
                    _WorkloadResult(defn_hash, problem_name, category, uuid, status)
                )
            if failed:
                print(f"{passed} passed, {failed} failed")
            else:
                print(f"{passed} passed")
        else:
            total_errors += 1
            rows.append(
                _WorkloadResult(defn_hash, problem_name, category, "*", "ERROR")
            )
            snippet = (error or "unknown error")[:200]
            print(f"ERROR: {snippet}")

        _append_rows(csv_path, rows)

    # Summary
    total = total_passed + total_failed + total_errors
    print(f"\nSummary: {total} workloads across {len(to_run)} problems")
    print(f"   Passed:  {total_passed}")
    if total_failed:
        print(f"   Failed:  {total_failed}")
    if total_errors:
        print(f"   Errors:  {total_errors}")
    print(f"\nCSV saved to: {csv_path}")


def verify_one(problem_dir: Path) -> None:
    """Run a single problem directory and print results to stdout (no CSV)."""
    problem_dir = problem_dir.resolve()
    if not (problem_dir / "definition.json").exists():
        print(f"No definition.json in {problem_dir}")
        sys.exit(1)
    if not (problem_dir / "workload.jsonl").exists():
        print(f"No workload.jsonl in {problem_dir}")
        sys.exit(1)

    print(f"Verifying {problem_dir.name} ...")
    workload_results, error = _run_one(problem_dir)

    if workload_results:
        passed = sum(1 for _, s in workload_results if s == "PASSED")
        failed = sum(1 for _, s in workload_results if s == "FAILED")
        for uuid, status in workload_results:
            print(f"  {uuid}: {status}")
        print(f"\n{passed} passed, {failed} failed")
        if failed:
            sys.exit(1)
    else:
        print(f"ERROR: {error or 'unknown error'}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify benchmark problems by running reference implementations"
    )
    parser.add_argument(
        "--problem-dir",
        type=Path,
        default=None,
        help="Path to a single problem directory to verify (debug mode, no CSV output)",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("data/benchmark"),
        help="Root directory containing benchmark problems (default: data/benchmark)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Only verify problems in this category (e.g. L1, L2, Quant, FlashInfer-Bench)",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="Re-run only previously failed/errored problems instead of discovering new ones",
    )
    args = parser.parse_args()

    if args.problem_dir:
        verify_one(args.problem_dir)
    else:
        verify_all_workloads(
            args.benchmark_dir,
            rerun_failed=args.rerun_failed,
            subset=args.subset,
        )


if __name__ == "__main__":
    main()
