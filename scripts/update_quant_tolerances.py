#!/usr/bin/env python
"""Update tolerance settings for quantization workloads.

Tolerances are based on gpu_mode.md research (GPU Mode benchmark industry-standard
values), with slightly looser settings. Includes max_error_cap (cuDNN pattern)
to prevent solutions with rare but arbitrarily large outlier errors.

Usage:
    # Preview changes
    python scripts/update_quant_tolerances.py --category Quant --dry-run

    # Update all Quant problems
    python scripts/update_quant_tolerances.py --category Quant

    # Update specific problem
    python scripts/update_quant_tolerances.py --problem-dir data/benchmark/Quant/...
"""

import argparse
import json
from pathlib import Path


# Tolerances based on gpu_mode.md research, slightly looser than gpu_mode values.
# gpu_mode uses: FP8 atol=1e-3 rtol=2e-2, NVFP4 atol=1e-3 rtol=1e-3.
# We use slightly higher atol (5e-3) for safety margin.
TOLERANCE_PRESETS = {
    "fp8": {
        "max_atol": 5e-3,
        "max_rtol": 2e-2,
        "required_matched_ratio": 0.95,
        "max_error_cap": 1.0,
    },
    "nvfp4": {
        "max_atol": 5e-3,
        "max_rtol": 5e-3,
        "required_matched_ratio": 0.95,
        "max_error_cap": 1.0,
    },
}


def detect_quant_type(problem_name: str) -> str:
    """Detect quantization type from problem name."""
    name_lower = problem_name.lower()
    if "nvfp4" in name_lower or "fp4" in name_lower:
        return "nvfp4"
    if "fp8" in name_lower:
        return "fp8"
    return "unknown"


def update_workload_file(
    workload_path: Path,
    tolerance_spec: dict,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Update correctness specs in a workload.jsonl file.

    Returns:
        (num_updated, num_total) - number of workloads updated and total workloads
    """
    if not workload_path.exists():
        print(f"  Warning: {workload_path} not found")
        return 0, 0

    workloads = []
    with open(workload_path) as f:
        for line in f:
            if line.strip():
                workloads.append(json.loads(line))

    num_updated = 0
    for workload in workloads:
        old = workload.get("correctness", {})
        if old != tolerance_spec:
            workload["correctness"] = tolerance_spec
            num_updated += 1

    if not dry_run and num_updated > 0:
        with open(workload_path, "w") as f:
            for workload in workloads:
                f.write(json.dumps(workload) + "\n")

    return num_updated, len(workloads)


def process_problem_dir(problem_dir: Path, dry_run: bool) -> None:
    """Process a single problem directory."""
    problem_name = problem_dir.name
    workload_path = problem_dir / "workload.jsonl"

    if not (problem_dir / "definition.json").exists():
        return

    quant_type = detect_quant_type(problem_name)
    if quant_type == "unknown":
        print(f"\n{problem_name}")
        print(f"  Warning: Cannot detect quant type, skipping")
        return

    tolerance_spec = TOLERANCE_PRESETS[quant_type]

    print(f"\n{problem_name}")
    print(f"  Type: {quant_type}")
    print(f"  Tolerance: atol={tolerance_spec['max_atol']}, "
          f"rtol={tolerance_spec['max_rtol']}, "
          f"ratio={tolerance_spec['required_matched_ratio']}, "
          f"cap={tolerance_spec['max_error_cap']}")

    num_updated, num_total = update_workload_file(workload_path, tolerance_spec, dry_run)

    if num_updated > 0:
        status = "[DRY RUN]" if dry_run else "[UPDATED]"
        print(f"  {status} {num_updated}/{num_total} workloads updated")
    else:
        print(f"  [SKIPPED] Already up to date")


def main():
    parser = argparse.ArgumentParser(
        description="Update tolerance settings for quantization workloads",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Process all problems in a category (e.g., 'Quant')",
    )
    parser.add_argument(
        "--problem-dir",
        type=Path,
        default=None,
        help="Process a specific problem directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying files",
    )

    args = parser.parse_args()

    if not args.category and not args.problem_dir:
        parser.error("Must specify --category or --problem-dir")

    if args.dry_run:
        print("DRY RUN: No files will be modified")
    print("=" * 60)

    if args.problem_dir:
        process_problem_dir(args.problem_dir, args.dry_run)
    else:
        benchmark_dir = Path("data/benchmark") / args.category
        if not benchmark_dir.exists():
            print(f"Error: Category directory not found: {benchmark_dir}")
            return 1

        problem_dirs = sorted(d for d in benchmark_dir.iterdir() if d.is_dir())
        print(f"Found {len(problem_dirs)} problems in {args.category}")

        for problem_dir in problem_dirs:
            process_problem_dir(problem_dir, args.dry_run)

    print("\n" + "=" * 60)
    if args.dry_run:
        print("Dry run complete. Use without --dry-run to apply changes.")
    else:
        print("Update complete!")

    return 0


if __name__ == "__main__":
    exit(main())
