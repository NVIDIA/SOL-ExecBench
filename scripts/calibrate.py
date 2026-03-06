"""Calibrate correctness thresholds for a benchmark problem.

This module estimates an error envelope for a problem by repeatedly running the
reference implementation with many randomized inputs and comparing:
1) GPU target precision vs CPU target precision, or
2) target precision vs higher precision (float64 where possible).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sol_execbench.bench.compile import BuilderRegistry, RunnableInputs
from sol_execbench.bench.evaluators.utils import normalize_result
from sol_execbench.bench.utils import gen_inputs, load_safetensors
from sol_execbench.data import Definition, Workload, load_json_file, load_jsonl_file

DEFAULT_CATEGORIES = ("L1", "L2", "Quant", "FlashInfer-Bench")


DEFAULT_RTOL_BY_DTYPE = {
    "torch.bfloat16": 1e-2,
    "torch.float16": 1e-3,
    "torch.float32": 1e-5,
    "torch.float64": 1e-8,
}


@dataclass
class ProbeError:
    max_abs: float
    required_atol: float


def _tensor_stats(t: torch.Tensor) -> dict[str, float]:
    x = t.to(torch.float32)
    if x.numel() == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
    }


def _compute_pair_error(a: torch.Tensor, b: torch.Tensor, rtol: float) -> ProbeError:
    """Compute error stats using torch.allclose-style tolerance.

    For each element, the allclose check is: |a - b| <= atol + rtol * |b|.
    The 'required_atol' is the minimum atol needed so that every element passes:
        required_atol = max(0, |a - b| - rtol * |b|) over all elements.
    """
    x = a.to(torch.float32)
    y = b.to(torch.float32)
    abs_err = torch.abs(x - y)
    if abs_err.numel() == 0:
        return ProbeError(max_abs=0.0, required_atol=0.0)
    # Minimum atol needed per element for allclose to pass
    needed_atol = abs_err - rtol * torch.abs(y)
    return ProbeError(
        max_abs=float(abs_err.max().item()),
        required_atol=float(needed_atol.max().clamp(min=0).item()),
    )


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _default_output_path(
    problem_dir: Path, workload_index: int | None = None, split_workloads: bool = False
) -> Path:
    """Return default calibration output path mirrored from data/benchmark to data/calibration."""
    filename = (
        f"workload_{workload_index}.json"
        if split_workloads and workload_index is not None
        else "calibration.json"
    )
    parts = problem_dir.parts
    if "data" in parts and "benchmark" in parts:
        data_idx = parts.index("data")
        benchmark_idx = parts.index("benchmark", data_idx + 1)
        if benchmark_idx == data_idx + 1:
            repo_root = Path(*parts[:data_idx]) if data_idx > 0 else Path(".")
            rel_after_benchmark = Path(*parts[benchmark_idx + 1 :])
            return repo_root / "data" / "calibration" / rel_after_benchmark / filename
    # Fallback: keep alongside the problem directory when no benchmark path pattern is found.
    return problem_dir / filename


def _iter_problem_dirs(benchmark_dir: Path, categories: list[str]) -> list[Path]:
    problem_dirs: list[Path] = []
    for category in categories:
        cat_dir = benchmark_dir / category
        if not cat_dir.exists():
            continue
        for problem_dir in sorted(cat_dir.iterdir()):
            if not problem_dir.is_dir():
                continue
            if not (problem_dir / "definition.json").exists():
                continue
            if not (problem_dir / "workload.jsonl").exists():
                continue
            problem_dirs.append(problem_dir)
    return problem_dirs


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, max(0, int(q * (len(sorted_vals) - 1))))
    return float(sorted_vals[idx])


def _clone_for_device(inputs: RunnableInputs, device: str) -> RunnableInputs:
    copied: list[Any] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            copied.append(value.to(device))
        else:
            copied.append(value)
    return RunnableInputs(
        callable_inputs=copied, resolved_axes=dict(inputs.resolved_axes)
    )


def _clone_for_high_precision_cpu(inputs: RunnableInputs) -> RunnableInputs:
    copied: list[Any] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                copied.append(value.detach().to(device="cpu", dtype=torch.float64))
            else:
                copied.append(value.detach().to(device="cpu"))
        else:
            copied.append(value)
    return RunnableInputs(
        callable_inputs=copied, resolved_axes=dict(inputs.resolved_axes)
    )


def _normalize_result_for_device(
    definition: Definition, result: Any, device: str
) -> list[torch.Tensor]:
    out = normalize_result(definition, result, device)
    return [t.detach() for t in out]


def _run_reference(
    definition: Definition, runnable: Any, inputs: RunnableInputs, device: str
) -> list[torch.Tensor]:
    with torch.no_grad():
        result = runnable(*inputs)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    return _normalize_result_for_device(definition, result, device)


def _calibrate_one(
    definition: Definition,
    workload: Workload,
    mode: str,
    probes: int,
    seed: int,
    threshold_percentile: float = 0.99,
    threshold_safety_factor: float = 1.25,
    include_probe_details: bool = False,
    print_probes: bool = False,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for calibration")

    gpu_device = "cuda:0"
    cpu_device = "cpu"
    reference = BuilderRegistry.get_instance().build_reference(definition)
    loaded_safe_tensors = (
        load_safetensors(definition, workload, None)
        if any(inp.type == "safetensors" for inp in workload.inputs.values())
        else {}
    )

    per_probe_abs: list[float] = []
    per_probe_atol: list[float] = []
    probe_details: list[dict[str, Any]] = []
    output_names = list(definition.outputs.keys())
    output_dtypes = dict(zip(output_names, definition.torch_output_dtypes))
    output_is_floating = {
        name: bool(dtype.is_floating_point) for name, dtype in output_dtypes.items()
    }
    # Determine rtol per output from dtype
    output_rtol: dict[str, float] = {}
    for name, dtype in output_dtypes.items():
        output_rtol[name] = DEFAULT_RTOL_BY_DTYPE.get(str(dtype), 1e-2)

    per_output_abs: dict[str, list[float]] = {name: [] for name in output_names}
    per_output_atol: dict[str, list[float]] = {name: [] for name in output_names}

    for probe_idx in range(probes):
        # Deterministic probe inputs: each probe gets a stable seed derived from base seed.
        _set_all_seeds(seed + probe_idx)
        inp_gpu = gen_inputs(
            definition=definition,
            workload=workload,
            runnable=reference,
            device=gpu_device,
            safe_tensors=loaded_safe_tensors,
        )

        gpu_out = _run_reference(definition, reference, inp_gpu, gpu_device)

        if mode == "gpu_cpu":
            inp_cpu = _clone_for_device(inp_gpu, cpu_device)
            other_out = _run_reference(definition, reference, inp_cpu, cpu_device)
        elif mode == "high_precision":
            inp_hi = _clone_for_high_precision_cpu(inp_gpu)
            other_out = _run_reference(definition, reference, inp_hi, cpu_device)
            other_out = [
                t.to(dtype=ref.dtype) if t.is_floating_point() else t
                for t, ref in zip(other_out, gpu_out)
            ]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        trial_abs = 0.0
        trial_atol = 0.0
        per_output: list[dict[str, Any]] = []
        for out_name, out_gpu, out_other in zip(output_names, gpu_out, other_out):
            rtol = output_rtol[out_name]
            err = _compute_pair_error(
                out_gpu.detach().cpu(), out_other.detach().cpu(), rtol
            )
            trial_abs = max(trial_abs, err.max_abs)
            per_output_abs[out_name].append(err.max_abs)
            if output_is_floating[out_name]:
                trial_atol = max(trial_atol, err.required_atol)
                per_output_atol[out_name].append(err.required_atol)
            if include_probe_details:
                per_output.append(
                    {
                        "name": out_name,
                        "max_abs": err.max_abs,
                        "required_atol": err.required_atol
                        if output_is_floating[out_name]
                        else None,
                        "rtol": rtol,
                        "dtype": str(output_dtypes[out_name]),
                        "gpu_stats": _tensor_stats(out_gpu),
                        "other_stats": _tensor_stats(out_other),
                    }
                )
        per_probe_abs.append(trial_abs)
        per_probe_atol.append(trial_atol)
        if include_probe_details:
            probe_details.append(
                {
                    "probe": probe_idx,
                    "max_abs": trial_abs,
                    "required_atol": trial_atol,
                    "outputs": per_output,
                }
            )
        if print_probes:
            print(
                f"[probe {probe_idx + 1:04d}/{probes}] "
                f"max_abs={trial_abs:.6e} required_atol={trial_atol:.6e}"
            )

    sorted_abs = sorted(per_probe_abs)
    sorted_atol = sorted(per_probe_atol) if per_probe_atol else []

    result: dict[str, Any] = {
        "workload_uuid": workload.uuid,
        "mode": mode,
        "probes": probes,
        "tolerance_method": "allclose: |a-b| <= atol + rtol * |b|",
        "max_abs": max(per_probe_abs) if per_probe_abs else 0.0,
        "max_required_atol": max(per_probe_atol) if per_probe_atol else 0.0,
        "p95_abs": _percentile(sorted_abs, 0.95) if sorted_abs else 0.0,
        "p95_required_atol": _percentile(sorted_atol, 0.95) if sorted_atol else 0.0,
        "p99_abs": _percentile(sorted_abs, 0.99) if sorted_abs else 0.0,
        "p99_required_atol": _percentile(sorted_atol, 0.99) if sorted_atol else 0.0,
    }
    per_output_stats: dict[str, Any] = {}
    per_output_thresholds: dict[str, Any] = {}
    for out_name in output_names:
        out_abs = per_output_abs[out_name]
        out_atol = per_output_atol[out_name]
        rtol = output_rtol[out_name]
        out_abs_max = max(out_abs) if out_abs else 0.0
        out_atol_max = max(out_atol) if out_atol else 0.0
        out_atol_pct = _percentile(out_atol, threshold_percentile)
        per_output_stats[out_name] = {
            "dtype": str(output_dtypes[out_name]),
            "is_floating_output": output_is_floating[out_name],
            "rtol": rtol,
            "max_abs": out_abs_max,
            "max_required_atol": out_atol_max if output_is_floating[out_name] else None,
            "p95_abs": _percentile(out_abs, 0.95),
            "p95_required_atol": _percentile(out_atol, 0.95)
            if output_is_floating[out_name]
            else None,
            "p99_abs": _percentile(out_abs, 0.99),
            "p99_required_atol": _percentile(out_atol, 0.99)
            if output_is_floating[out_name]
            else None,
        }
        per_output_thresholds[out_name] = (
            {
                "atol": out_atol_pct * threshold_safety_factor,
                "rtol": rtol,
            }
            if output_is_floating[out_name]
            else {
                "atol": 0.0,
                "rtol": 0.0,
                "note": "non-floating output; enforce exact match",
            }
        )
    global_atol_pct = (
        _percentile(per_probe_atol, threshold_percentile) if per_probe_atol else 0.0
    )
    result["per_output_stats"] = per_output_stats
    result["recommended_thresholds"] = {
        "method": "allclose_percentile_times_safety_factor",
        "description": "Tolerance check: |a - b| <= atol + rtol * |b|",
        "percentile": threshold_percentile,
        "safety_factor": threshold_safety_factor,
        "global": {
            "atol": global_atol_pct * threshold_safety_factor,
            "rtol": max(output_rtol.values()) if output_rtol else 1e-2,
        },
        "per_output": per_output_thresholds,
    }
    if include_probe_details:
        result["probe_details"] = probe_details
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate correctness thresholds for a problem"
    )
    parser.add_argument(
        "--problem-dir",
        type=Path,
        required=False,
        help="Path to one problem dir containing definition.json and workload.jsonl",
    )
    parser.add_argument(
        "--all-problems",
        action="store_true",
        help="Calibrate all problems under benchmark-dir and categories",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("data/benchmark"),
        help="Benchmark root for --all-problems mode",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATEGORIES),
        help="Categories to include in --all-problems mode",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="gpu_cpu",
        choices=["gpu_cpu", "high_precision", "auto"],
        help="Calibration mode",
    )
    parser.add_argument(
        "--probes", type=int, default=200, help="Number of probes per workload"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=0.99,
        help="Percentile used to derive recommended thresholds (0,1]",
    )
    parser.add_argument(
        "--threshold-safety-factor",
        type=float,
        default=1.25,
        help="Multiplier applied to percentile error for recommended thresholds (>=1.0)",
    )
    parser.add_argument(
        "--no-print-probes",
        action="store_true",
        help="Disable per-probe console output (enabled by default)",
    )
    parser.add_argument(
        "--no-probe-details",
        action="store_true",
        help="Disable per-probe and per-output stats in output JSON (enabled by default)",
    )
    parser.add_argument(
        "--workload-index",
        type=int,
        default=0,
        help="Workload index in workload.jsonl to calibrate (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: mirrors data/benchmark/* under data/calibration/*)",
    )
    args = parser.parse_args()
    if not (0.0 < args.threshold_percentile <= 1.0):
        raise ValueError("--threshold-percentile must be in (0, 1]")
    if args.threshold_safety_factor < 1.0:
        raise ValueError("--threshold-safety-factor must be >= 1.0")
    print_probes = not args.no_print_probes
    include_probe_details = not args.no_probe_details

    if args.all_problems:
        problem_dirs = _iter_problem_dirs(args.benchmark_dir, args.categories)
        if not problem_dirs:
            raise RuntimeError(
                f"No problems found in {args.benchmark_dir} for categories {args.categories}"
            )

        total_workloads = 0
        total_ok = 0
        total_failed = 0

        for problem_dir in problem_dirs:
            definition = load_json_file(Definition, problem_dir / "definition.json")
            workloads = load_jsonl_file(Workload, problem_dir / "workload.jsonl")
            if not workloads:
                continue
            rel = problem_dir.relative_to(args.benchmark_dir)
            print(f"\n=== {rel} ({len(workloads)} workloads) ===")
            for workload_idx, selected in enumerate(workloads):
                total_workloads += 1
                print(f"--- workload {workload_idx}: {selected.uuid} ---")
                try:
                    mode = args.mode
                    if mode == "auto":
                        try:
                            result = _calibrate_one(
                                definition,
                                selected,
                                "gpu_cpu",
                                args.probes,
                                args.seed,
                                threshold_percentile=args.threshold_percentile,
                                threshold_safety_factor=args.threshold_safety_factor,
                                include_probe_details=include_probe_details,
                                print_probes=print_probes,
                            )
                            mode = "gpu_cpu"
                        except Exception:
                            result = _calibrate_one(
                                definition,
                                selected,
                                "high_precision",
                                args.probes,
                                args.seed,
                                threshold_percentile=args.threshold_percentile,
                                threshold_safety_factor=args.threshold_safety_factor,
                                include_probe_details=include_probe_details,
                                print_probes=print_probes,
                            )
                            mode = "high_precision"
                    else:
                        result = _calibrate_one(
                            definition,
                            selected,
                            mode,
                            args.probes,
                            args.seed,
                            threshold_percentile=args.threshold_percentile,
                            threshold_safety_factor=args.threshold_safety_factor,
                            include_probe_details=include_probe_details,
                            print_probes=print_probes,
                        )

                    output_path = (
                        args.output
                        if args.output
                        else _default_output_path(
                            problem_dir, workload_idx, split_workloads=True
                        )
                    )
                    payload = {
                        "problem": definition.name,
                        "workload_index": workload_idx,
                        "mode": mode,
                        "stats": result,
                    }
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(json.dumps(payload, indent=2))
                    print(f"Saved calibration to: {output_path}")
                    total_ok += 1
                except Exception as e:
                    print(
                        f"FAILED {definition.name} workload {workload_idx} ({selected.uuid}): {e}"
                    )
                    total_failed += 1

        print("\n=== Calibration Summary ===")
        print(f"Total workloads: {total_workloads}")
        print(f"Succeeded: {total_ok}")
        print(f"Failed: {total_failed}")
        return

    if args.problem_dir is None:
        raise ValueError("--problem-dir is required unless --all-problems is set")

    problem_dir = args.problem_dir
    definition = load_json_file(Definition, problem_dir / "definition.json")
    workloads = load_jsonl_file(Workload, problem_dir / "workload.jsonl")
    if not workloads:
        raise RuntimeError("No workloads found")
    if args.workload_index < 0 or args.workload_index >= len(workloads):
        raise ValueError(
            f"workload-index out of range: {args.workload_index} (num workloads: {len(workloads)})"
        )

    selected = workloads[args.workload_index]
    mode = args.mode
    if mode == "auto":
        try:
            result = _calibrate_one(
                definition,
                selected,
                "gpu_cpu",
                args.probes,
                args.seed,
                threshold_percentile=args.threshold_percentile,
                threshold_safety_factor=args.threshold_safety_factor,
                include_probe_details=include_probe_details,
                print_probes=print_probes,
            )
            mode = "gpu_cpu"
        except Exception:
            result = _calibrate_one(
                definition,
                selected,
                "high_precision",
                args.probes,
                args.seed,
                threshold_percentile=args.threshold_percentile,
                threshold_safety_factor=args.threshold_safety_factor,
                include_probe_details=include_probe_details,
                print_probes=print_probes,
            )
            mode = "high_precision"
    else:
        result = _calibrate_one(
            definition,
            selected,
            mode,
            args.probes,
            args.seed,
            threshold_percentile=args.threshold_percentile,
            threshold_safety_factor=args.threshold_safety_factor,
            include_probe_details=include_probe_details,
            print_probes=print_probes,
        )

    output_path = args.output or _default_output_path(problem_dir)
    payload = {
        "problem": definition.name,
        "workload_index": args.workload_index,
        "mode": mode,
        "stats": result,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))

    print(f"\nSaved calibration to: {output_path}")


if __name__ == "__main__":
    main()
