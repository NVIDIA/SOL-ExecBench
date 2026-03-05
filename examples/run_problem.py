import sys
import math
import json
from pathlib import Path
import logging
import argparse


from sol_execbench import (
    Definition,
    Workload,
    Solution,
    load_json_file,
    load_jsonl_file,
    SupportedHardware,
    EvaluationStatus,
    BuildSpec,
    SourceFile,
    SupportedLanguages,
    BenchmarkConfig,
    Benchmark,
)

ap = argparse.ArgumentParser()
ap.add_argument("path", type=Path, help="Path to the problem directory")
ap.add_argument(
    "--override-reference", action="store_true", help="Override the reference code"
)
ap.add_argument(
    "--num-workloads",
    type=int,
    default=3,
    help="Number of workloads to run. Run all if set to 0.",
)
args = ap.parse_args()

HERE = args.path.absolute()

logging.basicConfig(level=logging.INFO)


def main() -> int:
    definition_path = HERE / "definition.json"
    reference_path = HERE / "reference.py"
    workload_path = HERE / "workload.jsonl"

    definition = load_json_file(Definition, definition_path)
    if args.override_reference:
        definition.reference = reference_path.read_text()
    load_json_file(Definition, definition_path)
    workloads = load_jsonl_file(Workload, workload_path)

    # create no-op solution
    solution = Solution(
        name=definition.name + "_noop",
        definition=definition.name,
        author="foobar",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=[SupportedHardware.LOCAL],
            entry_point="main.py::run",
            dependencies=["torch"],
        ),
        sources=[
            SourceFile(path="main.py", content="def run(*args):\n    return None\n"),
        ],
    )

    # find smallest workloads
    numels = []
    for workload in workloads:
        all_shapes = definition.get_input_shapes(workload.axes)
        numels.append(
            sum(
                [
                    math.prod(shape) if shape is not None else 1
                    for shape in all_shapes.values()
                ]
            )
        )
    assert len(numels) == len(workloads)
    zipped = list(zip(workloads, numels))
    zipped.sort(key=lambda x: x[1])
    if args.num_workloads:
        workloads_smallest = [x[0] for x in zipped[: args.num_workloads]]
    else:
        args.num_workloads = len(zipped)
        workloads_smallest = [x[0] for x in zipped]

    config = BenchmarkConfig(
        lock_clocks=False,
        profile_baseline=False,
        num_trials=1,
        allow_failed_baseline=True,
    )
    benchmark = Benchmark(hardware=SupportedHardware.LOCAL, config=config)
    traces = benchmark.run_all(
        definitions=[definition],
        workloads_per_def={definition.name: workloads_smallest},
        solutions_per_def={definition.name: [solution]},
        trace_set_root=HERE,
    )
    assert len(traces) == args.num_workloads
    # print eval status for each trace
    total_failed = 0
    for trace in traces:
        if trace.evaluation.status != EvaluationStatus.INCORRECT_NUMERICAL:
            print("STATUS:", trace.evaluation.status)
            print("LOG:", trace.evaluation.log)
            print("RESULT:", trace.workload.uuid, "FAILED")
            total_failed += 1
        else:
            print("RESULT:", trace.workload.uuid, "PASSED")

    print("TOTAL FAILED:", total_failed)
    assert total_failed == 0
    # save updated definition file
    definition_path.write_text(json.dumps(definition.model_dump(), indent=2))
    print("Verification complete! All checks passed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
