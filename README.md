# SOL ExecBench

<div align="center" id="top">

[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](/docs/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

[HuggingFace Dataset](https://huggingface.co/datasets/nvidia/SOL-ExecBench) | [Leaderboard](https://research.nvidia.com/benchmarks/sol-execbench)
| [Technical Report](https://arxiv.org/abs/2603.19173) </div>

**Speed-Of-Light ExecBench** is a rigorous GPU kernel evaluation and benchmarking framework built to benchmark AI-generated kernel solutions written with the variety of DSLs that NVIDIA hardware supports.

Kernels are:
* Checked for various forms of reward hacking
* Tested against a reference solution for numerical correctness
* Timed under reproducible conditions

Leaderboard submissions are ranked based on [SOL-Score](/src/sol_execbench/sol_score.py): a metric that grades custom kernel performance based on the theoretical roofline of a NVIDIA B200 GPU (obtained analytically with [SOLAR](https://github.com/NVlabs/SOLAR)).  

Supported kernel languages: PyTorch, Triton, CUTLASS, cuDNN, CuTe DSL, cuTile, CUDA C++.

## Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) (`pip install huggingface-hub[cli]`)
- NVIDIA driver version 580+

## Setup

### 1. Download benchmark data (one-time)

```bash
./scripts/download_data.sh
```

This downloads the [SOL-ExecBench](https://huggingface.co/datasets/nvidia/SOL-ExecBench) and [FlashInfer Trace](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace) datasets into `data/`.

### 2. Build and launch the Docker container

```bash
./scripts/run_docker.sh --build
```

This builds the image and drops you into an interactive shell inside the container. The repo's `src/`, `tests/`, and downloaded data are mounted automatically.

## Evaluating a Solution

Inside the container, use the `sol-execbench` CLI:

```bash
# Evaluate using a problem directory (contains definition.json + workload.jsonl)
sol-execbench <problem_dir> --solution solution.json

# Or specify files explicitly
sol-execbench --definition def.json --workload wkl.jsonl --solution sol.json
```

### Example

```bash
# From the host — build, launch, and evaluate in one command:
./scripts/run_docker.sh --build -- \
  sol-execbench examples/cute_dsl/jamba_attn_proj \
    --solution examples/cute_dsl/jamba_attn_proj/solution_cute_dsl.json

# Or from inside the container:
sol-execbench examples/cute_dsl/jamba_attn_proj \
  --solution examples/cute_dsl/jamba_attn_proj/solution_cute_dsl.json
```

### CLI Options

| Flag | Description |
|---|---|
| `--compile-timeout` | Compilation timeout in seconds (default: 120) |
| `--timeout` | Evaluation timeout in seconds (default: 600) |
| `--config` | Path to a BenchmarkConfig JSON (see [Benchmark Config](#benchmark-config) below) |
| `-o, --output` | Write JSONL traces to file |
| `--json` | Print traces as JSON to stdout |
| `--lock-clocks` | Lock GPU clocks for stable benchmarks |
| `--keep-staging` | Preserve staging directory after run |
| `-v, --verbose` | Show subprocess output |

### Benchmark Config

Pass `--config bench.json` to override evaluator defaults. All fields are optional.

| Field | Type | Default | Description |
|---|---|---|---|
| `warmup_runs` | int | `10` | GPU warmup iterations before timing |
| `iterations` | int | `50` | Timing iterations averaged into the latency report |
| `lock_clocks` | bool | `false` | Require GPU clocks to be locked (also exposed as `--lock-clocks`) |
| `benchmark_reference` | bool | `false` | When `true`, also time the reference implementation to compute speedup. **Disabled by default** because the reference can be dramatically slower than the kernel (sometimes >1 h), which dominates total evaluation time. Enable when you need a speedup factor in the trace. |
| `seed` | int | `200` | RNG seed for input generation |

### Evaluation Environment

Official evaluations run on NVIDIA B200 GPUs with SM clocks locked at 1500 MHz. CPU hardware and CPU-side load may vary between evaluation workers, so approximately 5% run-to-run latency variance is expected. Submissions that launch many small kernels are more sensitive to CPU launch and scheduling overhead and may see up to 10% variance.

A template with every field at its default value lives at [`bench_config.example.json`](bench_config.example.json) — copy it, edit the fields you want to override, and pass it via `--config`:

```bash
cp bench_config.example.json bench.json   # then edit bench.json
sol-execbench <problem_dir> --solution solution.json --config bench.json
```

## Running a Dataset

Use `scripts/run_dataset.py` to evaluate an entire dataset (or a single problem) in batch. By default it runs the definition's reference implementation as the solution unless `--solution-name` is specified. Saves to `./out/{subset}` by default.

```bash
# Run all problems in the benchmark.
# Auto builds solution.json from a single code file
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --solution-name solution.py

# Run specific categories with multiple solution code files
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --category L1 L2 --solution-name solution.json

# Run a single problem
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark/L1/my_problem

# Limit number of problems and workloads
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --limit 5 --max-workloads 3 -o ./results
```

Results (traces and a summary JSON) are written to `out/run_dataset/` by default (override with `-o`). Problems that already passed are skipped on subsequent runs unless `--rerun` is specified.

## Problem Format

A problem directory contains:

- **`definition.json`** — Kernel specification: function signature, tensor shapes, dtypes, reference implementation.
- **`workload.jsonl`** — One JSON object per line, each defining input shapes, values, and tolerance thresholds.

A solution is a separate JSON file referencing source files with the kernel implementation.

See the full schema docs:

- [Definition](docs/definition.md) — Kernel specification (function signature, tensor shapes, dtypes, reference code)
- [Workload](docs/workload.md) — Concrete input configurations and tolerance thresholds
- [Solution](docs/solution.md) — Source files and build specs for a kernel implementation
- [Trace](docs/trace.md) — Evaluation output (correctness and performance results)

## Citation

```bibtex
@misc{lin2026solexecbench,
      title={SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits}, 
      author={Edward Lin, Sahil Modi, Siva Kumar Sastry Hari, Qijing Huang, Zhifan Ye, Nestor Qin, Fengzhe Zhou, Yuan Zhang, Jingquan Wang, Sana Damani, Dheeraj Peri, Ouye Xie, Aditya Kane, Moshe Maor, Michael Behar, Triston Cao, Rishabh Mehta, Vartika Singh, Vikram Sharma Mailthody, Terry Chen, Zihao Ye, Hanfeng Chen, Tianqi Chen, Vinod Grover, Wei Chen, Wei Liu, Eric Chung, Luis Ceze, Roger Bringmann, Cyril Zeller, Michael Lightstone, Christos Kozyrakis, Humphrey Shi},
      year={2026},
      eprint={2603.19173},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.19173}, 
}
```

## License

Apache-2.0. See [LICENSE](LICENSE). Contributions require DCO sign-off — see [CONTRIBUTING.md](CONTRIBUTING.md).
