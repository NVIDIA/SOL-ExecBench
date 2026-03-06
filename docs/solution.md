# Solution

## Overview

This document describes the schema for a workload **Solution**.

The `Solution` provides a concrete, high-performance implementation for a given [Definition](/docs/flashinfer-trace/definition). Each `Solution` is a self-contained entry submitted by community members or autonomous agents, encapsulating the source code and all metadata required for building, interfacing, and benchmarking. The `Solution` will be benchmarked to collect evaluation statistics to be stored in a [Trace object](/docs/flashinfer-trace/trace).

> **Tip:** Visit [FlashInfer Bench Viewer](https://bench.flashinfer.ai/viewer) to see formatted code and visualized JSON for existing solutions.

## JSON Schema Description

### Top-Level Object Structure

| **Field** | **Type** | **Required** | **Description** |
| --- | --- | --- | --- |
| `name` | string | Yes | A unique, human-readable name for this specific solution (e.g.,`rmsnorm_triton_v1_h100`). |
| `definition` | string | Yes | The`name`of the `Definition`this implementation solves. |
| `description` | string | No | A human-readable brief description of the solution's technique or agent policy. |
| `author` | string | Yes | The name of the author or agent system. |
| `spec` | object | Yes | An object containing detailed technical specifications for the implementation. |
| `sources` | array | Yes | An array of file objects representing the complete source code, including any necessary files for building and execution. |

### `sources` : Source Code Files

The `sources` array contains any number of file objects, where each object represents a single source file in the project. The `flashinfer-bench` benchmarker will reconstruct the project's directory structure to properly build the binaries/executables.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `path` | string | Yes | The relative path of the file, including its name and extension (e.g., `src/kernel.cu`, `main.py`). |
| `content` | string | Yes | The complete text content of the source file. |

### `spec` : Build Specification

This object details the build requirements and properties of the source code.

| **Field** | **Type** | **Required** | **Description** |
| --- | --- | --- | --- |
| `language` | string | Yes | The primary programming language: `python`, `triton`, `cpp`, or `cuda`. |
| `target_hardware` | array | Yes | A list of hardware architectures this is compatible with (e.g., `NVIDIA_H100`, `B200`). |
| `entry_point` | string | Yes | The exact path to the function to be called. Format: `{file_path}::{function_name}` (e.g. `main.py::run`). |
| `destination_passing_style` | bool | No | If `true` (default), outputs are passed as last arguments. If `false`, outputs are returned. See [Destination Passing Style](#destination-passing-style-dps). |
| `binding` | string | No | Binding type for C++/CUDA solutions: `torch`. Ignored for Python/Triton. See [Language-Specific Guidelines](#language-specific-guidelines). |
| `dependencies` | array | No | A list of required libraries or toolchains. |
| `compile_options` | object | No | Custom compiler flags for C++/CUDA solutions. Contains `cflags`, `cuda_cflags`, and `ld_flags`. See [Compile Options](#compile-options). |


### Dependencies Handling

> **Note:** The `dependencies` field is currently only for semantic purposes and is not enforced.
The support for dependencies is coming soon.

The `dependencies` field is an array of strings declaring third-party packages needed to build/run the solution. In particular, we're handling the third-party CUDA libs and Python packages:

- CUDA/C++: Use version-pinned tokens.

  Example: `CUTLASS_3_7` → the builder injects CUTLASS 3.7 headers paths during compilation.

- Python libs: You may list package specifiers, but we do not manage Python package installs currently. We only validate against the current environment. If a listed lib/version isn't satisfied, the build fails fast with a validation error.

  Example: `torch`, `triton >= 2.3` → the builder validates the current environment for these packages and versions.


### Compile Options

The `compile_options` field allows you to specify custom compiler flags for C++/CUDA solutions. These flags are added onto the Evaluator's default flag set.

| **Field** | **Type** | **Description** |
| --- | --- | --- |
| `cflags` | list[str] | GCC/G++ compiler flags (e.g., `-O3`, `-march=native`) |
| `cuda_cflags` | list[str] | NVCC compiler flags (e.g., `-use_fast_math`, `--maxrregcount=64`) |
| `ld_flags` | list[str] | Linker flags (e.g., `-lcublas`, `-L/path/to/libs`) |

**Note:** All specified flags are appended to the Evaluator's default compilation settings. Use this to fine-tune compilation behavior for specific hardware or optimization requirements.


## Destination Passing Style (DPS)

The `destination_passing_style` field controls how outputs are handled:

| Style | `destination_passing_style` | Signature | Description |
| --- | --- | --- | --- |
| **DPS** | `true` (default) | `run(input1, input2, ..., output1, output2, ...)` | Outputs are pre-allocated and passed as the last arguments. The function writes results in-place. |
| **Value-returning** | `false` | `run(input1, input2, ...) -> output` | The function allocates and returns the output tensor(s). |

**Example comparison:**

```python
# DPS (destination_passing_style: true)
def run(input, weight, output):
    # output is pre-allocated, write results in-place
    output[:] = normalize(input, weight)

# Value-returning (destination_passing_style: false)
def run(input, weight):
    # allocate and return the result
    return normalize(input, weight)
```

**When to use which:**
- **DPS (`true`)**: Preferred for performance-critical code. Avoids output allocation overhead.
- **Value-returning (`false`)**: Simpler to write and doesn't need to manually allocate outputs.

## Language-Specific Guidelines

### Python / Triton

For `python` and `triton` languages, the entry point is a Python function.

**Signature requirements:**
- Parameter names must exactly match the keys in `Definition.inputs` (and `Definition.outputs` for DPS).
- The function is called with **positional arguments**.

**Parameter handling:**
- `*args`: Useful for flexible number of parameters. It requires (required params ≤ expected).
- `**kwargs`: Ignored in signature validation.

**Example (Triton with value-returning):**

```python
def run(input, weight, **kwargs):
    # **kwargs is allowed and ignored in validation
    output = torch.empty_like(input)
    _triton_kernel[grid](input, weight, output, ...)
    return output
```

### C++ / CUDA

For `cpp` and `cuda` languages, the source code must be compiled. Use the `binding` field to specify how to interface with Python:

| Binding | Description | Use Case |
| --- | --- | --- |
| `torch` | PyTorch C++/CUDA extension via `torch.utils.cpp_extension`. | Standard binding for C++/CUDA solutions. |

**Entry point:** The `entry_point` should reference a C/C++ function symbol that will be exposed to Python.

**Example spec for CUDA:**

```json
{
  "language": "cuda",
  "target_hardware": ["NVIDIA_H100"],
  "entry_point": "kernel.cu::my_kernel",
  "binding": "torch",
  "destination_passing_style": true
}
```

## Examples

### Example: Triton Implementation for GEMM

```json
{
  "name": "gemm_triton_h100_v1",
  "definition": "gemm",
  "description": "A high-performance GEMM implementation (C = A @ B.T) using Triton.",
  "author": "gemini-2.5-pro-mystery-agent",
  "spec": {
    "language": "triton",
    "target_hardware": ["NVIDIA_H100"],
    "dependencies": [
      "triton >= 2.3",
      "torch"
    ],
    "entry_point": "main.py::run",
    "destination_passing_style": false
  },
  "sources": [
    {
      "path": "main.py",
      "content": "import torch\nimport triton\nimport triton.language as tl\n\n@triton.autotune(\n    configs=[\n        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),\n        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8)\n    ],\n    key=['M', 'N', 'K'],\n)\n@triton.jit\ndef _gemm_kernel(\n    A, B, C, M, N, K, stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr\n):\n    pid = tl.program_id(axis=0)\n    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)\n    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)\n    num_pid_in_group = GROUP_SIZE_M * num_pid_n\n    group_id = pid // num_pid_in_group\n    first_pid_m = group_id * GROUP_SIZE_M\n    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)\n    pid_m = first_pid_m + (pid % group_size_m)\n    pid_n = (pid % num_pid_in_group) // group_size_m\n\n    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None]\n    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :]\n    offs_k = tl.arange(0, BLOCK_SIZE_K)\n    a_ptrs = A + (offs_am * stride_am + offs_k[None, :] * stride_ak)\n    b_ptrs = B + (offs_bn * stride_bn + offs_k[:, None] * stride_bk)\n\n    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)\n    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):\n        a = tl.load(a_ptrs)\n        b = tl.load(b_ptrs)\n        accumulator += tl.dot(a, b)\n        a_ptrs += BLOCK_SIZE_K * stride_ak\n        b_ptrs += BLOCK_SIZE_K * stride_bk\n    c = accumulator.to(C.dtype.element_ty)\n\n    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)\n    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)\n    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]\n    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)\n    tl.store(c_ptrs, c, mask=c_mask)\n\ndef run(A, B):\n    M, K = A.shape\n    N, _ = B.shape\n    C = torch.empty((M, N), device=A.device, dtype=A.dtype)\n    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )\n    _gemm_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1))\n    return C"
    }
  ]
}
```

### Example: Triton Implementation for RMS Norm

```json
{
  "name": "rmsnorm_triton_v1",
  "definition": "rmsnorm",
  "description": "A high-performance RMSNorm implementation using Triton.",
  "author": "gemini-2.5-pro-mystery-agent",
  "spec": {
    "language": "triton",
    "target_hardware": ["NVIDIA_H100", "NVIDIA_A100", "B200"],
    "dependencies": [
      "triton >= 2.3",
      "torch"
    ],
    "entry_point": "main.py::run",
    "destination_passing_style": false
  },
  "sources": [
    {
      "path": "main.py",
      "content": "import torch\nimport triton\nimport triton.language as tl\n\n@triton.jit\ndef _rmsnorm_kernel(x_ptr, weight_ptr, output_ptr,\n                    x_row_stride, output_row_stride,\n                    n_cols, eps, \n                    BLOCK_SIZE: tl.constexpr):\n    # Get the row index for this program instance\n    row_idx = tl.program_id(0)\n\n    # Create pointers to the beginning of the current row\n    row_x_ptr = x_ptr + row_idx * x_row_stride\n    row_output_ptr = output_ptr + row_idx * output_row_stride\n\n    # --- Pass 1: Calculate mean of squares ---\n    var_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)\n    for off in range(0, n_cols, BLOCK_SIZE):\n        cols = off + tl.arange(0, BLOCK_SIZE)\n        mask = cols < n_cols\n        # Load input data, converting to float32 for accumulation\n        x = tl.load(row_x_ptr + cols, mask=mask, other=0.0).to(tl.float32)\n        var_acc += x * x\n    \n    # Reduce the block-level accumulators to a single scalar value for the row variance\n    row_var = tl.sum(var_acc, axis=0) / n_cols\n    rstd = tl.rsqrt(row_var + eps)\n\n    # --- Pass 2: Normalize and apply weight ---\n    for off in range(0, n_cols, BLOCK_SIZE):\n        cols = off + tl.arange(0, BLOCK_SIZE)\n        mask = cols < n_cols\n\n        # Load input and weight\n        x = tl.load(row_x_ptr + cols, mask=mask, other=0.0)\n        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)\n\n        # Normalize, apply weight, and store\n        x_normalized = x * rstd\n        output = x_normalized * w\n        tl.store(row_output_ptr + cols, output, mask=mask)\n\ndef run(input: torch.Tensor, weight: torch.Tensor, eps: float):\n    \"\"\"\n    Launcher function for the RMSNorm Triton kernel.\n\n    Args:\n        input (torch.Tensor): The input tensor of shape (batch_size, hidden_size).\n        weight (torch.Tensor): The weight tensor of shape (hidden_size).\n        eps (float): A small value to prevent division by zero.\n\n    Returns:\n        dict: A dictionary containing the output tensor under the key 'output'.\n    \"\"\"\n    # Ensure input tensor is contiguous in the last dimension\n    input = input.contiguous()\n    n_rows, n_cols = input.shape\n\n    # Create the output tensor\n    output = torch.empty_like(input)\n\n    # Pick a block size. 1024 is a good default for typical hidden sizes.\n    BLOCK_SIZE = 1024\n\n    # Define the grid for launching the kernel\n    # One program instance per row\n    grid = (n_rows,)\n\n    # Launch the kernel\n    _rmsnorm_kernel[grid](input, weight, output,\n                         input.stride(0), output.stride(0),\n                         n_cols, eps, \n                         BLOCK_SIZE=BLOCK_SIZE)\n    \n    return output"
    }
  ]
}
```

### Example: CUDA Implementation with Custom Compile Options

```json
{
  "name": "flash_attention_cuda_h100_v1",
  "definition": "flash_attention",
  "description": "Optimized Flash Attention implementation using CUDA with custom compiler flags for H100.",
  "author": "cuda-optimization-agent",
  "spec": {
    "language": "cuda",
    "target_hardware": ["NVIDIA_H100"],
    "dependencies": [
      "CUTLASS_3_7"
    ],
    "entry_point": "flash_attention.cu::flash_attention_forward",
    "binding": "torch",
    "destination_passing_style": true,
    "compile_options": {
      "cflags": ["-O3", "-march=native", "-ffast-math"],
      "cuda_cflags": ["-use_fast_math", "--maxrregcount=128", "-arch=sm_90"],
      "ld_flags": ["-lcublas", "-lcudart"]
    }
  },
  "sources": [
    {
      "path": "flash_attention.cu",
      "content": "// CUDA kernel implementation for Flash Attention\n#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n\nextern \"C\" void flash_attention_forward(\n    const half* Q, const half* K, const half* V,\n    half* O, float* L,\n    int batch_size, int num_heads, int seq_len, int head_dim\n) {\n    // Optimized Flash Attention kernel\n    // Uses shared memory tiling and online softmax\n    // ...\n}\n"
    }
  ]
}
```
