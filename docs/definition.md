# Definition

## Overview

This document describes the JSON schema for a kernel **Definition**.

The `Definition` provides a formal, machine-readable specification for a computational workload found in a model's forward pass. It is designed to be the single source of truth that guides both human and agent-based kernel development. Specifically, this schema defines:

1. **Tensor Formats**: The shape, data type (`dtype`).
2. **Dimension Semantics**: The distinction between `constant` dimensions (fixed at compile time) and `variable` dimensions (determined at runtime).
3. **Computational Logic**: A clear, step-by-step **reference implementation** in plain PyTorch, which serves as the official mathematical specification of the kernel.

Note that a `Definition` does not contain specific input *data* for its variable axes. That data is provided by the `workload` field of each `Trace`, which is used for benchmarking `Solution` s.

## JSON Schema Description

### Top-Level Object Structure

| Field         | Type                                      | Required | Description                                                                                                                                                                                                       |
| ------------- | ----------------------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`        | string                                    | Yes      | A unique, human-readable name for the kernel, should include concrete problem information. Naming convention: `{op_type}_{props}_{constants}` (e.g. `gqa_paged_decode_h32_kv8_d128_ps1`).                         |
| `description` | string (Default: `null`)                  | No       | A brief, human-readable description of the definition and its purpose.                                                                                                                                            |
| `axes`        | Dict\[string, Union\[AxisConst, AxisVar, AxisExpr]] | Yes      | An object mapping symbolic dimension names (e.g., `"M"`, `"N"`, `"K"`) to their definitions. The value is a constant, variable, or expression axis. The axes will be bound to the input tensor dimensions at runtime. |
| `inputs`      | Dict\[string, TensorSpec]                 | Yes      | Named input tensors (e.g.,`"A"`,`"B"`).                                                                                                                                                                           |
| `outputs`     | Dict\[string, TensorSpec]                 | Yes      | Named output tensors (e.g.,`"C"`).                                                                                                                                                                                |
| `reference`   | string                                    | Yes      | The reference implementation in PyTorch, serving as the mathematical specification.                                                                                                                               |
| `custom_inputs_entrypoint` | string (Default: `null`)     | No       | Entrypoint function name to generate custom inputs. The signature should be `entrypoint(axes_and_scalars: dict[str, int], device: torch.device) -> dict[str, torch.Tensor]`. Required when workloads use `custom` input type. |
| `constraints` | array\[string] (Default: `null`)          | No       | An optional list of assertions describing relationships between axes.                                                                                                                                             |

### `axes` : Dimension Definitions

The `axes` object contains any number of keys, where each key is a symbolic dimension name (e.g., `"M"`, `"N"`, `"K"`), and the value is an object describing its type.

### `type`: `const`

Represents a constant dimension.

| Field         | Type    | Required | Description                |
| ------------- | ------- | -------- | -------------------------- |
| `type`        | string  | Yes      | Must be `"const"`          |
| `value`       | integer | Yes      | Constant value of the axis |
| `description` | string  | No       | Brief description.         |

Example:

```json  theme={null}
"hidden_size": {
  "type": "const",
  "value": 4096
}

```

### `type`: `var`

Represents a variable axis whose value will be determined by the input data.

| Field         | Type   | Required | Description       | Default |
| ------------- | ------ | -------- | ----------------- | ------- |
| `type`        | string | Yes      | Must be `"var"`   | —       |
| `description` | string | No       | Brief description |         |

Example:

```json  theme={null}
"sequence_length": {
  "type": "var",
}

```

### `type`: `expr`

Represents an expression axis whose value is computed from a mathematical expression referencing other axes. Supported operators are `+`, `-`, `*`, `/`, `//`, `%`, `**`, parentheses, and unary `+`/`-`. The expression is defined once in `axes` and the resulting axis name can be referenced in tensor shapes like any other axis. Note that mathematical expressions are not allowed directly in tensor `shape` arrays; define an `expr` axis and reference it by name instead.

| Field         | Type   | Required | Description                                                                    | Default |
| ------------- | ------ | -------- | ------------------------------------------------------------------------------ | ------- |
| `type`        | string | Yes      | Must be `"expr"`                                                               | —       |
| `expression`  | string | Yes      | A mathematical expression referencing other `const` or `var` axis names.       | —       |
| `description` | string | No       | Brief description                                                              |         |

Example:

```json  theme={null}
"head_dim": {
  "type": "expr",
  "expression": "hidden_size // num_heads",
  "description": "Per-head dimension"
}

```

### `inputs`, `outputs` : Tensor Definitions

These fields describe the input and output tensors of the kernel. They contain any number of key-value pairs, where each key is the name of a tensor (e.g., `"A"`, `"B"`, `"C"`). The value is a tensor description:

| Field         | Type            | Required | Description                                                  |
| ------------- | --------------- | -------- | ------------------------------------------------------------ |
| `shape`       | array or `null` | Yes      | List of axis names (strings). Represents a scalar if `null`. |
| `dtype`       | string          | Yes      | Data type of the tensor                                      |
| `description` | string          | No       | Brief description.                                           |

### `dtype` : Data Types

The following values are allowed for `dtype`:

* `float64`
* `float32`
* `float16`
* `bfloat16`
* `float8_e4m3fn`
* `float8_e5m2`
* `float4_e2m1`
* `float4_e2m1fn_x2`
* `int64`
* `int32`
* `int16`
* `int8`
* `bool`

### Scalar Values and 0-D Tensors

Specifically, a tensor with a shape `[]` (empty array) represents a 0-D tensor.

To represent a scalar value, we use shape `null`. The scalar input must receive a python scalar data (int, float, bool). The scalar output will return a python scalar value.

Example:

```json  theme={null}
"inputs": {
  "logits": {
    "shape": ["batch_size", "vocab_size"],
    "dtype": "float16"
  },
  "temperature": {
    "shape": null,
    "dtype": "float16"
  }
},
"outputs": {
  "probs": {
    "shape": ["batch_size", "vocab_size"],
    "dtype": "float16"
  }
}

```

### `reference` : Reference Implementation

The `reference` field is a string that contains the reference implementation of the kernel in plain PyTorch.

* It must contain a global function named `run` as the entry point.
* This code defines the **official mathematical specification** of the kernel.
* It should avoid high-level packagings (e.g., **`torch.nn.functional`**) in favor of explicit, step-by-step computations to ensure maximum clarity for all consumers (human or agent).

## Examples

### Example 1: Standard GEMM

```json  theme={null}
{
  "name": "gemm_n_4096_k_4096",
  "description": "General matrix multiply (GEMM) C = A @ B.T.",
  "op_type": "gemm",
  "tags": [
    "status:verified",
    "model:llama-3.1-8b"
  ],
  "axes": {
    "M": { "type": "var" },
    "N": { "type": "const", "value": 4096 },
    "K": { "type": "const", "value": 4096 }
  },
  "inputs": {
    "A": { "shape": ["M", "K"], "dtype": "float16" },
    "B": { "shape": ["N", "K"], "dtype": "float16" }
  },
  "outputs": {
    "C": { "shape": ["M", "N"], "dtype": "float16" }
  },
  "reference": "import torch\n\ndef run(A, B):\n    C = torch.matmul(A, B.T)\n    return C"
}

```

### Example 2: Quantized GEMM

```json  theme={null}
{
  "name": "quantized_gemm_n4096_k4096_ng128_kg128",
  "description": "A GEMM operation with per-tensor quantized inputs and per-group scaling factors.",
  "op_type": "gemm",
  "tags": [
      "status:draft",
      "model:some_model",
    "quantization:float8_e4m3fn"
    ]
  "axes": {
    "M": { "type": "var" },
    "N": { "type": "const", "value": 4096 },
    "K": { "type": "const", "value": 4096 },
    "N_group": { "type": "const", "value": 128 },
    "K_group": { "type": "const", "value": 128 }
  },
  "inputs": {
    "A": {
      "shape": ["M", "K"],
      "dtype": "float8_e4m3fn"
    },
    "B": {
      "shape": ["N", "K"],
      "dtype": "float8_e4m3fn"
    },
    "A_scale": {
      "shape": ["M", "K_group"],
      "dtype": "float32"
    },
    "B_scale": {
      "shape": ["N_group", "K_group"],
      "dtype": "float32"
    }
  },
  "outputs": {
    "C": {
      "shape": ["M", "N"],
      "dtype": "bfloat16"
    }
  },
  "reference": "..."
}
```

### Example 3: Grouped GEMM

```json  theme={null}
{
  "name": "grouped_gemm_n4096_k4096",
  "description": "A batch of independent GEMM operations, grouped along a 'G' dimension.",
  "op_type": "grouped_gemm",
  "tags": [
    "status:draft",
    "model:some_model"
  ]
  "axes": {
    "G": { "type": "var" },
    "M": { "type": "var" },
    "N": { "type": "const", "value": 4096 },
    "K": { "type": "const", "value": 4096 }
  },
  "inputs": {
    "A": {
      "shape": ["G", "M", "K"],
      "dtype": "float16"
    },
    "B": {
      "shape": ["G", "K", "N"],
      "dtype": "float16"
    }
  },
  "outputs": {
    "C": {
      "shape": ["G", "M", "N"],
      "dtype": "float16"
    }
  },
  "reference": "...",
}
```

### Example 4: Quantized Grouped GEMM

```json  theme={null}
{
  "name": "quantized_grouped_gemm_n4096_k4096_kg128",
  "description": "A batched GEMM operation where the inputs are quantized, with per-group scaling factors.",
  "op_type": "grouped_gemm",
  "tags": [
    "status:draft",
    "quantization:float8_e4m3fn",
    "model:some_model"
  ]
  "axes": {
    "G": { "type": "var" },
    "M": { "type": "var" },
    "N": { "type": "const", "value": 4096 },
    "K": { "type": "const", "value": 4096 },
    "K_group": { "type": "const", "value": 128 }
  },
  "inputs": {
    "A": {
      "shape": ["G", "M", "K"],
      "dtype": "float8_e4m3fn"
    },
    "B": {
      "shape": ["G", "K", "N"],
      "dtype": "float8_e4m3fn"
    },
    "A_scale": {
      "shape": ["G", "M", "K_group"],
      "dtype": "float32"
    },
    "B_scale": {
      "shape": ["G", "K_group", "N"],
      "dtype": "float32"
    }
  },
  "outputs": {
    "C": {
      "shape": ["G", "M", "N"],
      "dtype": "bfloat16"
    }
  },
  "reference": "..."
}
```

### Example 5: RMSNorm

```json  theme={null}
{
  "name": "rmsnorm_d4096",
  "description": "Root Mean Square Normalization, a common layer normalization variant.",
  "type": "norm",
  "tags": [
    "status:draft",
    "model:some_model"
  ],
  "axes": {
    "batch_size": { "type": "var" },
    "hidden_size": { "type": "const", "value": 4096 }
  },
  "inputs": {
    "input": {
      "shape": ["batch_size", "hidden_size"],
      "dtype": "float16"
    },
    "weight": {
      "shape": ["hidden_size"],
      "dtype": "float16"
    },
    "eps": {
      "shape": null,
      "dtype": "float32"
    }
  },
  "outputs": {
    "output": {
      "shape": ["batch_size", "hidden_size"],
      "dtype": "float16"
    }
  },
  "reference": "import torch\n\ndef run(input, weight, eps):\n    variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)\n    rstd = torch.rsqrt(variance + eps)\n    hidden_states = input * rstd\n    output = (hidden_states * weight).to(weight.dtype)\n    return output",
}
```

### Example 6: Attention (GQA-4)

```json  theme={null}
{
  "name": "gqa_hr4_dqk128_dvo128",
  "description": "Grouped-Query Attention with a query-to-key-value head ratio of 4.",
  "op_type": "gqa",
  "tags": [
    "status:draft",
    "model:some_model"
  ]
  "axes": {
    "B": { "type": "var" },
    "Q": { "type": "var" },
    "KV": { "type": "var" },
    "H_qo": { "type": "var" },
    "H_kv": { "type": "var" },
    "H_r": { "type": "const", "value": 4 },
    "D_qk": { "type": "const", "value": 128 },
    "D_vo": { "type": "const", "value": 128 }
  },
  "constraints": [
    "H_qo == H_kv * H_r"
  ],
  "inputs": {
    "q": {
      "shape": ["B", "Q", "H_qo", "D_qk"],
      "dtype": "float16"
    },
    "k": {
      "shape": ["B", "KV", "H_kv", "D_qk"],
      "dtype": "float16"
    },
    "v": {
      "shape": ["B", "KV", "H_kv", "D_vo"],
      "dtype": "float16"
    }
  },
  "outputs": {
    "out": {
      "shape": ["B", "Q", "H_qo", "D_vo"],
      "dtype": "float16"
    },
    "lse": {
      "shape": ["B", "Q", "H_qo"],
      "dtype": "float32"
    }
  },
  "reference": "...",
}
```

### Example 7: Definition with Expression Axis

```json  theme={null}
{
  "name": "split_heads_h32_d4096",
  "description": "Reshape hidden states into multi-head format.",
  "op_type": "reshape",
  "axes": {
    "batch_size": { "type": "var" },
    "seq_len": { "type": "var" },
    "hidden_size": { "type": "const", "value": 4096 },
    "num_heads": { "type": "const", "value": 32 },
    "head_dim": { "type": "expr", "expression": "hidden_size // num_heads", "description": "Per-head dimension" }
  },
  "inputs": {
    "x": { "shape": ["batch_size", "seq_len", "hidden_size"], "dtype": "float16" }
  },
  "outputs": {
    "y": { "shape": ["batch_size", "num_heads", "seq_len", "head_dim"], "dtype": "float16" }
  },
  "reference": "import torch\n\ndef run(x):\n    B, S, _ = x.shape\n    return (x.view(B, S, 32, 128).transpose(1, 2),)"
}
```

### Example 8: Definition with Custom Inputs Entrypoint

When inputs require special generation logic (e.g., ragged tensors, structured sparsity), use `custom_inputs_entrypoint` to specify a function that generates all inputs. The corresponding workloads must use `"type": "custom"` for all inputs.

```json  theme={null}
{
  "name": "ragged_attention_d4096",
  "description": "Attention over variable-length sequences packed into a single tensor.",
  "op_type": "gqa_ragged",
  "custom_inputs_entrypoint": "generate_ragged_inputs",
  "axes": {
    "total_tokens": { "type": "var" },
    "hidden_size": { "type": "const", "value": 4096 }
  },
  "inputs": {
    "q": { "shape": ["total_tokens", "hidden_size"], "dtype": "float16" },
    "k": { "shape": ["total_tokens", "hidden_size"], "dtype": "float16" },
    "seq_offsets": { "shape": ["total_tokens"], "dtype": "int32" }
  },
  "outputs": {
    "out": { "shape": ["total_tokens", "hidden_size"], "dtype": "float16" }
  },
  "reference": "..."
}
```
