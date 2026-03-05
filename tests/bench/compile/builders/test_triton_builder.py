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

"""Tests for TritonBuilder.

Tests that only verify builder properties (inheritance, error handling on bad
inputs) run unconditionally.  Tests that actually execute Triton kernels are
gated with @pytest.mark.requires_torch_cuda.
"""

import sys
from pathlib import Path

import pytest
import torch

from sol_execbench.bench.compile.builder import BuildError
from sol_execbench.bench.compile.builders.python_builder import PythonBuilder
from sol_execbench.bench.compile.builders.triton_builder import TritonBuilder
from sol_execbench.data import (
    AxisConst,
    BuildSpec,
    Definition,
    Solution,
    SourceFile,
    SupportedHardware,
    SupportedLanguages,
    TensorSpec,
)


@pytest.fixture(autouse=True)
def _use_tmp_cache_dir(tmp_cache_dir: Path) -> None:
    """Automatically use tmp_cache_dir for all tests in this module."""


# ──────────────────────────────────────────────────────────────────────────────
# Availability, inheritance, and can_build
# ──────────────────────────────────────────────────────────────────────────────


def test_is_subclass_of_python_builder():
    assert issubclass(TritonBuilder, PythonBuilder)


def test_can_build_triton_solution():
    solution = Solution(
        name="triton_sol",
        definition="op",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.TRITON,
            target_hardware=[SupportedHardware.LOCAL],
            entry_point="kernel.py::run",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="kernel.py", content="def run(x): return x")],
    )
    assert TritonBuilder().can_build(solution) is True


def test_cannot_build_python_solution():
    solution = Solution(
        name="py_sol",
        definition="op",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=[SupportedHardware.LOCAL],
            entry_point="main.py::run",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="main.py", content="def run(x): return x")],
    )
    assert TritonBuilder().can_build(solution) is False


# ──────────────────────────────────────────────────────────────────────────────
# Error handling (no GPU required)
# ──────────────────────────────────────────────────────────────────────────────


def test_raises_non_python_entry_file():
    """BuildError raised when entry file is not a .py file."""
    definition = Definition(
        name="op",
        op_type="op",
        axes={"N": AxisConst(value=4)},
        inputs={"x": TensorSpec(shape=["N"], dtype="float32")},
        outputs={"y": TensorSpec(shape=["N"], dtype="float32")},
        reference="def run(x):\n    return x",
    )
    solution = Solution(
        name="bad_ext",
        definition="op",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.TRITON,
            target_hardware=[SupportedHardware.LOCAL],
            entry_point="kernel.cu::run",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="kernel.cu", content="__global__ void run() {}")],
    )
    with pytest.raises(BuildError, match="not a Python file"):
        TritonBuilder().build(definition, solution)


def test_raises_missing_entry_symbol():
    """BuildError raised when the entry symbol is absent from the module."""
    definition = Definition(
        name="op",
        op_type="op",
        axes={"N": AxisConst(value=4)},
        inputs={"x": TensorSpec(shape=["N"], dtype="float32")},
        outputs={"y": TensorSpec(shape=["N"], dtype="float32")},
        reference="def run(x):\n    return x",
    )
    solution = Solution(
        name="missing_sym",
        definition="op",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.TRITON,
            target_hardware=[SupportedHardware.LOCAL],
            entry_point="kernel.py::nonexistent_fn",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="kernel.py", content="def other_fn(): pass")],
    )
    with pytest.raises(BuildError, match="not found"):
        TritonBuilder().build(definition, solution)


def test_build_type_is_triton():
    """TritonBuilder must tag the runnable with build_type='triton', not 'python'."""
    definition = Definition(
        name="identity",
        op_type="op",
        axes={"N": AxisConst(value=4)},
        inputs={"x": TensorSpec(shape=["N"], dtype="float32")},
        outputs={"y": TensorSpec(shape=["N"], dtype="float32")},
        reference="def run(x):\n    return x",
    )
    solution = Solution(
        name="triton_identity",
        definition="identity",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.TRITON,
            target_hardware=[SupportedHardware.LOCAL],
            entry_point="main.py::run",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="main.py", content="def run(x):\n    return x")],
    )
    runnable = TritonBuilder().build(definition, solution)
    try:
        assert runnable.metadata.build_type == "triton"
    finally:
        runnable.cleanup()


# ──────────────────────────────────────────────────────────────────────────────
# GPU execution — requires CUDA + Triton
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.requires_torch_cuda
def test_vector_add():
    """Build and run a Triton vector addition kernel."""
    definition = Definition(
        name="vec_add",
        op_type="op",
        axes={"N": AxisConst(value=256)},
        inputs={
            "x": TensorSpec(shape=["N"], dtype="float32"),
            "y": TensorSpec(shape=["N"], dtype="float32"),
        },
        outputs={"z": TensorSpec(shape=["N"], dtype="float32")},
        reference="import torch\n\ndef run(x, y):\n    return x + y",
    )

    triton_code = """
import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(z_ptr + offs, x + y, mask=mask)

def run(x, y):
    n = x.numel()
    z = torch.empty_like(x)
    BLOCK = 128
    grid = lambda meta: ((n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _add_kernel[grid](x, y, z, n, BLOCK_SIZE=BLOCK)
    return z
"""

    solution = Solution(
        name="triton_vec_add",
        definition="vec_add",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.TRITON,
            target_hardware=[SupportedHardware.LOCAL],
            entry_point="main.py::run",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="main.py", content=triton_code)],
    )

    runnable = TritonBuilder().build(definition, solution)
    x = torch.arange(256, dtype=torch.float32, device="cuda")
    y = 2 * torch.ones(256, dtype=torch.float32, device="cuda")
    z = runnable(x, y)
    assert torch.allclose(z, x + y)


if __name__ == "__main__":
    pytest.main(sys.argv)
