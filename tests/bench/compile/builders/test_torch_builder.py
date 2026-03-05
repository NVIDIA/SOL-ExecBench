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

"""Tests for TorchBuilder.

CPU/C++ tests run unconditionally (only require a C++ compiler).
CUDA-specific tests are gated with @pytest.mark.requires_torch_cuda and skipped
automatically when CUDA is unavailable.
"""

import sys
from pathlib import Path

import pytest
import torch

from sol_execbench.bench.compile.builder import BuildError
from sol_execbench.bench.compile.builders.torch_builder import TorchBuilder
from sol_execbench.data import (
    AxisConst,
    BuildSpec,
    Definition,
    Solution,
    SourceFile,
    SupportedBindings,
    SupportedHardware,
    SupportedLanguages,
    TensorSpec,
)


@pytest.fixture(autouse=True)
def _use_tmp_cache_dir(tmp_cache_dir: Path) -> None:
    """Automatically use tmp_cache_dir for all tests in this module."""


# ──────────────────────────────────────────────────────────────────────────────
# Availability and can_build
# ──────────────────────────────────────────────────────────────────────────────


def test_is_available():
    assert TorchBuilder.is_available() == torch.cuda.is_available()


def test_can_build_cuda_solution():
    solution = Solution(
        name="cuda_sol",
        definition="op",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.CUDA,
            target_hardware=[SupportedHardware.LOCAL],
            entry_point="kernel.cu::run",
            destination_passing_style=True,
            binding=SupportedBindings.TORCH,
        ),
        sources=[SourceFile(path="kernel.cu", content="__global__ void run() {}")],
    )
    assert TorchBuilder().can_build(solution) is True


def test_can_build_cpp_solution():
    solution = Solution(
        name="cpp_sol",
        definition="op",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.CPP,
            target_hardware=[SupportedHardware.LOCAL],
            entry_point="kernel.cpp::run",
            destination_passing_style=True,
            binding=SupportedBindings.TORCH,
        ),
        sources=[SourceFile(path="kernel.cpp", content="void run() {}")],
    )
    assert TorchBuilder().can_build(solution) is True


def test_cannot_build_python_solution():
    solution = Solution(
        name="py_sol",
        definition="op",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=[SupportedHardware.LOCAL],
            entry_point="main.py::run",
            destination_passing_style=True,
        ),
        sources=[SourceFile(path="main.py", content="def run(x, y): pass")],
    )
    assert TorchBuilder().can_build(solution) is False


def test_cannot_build_cuda_without_binding():
    solution = Solution(
        name="cuda_no_binding",
        definition="op",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.CUDA,
            target_hardware=[SupportedHardware.LOCAL],
            entry_point="kernel.cu::run",
            destination_passing_style=True,
        ),
        sources=[SourceFile(path="kernel.cu", content="__global__ void run() {}")],
    )
    assert TorchBuilder().can_build(solution) is False


# ──────────────────────────────────────────────────────────────────────────────
# Error handling (no GPU required)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.cpp
def test_raises_invalid_entry_extension():
    """BuildError raised immediately when entry file is not a C/C++/CUDA file."""
    definition = Definition(
        name="op",
        op_type="op",
        axes={"N": AxisConst(value=4)},
        inputs={"x": TensorSpec(shape=["N"], dtype="float32")},
        outputs={"y": TensorSpec(shape=["N"], dtype="float32")},
        reference="def run(x):\n    return x",
    )
    solution = Solution(
        name="bad_extension",
        definition="op",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.CUDA,
            target_hardware=[SupportedHardware.LOCAL],
            entry_point="main.py::run",
            destination_passing_style=False,
            binding=SupportedBindings.TORCH,
        ),
        sources=[SourceFile(path="main.py", content="def run(x): return x")],
    )
    with pytest.raises(BuildError, match="Entry file type not recognized"):
        TorchBuilder().build(definition, solution)


# ──────────────────────────────────────────────────────────────────────────────
# CPU C++ — no CUDA required
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.cpp
def test_cpu_echo():
    """Build a C++ echo extension and execute it on CPU."""
    definition = Definition(
        name="cpu_echo",
        op_type="op",
        axes={"N": AxisConst(value=4)},
        inputs={"x": TensorSpec(shape=["N"], dtype="float32")},
        outputs={"y": TensorSpec(shape=["N"], dtype="float32")},
        reference="def run(x):\n    return x",
    )

    cpp_source = r"""
#include <torch/extension.h>
torch::Tensor echo(torch::Tensor x) { return x; }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("echo", &echo); }
"""

    solution = Solution(
        name="cpu_echo_sol",
        definition="cpu_echo",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.CPP,
            target_hardware=[SupportedHardware.LOCAL],
            entry_point="bind.cpp::echo",
            destination_passing_style=False,
            binding=SupportedBindings.TORCH,
        ),
        sources=[SourceFile(path="bind.cpp", content=cpp_source)],
    )

    runnable = TorchBuilder().build(definition, solution)
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = runnable(x)
    assert torch.allclose(y, x)


@pytest.mark.cpp
def test_cpu_add():
    """Build a C++ element-wise add extension and execute it on CPU."""
    definition = Definition(
        name="cpu_add",
        op_type="op",
        axes={"N": AxisConst(value=3)},
        inputs={
            "a": TensorSpec(shape=["N"], dtype="float32"),
            "b": TensorSpec(shape=["N"], dtype="float32"),
        },
        outputs={"c": TensorSpec(shape=["N"], dtype="float32")},
        reference="def run(a, b):\n    return a + b",
    )

    cpp_source = r"""
#include <torch/extension.h>
torch::Tensor add(torch::Tensor a, torch::Tensor b) { return a + b; }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("add", &add); }
"""

    solution = Solution(
        name="cpu_add_sol",
        definition="cpu_add",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.CPP,
            target_hardware=[SupportedHardware.LOCAL],
            entry_point="bind.cpp::add",
            destination_passing_style=False,
            binding=SupportedBindings.TORCH,
        ),
        sources=[SourceFile(path="bind.cpp", content=cpp_source)],
    )

    runnable = TorchBuilder().build(definition, solution)
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    c = runnable(a, b)
    assert torch.allclose(c, a + b)


# ──────────────────────────────────────────────────────────────────────────────
# CUDA — requires GPU
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.cpp
@pytest.mark.requires_torch_cuda
def test_cuda_vector_add():
    """Build and run a CUDA vector addition kernel."""
    definition = Definition(
        name="vec_add_cuda",
        op_type="op",
        axes={"N": AxisConst(value=256)},
        inputs={
            "x": TensorSpec(shape=["N"], dtype="float32"),
            "y": TensorSpec(shape=["N"], dtype="float32"),
        },
        outputs={"z": TensorSpec(shape=["N"], dtype="float32")},
        reference="import torch\n\ndef run(x, y):\n    return x + y",
    )

    cuda_kernel = r"""
#include <cuda_runtime.h>

extern "C" __global__ void vec_add_kernel(const float* x, const float* y, float* z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = x[i] + y[i];
}

extern "C" void launch_vec_add(const float* x, const float* y, float* z, int n) {
    int threads = 128;
    int blocks = (n + threads - 1) / threads;
    vec_add_kernel<<<blocks, threads>>>(x, y, z, n);
}
"""

    binding_cpp = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" void launch_vec_add(const float* x, const float* y, float* z, int n);

torch::Tensor vec_add(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "Inputs must be contiguous");
    auto z = torch::empty_like(x);
    int n = x.numel();
    launch_vec_add(x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), n);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");
    return z;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("vec_add", &vec_add); }
"""

    solution = Solution(
        name="cuda_vec_add",
        definition="vec_add_cuda",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.CUDA,
            target_hardware=[SupportedHardware.LOCAL],
            entry_point="binding.cpp::vec_add",
            destination_passing_style=False,
            binding=SupportedBindings.TORCH,
        ),
        sources=[
            SourceFile(path="kernel.cu", content=cuda_kernel),
            SourceFile(path="binding.cpp", content=binding_cpp),
        ],
    )

    runnable = TorchBuilder().build(definition, solution)
    x = torch.arange(256, dtype=torch.float32, device="cuda")
    y = 2 * torch.ones(256, dtype=torch.float32, device="cuda")
    z = runnable(x, y)
    assert torch.allclose(z, x + y)


if __name__ == "__main__":
    pytest.main(sys.argv)
