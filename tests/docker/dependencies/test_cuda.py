"""Smoke test: compile and link a CUDA C program against -lcuda -lcudart."""

import pytest
import subprocess
import tempfile
import os

CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // Verify CUDA driver API is linkable
    int driver_version = 0;
    CUresult res = cuDriverGetVersion(&driver_version);
    printf("cuDriverGetVersion: %d (result=%d)\n", driver_version, res);

    // Verify CUDA runtime API is linkable
    int runtime_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    printf("cudaRuntimeGetVersion: %d\n", runtime_version);

    // Verify kernel symbol is present (don't launch without a GPU)
    printf("vector_add kernel compiled at %p\n", (void*)vector_add);
    printf("PASS\n");
    return 0;
}
"""


@pytest.mark.cpp
def test_cudac_compile_and_link():
    """Compile a .cu file and link against -lcuda -lcudart."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "test.cu")
        exe = os.path.join(tmpdir, "test")

        with open(src, "w") as f:
            f.write(CUDA_SRC)

        # Compile and link
        result = subprocess.run(
            [
                "nvcc",
                src,
                "-o",
                exe,
                "-lcuda",
                "-I/usr/local/cuda/include",
                "-L/usr/local/cuda/lib64",
                "-L/usr/lib/x86_64-linux-gnu",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"nvcc compile failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert os.path.isfile(exe), "Binary was not produced"

        # Run the binary
        result = subprocess.run([exe], capture_output=True, text=True, timeout=30)
        assert "PASS" in result.stdout, (
            f"Binary did not produce PASS:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        print("✓ CUDA C compile+link+run passed!")


if __name__ == "__main__":
    test_cudac_compile_and_link()
