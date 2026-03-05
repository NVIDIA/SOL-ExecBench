"""Smoke test: compile and link a C++ program against -lcudnn."""

import pytest
import subprocess
import tempfile
import os

CUDNN_SRC = r"""
#include <cudnn.h>
#include <cstdio>

int main() {
    // Verify cuDNN version macros are accessible
    int major = CUDNN_MAJOR;
    int minor = CUDNN_MINOR;
    int patch = CUDNN_PATCHLEVEL;
    printf("cuDNN compile-time version: %d.%d.%d\n", major, minor, patch);

    // Verify cuDNN runtime symbol is linkable
    size_t runtime_version = cudnnGetVersion();
    printf("cuDNN runtime version: %zu\n", runtime_version);

    // Verify handle creation symbol is linkable
    cudnnHandle_t handle = nullptr;
    cudnnStatus_t status = cudnnCreate(&handle);
    printf("cudnnCreate status: %d\n", status);
    if (handle) {
        cudnnDestroy(handle);
    }

    printf("PASS\n");
    return 0;
}
"""


@pytest.mark.cpp
def test_cudnn_compile_and_link():
    """Compile a .cpp file and link against -lcudnn."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "test.cpp")
        exe = os.path.join(tmpdir, "test")

        with open(src, "w") as f:
            f.write(CUDNN_SRC)

        result = subprocess.run(
            [
                "g++",
                src,
                "-o",
                exe,
                "-I/usr/local/cuda/include",
                "-I/usr/include",
                "-L/usr/local/cuda/lib64",
                "-L/usr/lib/x86_64-linux-gnu",
                "-lcudnn",
                "-lcudart",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"g++ compile failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert os.path.isfile(exe), "Binary was not produced"

        # Run the binary
        result = subprocess.run([exe], capture_output=True, text=True, timeout=30)
        assert "PASS" in result.stdout, (
            f"Binary did not produce PASS:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        print("✓ cuDNN compile+link+run passed!")


if __name__ == "__main__":
    test_cudnn_compile_and_link()
