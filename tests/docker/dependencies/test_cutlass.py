"""Smoke test: compile and link a CUDA program against CUTLASS headers."""

import pytest
import subprocess
import tempfile
import os

CUTLASS_SRC = r"""
#include <cutlass/version.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cstdio>

int main() {
    // Verify CUTLASS version macros are accessible
    printf("CUTLASS version: %d\n", CUTLASS_MAJOR);

    // Verify CUTLASS numeric types compile
    cutlass::half_t h(1.5f);
    cutlass::bfloat16_t bf(2.5f);
    float fh = float(h);
    float fbf = float(bf);
    printf("half_t(1.5) = %.1f, bfloat16_t(2.5) = %.1f\n", fh, fbf);

    // Verify CUTLASS GEMM type alias compiles
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,               // A element type
        cutlass::layout::RowMajor,     // A layout
        cutlass::half_t,               // B element type
        cutlass::layout::RowMajor,     // B layout
        cutlass::half_t,               // C element type
        cutlass::layout::RowMajor      // C layout
    >;
    printf("GEMM type size: %zu\n", sizeof(Gemm));

    printf("PASS\n");
    return 0;
}
"""


@pytest.mark.cpp
def test_cutlass_compile_and_link():
    """Compile a .cu file with CUTLASS headers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "test.cu")
        exe = os.path.join(tmpdir, "test")

        with open(src, "w") as f:
            f.write(CUTLASS_SRC)

        result = subprocess.run(
            [
                "nvcc",
                src,
                "-o",
                exe,
                "-std=c++17",
                "-I/usr/include",
                "-I/usr/local/cuda/include",
                "-L/usr/local/cuda/lib64",
                "--expt-relaxed-constexpr",
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
        print("✓ CUTLASS compile+link+run passed!")


if __name__ == "__main__":
    test_cutlass_compile_and_link()
