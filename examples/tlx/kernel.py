# TLX (Triton Low-level Extensions) softmax.
#
# This kernel imports `triton.language.extra.tlx`, which exists only in Meta's
# **fbtriton** fork, not upstream triton. The solution therefore declares
# `fbtriton` in `spec.pip_packages`; SOL-ExecBench installs it before evaluation
# (offline, from the curated wheelhouse) into an isolated dir prepended to
# PYTHONPATH, so `import triton` here resolves to the fork that provides TLX.
#
# Pattern: the canonical single-tile async-load softmax — copy each row into
# shared memory with cp.async (`tlx.async_load`), then reduce in registers.
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def _softmax_tlx_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    # Persistent grid: each CTA strides over multiple rows.
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    # Hoist the SMEM buffer outside the loop (required by TLX scoping rules).
    row_smem = tlx.local_alloc((BLOCK_SIZE,), input_ptr.dtype.element_ty, 1)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets

        # Async-copy the row into shared memory, commit the group, then wait.
        row_buf = tlx.local_view(row_smem, 0)
        token = tlx.async_load(input_ptrs, row_buf, mask=mask, other=-float("inf"))
        tlx.async_load_commit_group([token])
        tlx.async_load_wait_group(0)

        # Pull into registers and do the row-wise softmax in fp32.
        row = tlx.local_load(row_buf).to(tl.float32)
        row = tl.where(mask, row, -float("inf"))
        row = row - tl.max(row, axis=0)
        numerator = tl.exp(row)
        out = numerator / tl.sum(numerator, axis=0)

        output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
        tl.store(output_ptrs, out.to(output_ptr.dtype.element_ty), mask=mask)


def run(x):
    """Return-value style: row-wise softmax using the TLX kernel."""
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_sm = torch.cuda.get_device_properties(x.device).multi_processor_count
    grid = (min(num_sm, n_rows), 1, 1)
    _softmax_tlx_kernel[grid](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=2,
        num_warps=8,
    )
    return y
