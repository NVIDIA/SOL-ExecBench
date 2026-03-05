"""CuTe DSL kernel functions for test_cutedsl.py.

Kept in a separate non-test module so pytest's assertion rewriter does not
interfere with inspect.getsourcelines(), which CUTLASS DSL needs to read
the decorated function source at compile time.
"""

import cutlass
import cutlass.cute as cute


@cute.kernel
def _add_kernel(gA, gB, gC, cC, shape, thr_layout, val_layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    blk_coord = ((None, None), bidx)
    blkA, blkB, blkC = gA[blk_coord], gB[blk_coord], gC[blk_coord]
    blkCrd = cC[blk_coord]

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    tiled_cpy_A = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    tiled_cpy_B = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    tiled_cpy_C = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

    thr_A = tiled_cpy_A.get_slice(tidx)
    thr_B = tiled_cpy_B.get_slice(tidx)
    thr_C = tiled_cpy_C.get_slice(tidx)

    thrA = thr_A.partition_S(blkA)
    thrB = thr_B.partition_S(blkB)
    thrC = thr_C.partition_S(blkC)
    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgC = cute.make_fragment_like(thrC)

    thrCrd = thr_C.partition_S(blkCrd)
    frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)
    for i in range(0, cute.size(frgPred), 1):
        frgPred[i] = cute.elem_less(thrCrd[i], shape)

    cute.copy(copy_atom, thrA, frgA, pred=frgPred)
    cute.copy(copy_atom, thrB, frgB, pred=frgPred)
    frgC.store(frgA.load() + frgB.load())
    cute.copy(copy_atom, frgC, thrC, pred=frgPred)


@cute.jit
def _elementwise_add_2d(mA, mB, mC, copy_bits: cutlass.Constexpr = 128):
    dtype = mA.element_type
    vector_size = copy_bits // dtype.width  # 8 for bf16
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)
    cC = cute.zipped_divide(cute.make_identity_tensor(mC.shape), tiler=tiler_mn)

    _add_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )
