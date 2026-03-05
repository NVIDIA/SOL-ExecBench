"""CuTile vector add smoke test.
PYTEST_DONT_REWRITE
"""

import torch
import numpy as np
import cuda.tile as ct


@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    # Get the 1D pid
    pid = ct.bid(0)

    # Load input tiles
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))

    # Perform elementwise addition
    result = a_tile + b_tile

    # Store result
    ct.store(c, index=(pid,), tile=result)


def test_cutile():
    # Create input data
    vector_size = 4096
    tile_size = 128
    grid = (ct.cdiv(vector_size, tile_size), 1, 1)

    a = torch.randn(vector_size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(vector_size, device="cuda", dtype=torch.bfloat16)
    c = torch.zeros_like(a, device="cuda", dtype=torch.bfloat16)

    # Launch kernel
    ct.launch(
        torch.cuda.current_stream(),
        grid,  # 1D grid of processors
        vector_add,
        (a, b, c, tile_size),
    )

    # Copy to host only to compare
    a_np = a.cpu().numpy()
    b_np = b.cpu().numpy()
    c_np = c.cpu().numpy()

    # Verify results
    expected = a_np + b_np
    np.testing.assert_array_almost_equal(c_np, expected)

    print("✓ vector_add_example passed!")


if __name__ == "__main__":
    test_cutile()
