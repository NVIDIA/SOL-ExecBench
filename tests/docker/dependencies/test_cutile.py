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

"""CuTile vector add smoke test.
PYTEST_DONT_REWRITE
"""

import pytest
import torch
import numpy as np
import cuda.tile as ct


def _gpu_sm_version() -> int:
    if not torch.cuda.is_available():
        return 0
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


requires_sm100 = pytest.mark.skipif(
    _gpu_sm_version() < 100,
    reason=f"cuTile requires sm_100+ (detected sm_{_gpu_sm_version()})",
)


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


@requires_sm100
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

    # Copy to host only to compare (cast to float32; numpy has no bfloat16)
    a_np = a.cpu().float().numpy()
    b_np = b.cpu().float().numpy()
    c_np = c.cpu().float().numpy()

    # Verify results (bfloat16 has ~7.8e-3 relative precision)
    expected = a_np + b_np
    np.testing.assert_allclose(c_np, expected, rtol=1e-2, atol=1e-2)

    print("✓ vector_add_example passed!")


if __name__ == "__main__":
    test_cutile()
