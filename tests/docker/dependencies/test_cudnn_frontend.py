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

"""Smoke test for cuDNN Frontend — RMSNorm via cudnn.graph()."""

import torch
import cudnn


def test_cudnn_rmsnorm():
    """RMSNorm forward pass using cuDNN Frontend graph API."""
    B, H = 4, 256
    eps = 1e-5

    x = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
    scale = torch.ones(H, device="cuda", dtype=torch.bfloat16)

    # Build cuDNN graph
    graph = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT)

    x_t = graph.tensor(
        name="x", dim=[B, H], stride=[H, 1], data_type=cudnn.data_type.BFLOAT16
    )
    scale_t = graph.tensor(
        name="scale", dim=[1, H], stride=[H, 1], data_type=cudnn.data_type.BFLOAT16
    )
    epsilon_t = graph.tensor(
        name="epsilon",
        dim=[1, 1],
        stride=[1, 1],
        data_type=cudnn.data_type.FLOAT,
        is_pass_by_value=True,
    )

    y_t, inv_var_t = graph.rmsnorm(
        name="rmsnorm",
        norm_forward_phase=cudnn.norm_forward_phase.INFERENCE,
        input=x_t,
        scale=scale_t,
        epsilon=epsilon_t,
    )
    y_t.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

    graph.build([cudnn.heur_mode.A])

    # Allocate outputs and workspace
    y = torch.empty_like(x)
    epsilon = torch.full((1, 1), eps, dtype=torch.float32)
    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )

    # Execute
    graph.execute(
        {x_t: x, scale_t: scale, epsilon_t: epsilon, y_t: y},
        workspace,
    )

    # Reference in PyTorch
    x_f32 = x.float()
    rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    ref = (x_f32 * rms * scale.float()).to(torch.bfloat16)

    torch.testing.assert_close(y, ref, atol=1e-2, rtol=1e-2)
    print("✓ cuDNN Frontend RMSNorm passed!")


if __name__ == "__main__":
    test_cudnn_rmsnorm()
