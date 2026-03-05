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

import torch


# Need a custom inputs method since some inputs have non-random values
def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Returns the input arguments for the reference forward pass. Required method."""
    batch_size, seq_len, hidden_size = (
        axes_and_scalars["batch_size"],
        axes_and_scalars["seq_len"],
        axes_and_scalars["hidden_size"],
    )
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]

    batch_seq_len = batch_size * seq_len
    num_selected_tokens = batch_size * seq_len * num_experts_per_tok

    # Initialize accumulation buffer with random values (not zeros) to detect no-op
    final_hidden_states = torch.randn(batch_seq_len, hidden_size, device=device)

    # Expert outputs (weighted outputs from expert computation)
    expert_outputs = torch.randn(num_selected_tokens, hidden_size, device=device)

    # Token indices (which token position each expert output belongs to)
    # These should be in range [0, batch_seq_len)
    # Simulate scattered indices with potential duplicates (for top_k > 1)
    token_indices = torch.randint(
        0, batch_seq_len, (num_selected_tokens,), dtype=torch.long, device=device
    )

    return {
        "final_hidden_states": final_hidden_states,
        "expert_outputs": expert_outputs,
        "token_indices": token_indices,
    }


# Always use no_grad to prevent PyTorch autograd from slowing down computation.
# This should always be here even when creating drivers for backward kernels.
@torch.no_grad()
def run(
    final_hidden_states: torch.Tensor,
    expert_outputs: torch.Tensor,
    token_indices: torch.Tensor,
):
    # Critical atomic accumulation operation
    # This performs: final_hidden_states[token_indices[i]] += expert_outputs[i]
    # for all i in parallel with atomic semantics
    final_hidden_states.index_add_(dim=0, index=token_indices, source=expert_outputs)

    return final_hidden_states


if __name__ == "__main__":
    inputs = get_inputs(
        axes={
            "batch_size": 16,
            "seq_len": 512,
            "hidden_size": 4096,
            "num_experts_per_tok": 2,
        },
        device=torch.device("cuda:0"),
    )
    out = run(**inputs)
    print(out.shape)
