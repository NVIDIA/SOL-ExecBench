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

"""Utility functions for kernel evaluation.

This module provides helper functions for allocating output tensors and
normalizing kernel results during evaluation.
"""

from typing import Any

import torch

from ...data import Definition


def allocate_outputs(
    definition: Definition, resolved_axes: dict[str, int], device: str
) -> list[torch.Tensor]:
    """Allocate output tensors based on definition and input shapes.

    Infers variable axis values from input tensor shapes and allocates
    empty output tensors with the correct shapes and dtypes.

    Parameters
    ----------
    definition : Definition
        The kernel definition specifying output tensor specs.
    inputs : List[Any]
        List of input values (tensors or scalars) in definition order.
    device : str
        The device to allocate tensors on (e.g., "cuda:0").

    Returns
    -------
    list[torch.Tensor]
        List of allocated (uninitialized) output tensors in definition order.
    """
    output_shapes = list(definition.get_output_shapes(resolved_axes).values())

    dtypes = definition.torch_output_dtypes
    return [
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in zip(output_shapes, dtypes)
    ]


def normalize_result(
    definition: Definition, result: Any, device: str
) -> list[torch.Tensor]:
    """Normalize a value-returning kernel result to a tensor list.

    Converts various return types (scalar, tensor, tuple, list) to a
    standardized list of tensors matching the definition's output order.

    Parameters
    ----------
    definition : Definition
        The kernel definition specifying expected outputs.
    result : Any
        The kernel return value. Can be:
        - A single value (int, float, bool, or torch.Tensor)
        - A tuple or list of values
    device : str
        The device to place resulting tensors on.

    Returns
    -------
    list[torch.Tensor]
        List of output tensors in definition order.

    Raises
    ------
    ValueError
        If the number of returned values doesn't match the expected outputs.
    """

    if result is None:
        raise ValueError("Solution entrypoint returned None.")

    dtypes = definition.torch_output_dtypes
    n_outputs = len(dtypes)

    def to_tensor(v: Any, dtype: torch.dtype) -> torch.Tensor:
        if v is None:
            raise ValueError("Output tensor cannot be None")
        if isinstance(v, torch.Tensor):
            return v.to(device) if str(v.device) != device else v
        return torch.tensor(v, dtype=dtype, device=device)

    if isinstance(result, dict):
        result = list(result.values())

    if isinstance(result, (tuple, list)):
        if len(result) != n_outputs:
            raise ValueError(
                f"Tuple/list has {len(result)} elements but {n_outputs} outputs expected"
            )
        return [to_tensor(v, dtypes[i]) for i, v in enumerate(result)]

    # Single value: tensor, int, float, bool
    if n_outputs != 1:
        raise ValueError(f"Single value returned but {n_outputs} outputs expected")

    return [to_tensor(result, dtypes[0])]
