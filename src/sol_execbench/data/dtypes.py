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

"""Common utilities and base classes for data models."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@cache
def _get_dtype_str_to_python_dtype() -> dict[str, type]:
    """Get dtype string to Python type mapping (cached)."""
    return {
        "float64": float,
        "float32": float,
        "float16": float,
        "bfloat16": float,
        "float8_e4m3fn": float,
        "float8_e5m2": float,
        "float4_e2m1": float,
        "float4_e2m1fn_x2": float,
        "int64": int,
        "int32": int,
        "int16": int,
        "int8": int,
        "bool": bool,
    }


def dtype_str_to_python_dtype(dtype_str: str) -> type:
    if not dtype_str:
        raise ValueError("dtype is None or empty")
    dtype = _get_dtype_str_to_python_dtype().get(dtype_str, None)
    if dtype is None:
        raise ValueError(f"Unsupported dtype '{dtype_str}'")
    return dtype


@cache
def _get_dtype_str_to_torch_dtype() -> dict[str, torch.dtype]:
    """Lazily build dtype string to torch dtype mapping."""
    import torch

    return {
        "float64": torch.float64,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
        "float4_e2m1": torch.float4_e2m1fn_x2,
        "float4_e2m1fn_x2": torch.float4_e2m1fn_x2,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "bool": torch.bool,
    }


def dtype_str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    if not dtype_str:
        raise ValueError("dtype is None or empty")
    dtype = _get_dtype_str_to_torch_dtype().get(dtype_str, None)
    if dtype is None:
        raise ValueError(f"Unsupported dtype '{dtype_str}'")
    return dtype


@cache
def _get_integer_dtypes() -> frozenset[torch.dtype]:
    """Get frozenset of integer and boolean dtypes (cached)."""
    import torch

    return frozenset(
        (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
            torch.bool,
        )
    )


def is_dtype_integer(dtype: torch.dtype) -> bool:
    """Check if dtype is an integer or boolean type."""
    return dtype in _get_integer_dtypes()
