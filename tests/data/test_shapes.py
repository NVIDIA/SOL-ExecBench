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

import sys

import pytest

from sol_execbench.data.shapes import resolve_shape_expression


# --- Basic literals ---


def test_integer_literal():
    assert resolve_shape_expression("42", {}) == 42


def test_zero():
    assert resolve_shape_expression("0", {}) == 0


# --- Variables ---


def test_single_variable():
    assert resolve_shape_expression("x", {"x": 8}) == 8


def test_multiple_variables():
    assert resolve_shape_expression("a + b", {"a": 3, "b": 5}) == 8


# --- Arithmetic operators ---


def test_addition():
    assert resolve_shape_expression("2 + 3", {}) == 5


def test_subtraction():
    assert resolve_shape_expression("10 - 4", {}) == 6


def test_multiplication():
    assert resolve_shape_expression("3 * 4", {}) == 12


def test_floor_division():
    assert resolve_shape_expression("7 // 2", {}) == 3


def test_modulo():
    assert resolve_shape_expression("7 % 3", {}) == 1


def test_power():
    assert resolve_shape_expression("2 ** 8", {}) == 256


# --- Unary operators ---


def test_unary_minus():
    assert resolve_shape_expression("-(-4)", {}) == 4


def test_unary_plus():
    assert resolve_shape_expression("+5", {}) == 5


# --- Compound expressions ---


def test_complex_expression():
    assert (
        resolve_shape_expression(
            "hidden_size // num_heads * 2", {"hidden_size": 512, "num_heads": 8}
        )
        == 128
    )


def test_nested_parens():
    assert resolve_shape_expression("(a + b) * c", {"a": 2, "b": 3, "c": 4}) == 20


def test_typical_shape_formula():
    # head_dim = hidden_size // num_heads
    assert (
        resolve_shape_expression(
            "hidden_size // num_heads", {"hidden_size": 4096, "num_heads": 32}
        )
        == 128
    )


# --- Error: non-integer result ---


def test_true_division_raises():
    with pytest.raises(TypeError, match="must evaluate to an integer"):
        resolve_shape_expression("7 / 2", {})


def test_float_literal_raises():
    with pytest.raises(TypeError, match="must evaluate to an integer"):
        resolve_shape_expression("1.5", {})


def test_float_variable_raises():
    with pytest.raises(TypeError, match="must be int"):
        resolve_shape_expression("x", {"x": 1.5})


# --- Error: unknown variable ---


def test_unknown_variable_raises():
    with pytest.raises(NameError, match="Unknown variable"):
        resolve_shape_expression("z", {})


# --- Error: disallowed node types ---


def test_function_call_raises():
    with pytest.raises(TypeError, match="Unsupported expression node"):
        resolve_shape_expression("len(x)", {"x": 4})


def test_attribute_access_raises():
    with pytest.raises(TypeError, match="Unsupported expression node"):
        resolve_shape_expression("x.shape", {"x": 4})


def test_subscript_raises():
    with pytest.raises(TypeError, match="Unsupported expression node"):
        resolve_shape_expression("x[0]", {"x": 4})


def test_string_literal_raises():
    with pytest.raises(TypeError, match="Unsupported constant type"):
        resolve_shape_expression("'hello'", {})


def test_variable_with_wrong_type_raises():
    with pytest.raises(TypeError, match="must be int"):
        resolve_shape_expression("x", {"x": "big"})


if __name__ == "__main__":
    pytest.main(sys.argv)
