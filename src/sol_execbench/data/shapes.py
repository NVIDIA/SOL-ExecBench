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

import ast
import operator as op

_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

_UNARY_OPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def resolve_shape_expression(expr: str, vars: dict[str, int]) -> int:
    """
    Safely evaluate a simple arithmetic expression with variables.
    Allowed: numbers, variable names, + - * / // % **, parentheses, unary +/-
    Disallowed: function calls, attributes, subscripts, comprehensions, etc.
    """

    def eval_node(node):
        if isinstance(node, ast.Constant):  # numbers like 1, 2.5
            if isinstance(node.value, (int, float)):
                return node.value
            raise TypeError(f"Unsupported constant type: {type(node.value).__name__}")

        if isinstance(node, ast.Name):  # variable like x
            if node.id in vars:
                v = vars[node.id]
                if not isinstance(v, int):
                    raise TypeError(
                        f"Variable '{node.id}' must be int, got {type(v).__name__}"
                    )
                return v
            raise NameError(f"Unknown variable: {node.id}")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _BIN_OPS:
                raise TypeError(f"Unsupported operator: {op_type.__name__}")
            return _BIN_OPS[op_type](eval_node(node.left), eval_node(node.right))

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _UNARY_OPS:
                raise TypeError(f"Unsupported unary operator: {op_type.__name__}")
            return _UNARY_OPS[op_type](eval_node(node.operand))

        # Anything else is not allowed (Call, Attribute, Subscript, etc.)
        raise TypeError(f"Unsupported expression node: {type(node).__name__}")

    tree = ast.parse(expr, mode="eval")
    value = eval_node(tree.body)
    # check if float can be cleanly converted to int
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int):
        raise TypeError(
            f"Expression '{expr}' must evaluate to an integer, got {type(value).__name__}"
        )
    return value
