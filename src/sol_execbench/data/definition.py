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

"""The definition of kernels in the FlashInfer Trace schema."""

from __future__ import annotations

import ast
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Any, Iterable, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from .base_model import BaseModelWithDocstrings, NonEmptyString, NonNegativeInt
from .dtypes import dtype_str_to_torch_dtype
from .shapes import resolve_shape_expression

if TYPE_CHECKING:
    import torch


class AxisConst(BaseModelWithDocstrings):
    """Constant axis with a fixed value.

    A constant axis represents a dimension that has a fixed, compile-time known value.
    This is useful for dimensions that don't vary across different instances of the
    same kernel definition, such as embedding dimensions or hidden layer sizes.
    """

    type: Literal["const"] = "const"
    """The type identifier for constant axes."""
    value: NonNegativeInt
    """The constant integer value of this axis dimension."""
    description: Optional[str] = None
    """An optional human-readable description explaining the purpose of this axis."""


class AxisVar(BaseModel):
    """Variable axis that can be specified at runtime.

    A variable axis represents a dimension whose value is determined at runtime
    based on the actual input data. Its value will be bound to the input tensor
    dimension at runtime.
    """

    type: Literal["var"] = "var"
    """The type identifier for variable axes."""
    description: Optional[str] = Field(default=None)
    """An optional human-readable description explaining the purpose of this axis."""


class AxisExpr(BaseModel):
    """Expression axis that can be specified at runtime.

    An expression axis represents a dimension whose value is determined at runtime
    based on a mathematical expression. Its value will be bound to the input tensor
    dimension at runtime. Supported operators are +, -, *, /, //, %, **, parentheses, and unary +/-.
    """

    type: Literal["expr"] = "expr"
    """The type identifier for expression axes."""
    expression: NonEmptyString
    """The mathematical expression that defines the value of this axis. The names may reference either Constant or Variable axes."""
    description: Optional[str] = Field(default=None)
    """An optional human-readable description explaining the purpose of this axis."""


class DType(str, Enum):
    """Supported data types for tensors.

    Enumeration of all data types that can be used in tensor specifications.
    Includes both floating-point and integer types commonly used in machine
    learning and high-performance computing applications.
    """

    FLOAT64 = "float64"
    """64-bit IEEE 754 floating point."""
    FLOAT32 = "float32"
    """32-bit IEEE 754 floating point."""
    FLOAT16 = "float16"
    """16-bit IEEE 754 half-precision floating point."""
    BFLOAT16 = "bfloat16"
    """16-bit Brain Floating Point format."""
    FLOAT8_E4M3FN = "float8_e4m3fn"
    """8-bit floating point with 4 exponent bits and 3 mantissa bits."""
    FLOAT8_E5M2 = "float8_e5m2"
    """8-bit floating point with 5 exponent bits and 2 mantissa bits."""
    FLOAT4_E2M1 = "float4_e2m1"
    """4-bit floating point with 2 exponent bits and 1 mantissa bit."""
    FLOAT4_E2M1FN_X2 = "float4_e2m1fn_x2"
    """4-bit floating point with 2 exponent bits and 1 mantissa bit, packed into a single byte."""
    INT64 = "int64"
    """64-bit signed integer."""
    INT32 = "int32"
    """32-bit signed integer."""
    INT16 = "int16"
    """16-bit signed integer."""
    INT8 = "int8"
    """8-bit signed integer."""
    BOOL = "bool"
    """Boolean type."""


class TensorSpec(BaseModelWithDocstrings):
    """Specification for a tensor including shape and data type, to use as input or output of a
    kernel.

    This includes the symbolic shape (referencing defined axes) and the data type.
    Scalars are represented with a None shape.
    """

    shape: Optional[list[NonEmptyString]]
    """List of axis names defining the tensor shape. None for scalar values."""
    dtype: DType
    """The data type of all elements in this tensor."""
    description: Optional[str] = None
    """An optional human-readable description of this tensor's purpose and usage."""


AxisSpec = Union[AxisConst, AxisVar, AxisExpr]
"""Union type representing all possible axis specifications."""


class Definition(BaseModelWithDocstrings):
    """Complete definition of a computational workload.

    A Definition provides a formal, machine-readable specification for a computational
    workload. It defines the tensor formats, dimension semantics, and computational
    logic through a reference implementation. This serves as the single source of
    truth for kernel development and optimization.
    """

    name: NonEmptyString
    """A unique, human-readable name for the kernel definition."""
    op_type: Optional[NonEmptyString] = Field(default=None)
    """The general compute category."""
    axes: dict[NonEmptyString, AxisSpec]
    """Dictionary of symbolic dimensions used in tensor shapes. The axes will be bound to the
    input tensor dimensions at runtime."""
    custom_inputs_entrypoint: Optional[NonEmptyString] = Field(default=None)
    """The entrypoint function to generate the inputs. The signature should follow entrypoint(axes_and_scalars: dict[str, int], device: torch.device) -> dict[str, torch.Tensor]"""
    inputs: dict[NonEmptyString, TensorSpec]
    """Named input tensors required by this kernel. The order of inputs is preserved as the
    kernel's interface."""
    outputs: dict[NonEmptyString, TensorSpec]
    """Named output tensors produced by this kernel. The names of the output must not overlap
    with the names of the inputs. The order of outputs is preserved as the kernel's interface."""
    reference: NonEmptyString
    """Reference implementation code. It defines the compute logic of the kernel. Must be a valid
    Python code with a 'run' function that takes the input tensors and returns the output tensors.
    """
    description: Optional[str] = Field(default=None)
    """Optional human-readable description of the kernel's purpose."""
    constraints: list[NonEmptyString] = Field(default_factory=list)
    """Optional list of constraint expressions describing relationships between axes."""

    @model_validator(mode="after")
    def _validate_reference_code(self) -> Definition:
        """Validate that reference contains valid Python code with a 'run' function.

        Raises
        ------
        ValueError
            If the reference code is not valid Python syntax or doesn't contain
            a top-level 'run' function.
        """
        try:
            mod = ast.parse(self.reference, mode="exec")
        except SyntaxError as e:
            raise ValueError(f"Reference must be valid Python code: {e}") from e

        # Check for 'run' function
        has_run_func = any(
            isinstance(node, ast.FunctionDef) and node.name == "run"
            for node in mod.body
        )
        if not has_run_func:
            raise ValueError("Reference must define a top-level function named 'run'")
        return self

    @model_validator(mode="after")
    def _validate_reference_inputs_match(self) -> Definition:
        """Validate that ``run()`` parameter names match the ``inputs`` keys in order.

        Raises
        ------
        ValueError
            If the parameter count or names of the ``run`` function do not match
            the ``inputs`` dictionary.
        """
        """Check that ``run()`` parameter names match *inputs* keys (in order).

        Uses AST to extract the parameter list of the top-level ``run`` function
        and compares it against the ordered keys of *inputs*. Both count and names
        must agree exactly.

        Returns an error string on mismatch, or ``None`` on success.
        """
        try:
            tree = ast.parse(self.reference, mode="exec")
        except SyntaxError:
            # already caught by Definition's reference validator
            return None

        run_func = next(
            (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"),
            None,
        )
        if run_func is None:
            # already caught by Definition's reference validator
            return None  

        args = run_func.args
        param_names: list[str] = [a.arg for a in (args.posonlyargs or [])] + [
            a.arg for a in args.args
        ]
        input_names = list(self.inputs.keys())

        if len(param_names) != len(input_names):
            raise ValueError(
                f"run() has {len(param_names)} parameter(s) {param_names} but "
                f"definition 'inputs' has {len(input_names)} "
                f"entr{'y' if len(input_names) == 1 else 'ies'} "
                f"{input_names}. They must match exactly."
            )

        mismatched = [(p, i) for p, i in zip(param_names, input_names) if p != i]
        if mismatched:
            raise ValueError(
                f"run() parameter names don't match definition 'inputs' keys. "
                f"Mismatches (run_param → input_key): {mismatched}. "
                f"run() params: {param_names}, inputs: {input_names}."
            )

        return self

    @model_validator(mode="after")
    def _verify_custom_inputs_entrypoint(self) -> Definition:
        """Verify that the custom inputs entrypoint is a valid identifier defined in reference code."""
        if self.custom_inputs_entrypoint is None:
            return self

        if not self.custom_inputs_entrypoint.isidentifier():
            raise ValueError(
                f"custom_inputs_entrypoint must be a valid Python identifier, "
                f"got: {self.custom_inputs_entrypoint!r}"
            )

        # Check that the function is defined in the reference code
        try:
            tree = ast.parse(self.reference, mode="exec")
        except SyntaxError:
            # Reference syntax is validated by _validate_reference_code
            return self

        has_entrypoint = any(
            isinstance(node, ast.FunctionDef)
            and node.name == self.custom_inputs_entrypoint
            for node in tree.body
        )
        if not has_entrypoint:
            raise ValueError(
                f"custom_inputs_entrypoint '{self.custom_inputs_entrypoint}' "
                f"is not defined as a top-level function in the reference code"
            )

        return self

    @model_validator(mode="after")
    def _validate_input_names_are_not_axes(self) -> Definition:
        """Validate that input names are not axes.

        Raises
        ------
        ValueError
            If any input name is an axis.
        """
        for name in self.inputs.keys():
            if name in self.axes:
                raise ValueError(f"Input name '{name}' is not allowed to be an axis.")
        return self

    @model_validator(mode="after")
    def _validate_input_output_names(self) -> Definition:
        """Validate that input and output names are unique and do not overlap.

        Raises
        ------
        ValueError
            If the input or output names are not unique or overlap.
        """
        if set(self.inputs.keys()) & set(self.outputs.keys()):
            raise ValueError("Input and output names must not overlap")
        return self

    @model_validator(mode="after")
    def _validate_constraints_syntax(self) -> Definition:
        """Validate that constraints are valid Python expressions.

        Raises
        ------
        ValueError
            If any constraint is not a valid Python expression.
        """
        if self.constraints is not None:
            for constraint in self.constraints:
                try:
                    ast.parse(constraint, mode="eval")
                except SyntaxError as e:
                    raise ValueError(
                        f"Constraints must be valid Python expressions: {e}"
                    ) from e
        return self

    @model_validator(mode="after")
    def _validate_tensor_axis_references(self) -> Definition:
        """Validate that tensor shapes reference defined axes.

        Ensures that all axis names used in input and output tensor shapes
        are properly defined in the axes dictionary.

        Raises
        ------
        ValueError
            If any tensor shape references an undefined axis.
        """
        all_tensors = {**self.inputs, **self.outputs}

        for tensor_name, tensor_spec in all_tensors.items():
            if tensor_spec.shape is None:
                continue
            for axis_name in tensor_spec.shape:
                if axis_name.isdigit():
                    continue
                if axis_name not in self.axes:
                    tensor_type = (
                        "input" if tensor_name in self.inputs else "output"
                    )
                    raise ValueError(
                        f'{tensor_type.capitalize()} "{tensor_name}" references undefined '
                        f'axis "{axis_name}".'
                    )
        return self

    def _get_variable_names(self, expr: str) -> list[str]:
        """Get all variable names from a mathematical expression."""
        tree = ast.parse(expr, mode="eval")
        return [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]

    @cached_property
    def const_axes(self) -> dict[str, int]:
        """Get all constant axes and their values.

        Returns
        -------
        dict[str, int]
            Dictionary mapping constant axis names to their fixed values.
        """
        return {
            name: axis.value
            for name, axis in self.axes.items()
            if isinstance(axis, AxisConst)
        }

    @cached_property
    def var_axes(self) -> list[str]:
        """Get all variable axis names.

        Returns
        -------
        list[str]
            List of all variable axis names defined in this Definition.
        """
        return [name for name, axis in self.axes.items() if isinstance(axis, AxisVar)]

    @cached_property
    def expr_axes(self) -> dict[str, AxisExpr]:
        """Get all expression axis names.

        Returns
        -------
        dict[str, AxisExpr]
            List of all expression axis names defined in this Definition.
        """
        return {name: axis for name, axis in self.axes.items() if isinstance(axis, AxisExpr)}

    def get_axes_values(
        self, input_shapes: Iterable[Optional[tuple[int, ...]]]
    ) -> dict[str, int]:
        """Get concrete variable axis values from input shapes.

        Parameters
        ----------
        input_shapes : Iterable[Optional[tuple[int, ...]]]
            Iterable of input tensor shapes.

        Returns
        -------
        dict[str, int]
            Dictionary mapping variable axis names to their concrete values.

        Raises
        ------
        ValueError
            If a required variable axis value is missing from input_shapes, or a axis occurs in
            multiple input tensors, but the values are not consistent.
        """
        var_axes_values: dict[str, int] = {}
        for (inp_name, inp_spec), inp_shape in zip(self.inputs.items(), input_shapes):
            if inp_spec.shape is None:  # scalar, no shape
                continue
            if len(inp_spec.shape) != len(inp_shape):
                raise ValueError(
                    f"Input '{inp_name}''s defined dimension is {len(inp_spec.shape)} but the "
                    f"actual dimension is {len(inp_shape)}"
                )
            for axis_name, axis_value in zip(inp_spec.shape, inp_shape):
                if axis_name in self.axes and self.axes[axis_name].type == "var":
                    if axis_name in var_axes_values:
                        if var_axes_values[axis_name] != axis_value:
                            raise ValueError(
                                f"Axis '{axis_name}' has different values for different input "
                                f"tensors: {var_axes_values[axis_name]} and {axis_value}"
                            )
                    else:
                        var_axes_values[axis_name] = axis_value

        if len(var_axes_values) != len(self.var_axes):
            raise ValueError(
                f"Missing values for variable axes: "
                f"{set(self.var_axes) - set(var_axes_values.keys())}"
            )
        return var_axes_values

    def get_axes_values_from_inputs(self, inputs: Iterable[Any]) -> dict[str, int]:
        """Get concrete variable axis values directly from input values.

        Convenience method that combines extract_shapes and get_var_axes_values.

        Parameters
        ----------
        inputs : Iterable[Any]
            Iterable of input values (tensors or other types).

        Returns
        -------
        dict[str, int]
            Dictionary mapping variable axis names to their concrete values.
        """
        shapes = [tuple(val.shape) if hasattr(val, "shape") else None for val in inputs]
        return self.get_axes_values(shapes)

    def get_resolved_axes_values(self, var_axes_values: dict[str, int]) -> dict[str, int]:
        """Get concrete axis values from variable axis values. Resolves all expressions.

        Parameters
        ----------
        var_axes_values : dict[str, int]
            Dictionary mapping variable axis names to their concrete values.
        """
        resolved_axes_values: dict[str, int] = self.const_axes.copy()

        for name, axis_value in var_axes_values.items():
            if name not in var_axes_values:
                raise ValueError(f"Missing value for variable axis '{name}'")
            resolved_axes_values[name] = axis_value

        for name, axis in self.expr_axes.items():
            resolved_axes_values[name] = resolve_shape_expression(axis.expression, resolved_axes_values)
        return resolved_axes_values

    def _get_shapes(
        self,
        tensors: Iterable[TensorSpec],
        var_axes_values: Optional[dict[str, int]] = None,
    ) -> list[Optional[tuple[int, ...]]]:
        """Get concrete tensor shapes given variable axis values.

        Parameters
        ----------
        tensors : Iterable[TensorSpec]
            List of tensor specifications to compute shapes for.
        var_values : Optional[dict[str, int]], default=None
            Values for variable axes. If None, defaults to empty dictionary.

        Returns
        -------
        list[Optional[tuple[int, ...]]]
            List of concrete shapes as tuples of integers. None for scalar tensors.

        Raises
        ------
        ValueError
            If a required variable axis value is missing from var_values.
        """
        var_axes_values = var_axes_values or {}
        shapes = []

        resolved_axes = self.get_resolved_axes_values(var_axes_values)

        for tensor_spec in tensors:
            if tensor_spec.shape is None:  # scalar, no shape
                shapes.append(None)
                continue
            shape = []
            for axis_name in tensor_spec.shape:
                if axis_name.isdigit():
                    value = int(axis_name)
                else:
                    value = resolved_axes[axis_name]
                shape.append(value)
            shapes.append(tuple(shape))

        return shapes

    def get_input_shapes(
        self, var_axes_values: Optional[dict[str, int]] = None
    ) -> dict[str, Optional[tuple[int, ...]]]:
        """Get concrete input shapes given variable axis values.

        Parameters
        ----------
        var_values : Optional[dict[str, int]], default=None
            Values for variable axes. If None, defaults to empty dictionary.

        Returns
        -------
        dict[str, Optional[tuple[int, ...]]]
            Dictionary mapping input tensor names to their concrete shapes as tuples of integers. None for scalar tensors.

        Raises
        ------
        ValueError
            If a required variable axis value is missing from var_values.
        """
        shapes = self._get_shapes(self.inputs.values(), var_axes_values)
        return dict(zip(self.inputs.keys(), shapes))

    def get_output_shapes(
        self, var_values: Optional[dict[str, int]] = None
    ) -> dict[str, Optional[tuple[int, ...]]]:
        """Get concrete output shapes given variable axis values.

        Parameters
        ----------
        var_values : Optional[dict[str, int]], default=None
            Values for variable axes. If None, defaults to empty dictionary.

        Returns
        -------
        dict[str, Optional[tuple[int, ...]]]
            Dictionary mapping output tensor names to their concrete shapes as tuples of integers. None for scalar tensors.

        Raises
        ------
        ValueError
            If a required variable axis value is missing from var_values.
        """
        shapes = self._get_shapes(self.outputs.values(), var_values)
        return dict(zip(self.outputs.keys(), shapes))

    @cached_property
    def torch_input_dtypes(self) -> list[torch.dtype]:
        """Get the torch data types of the input tensors.

        Returns
        -------
        list[torch.dtype]
            List of torch data types of the input tensors.
        """
        return [dtype_str_to_torch_dtype(spec.dtype) for spec in self.inputs.values()]

    @cached_property
    def torch_output_dtypes(self) -> list[torch.dtype]:
        """Get the torch data types of the output tensors.

        Returns
        -------
        list[torch.dtype]
            List of torch data types of the output tensors.
        """
        return [dtype_str_to_torch_dtype(spec.dtype) for spec in self.outputs.values()]
