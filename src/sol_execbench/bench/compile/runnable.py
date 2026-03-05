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

"""Runnable wrapper for compiled solutions."""

from __future__ import annotations

import torch
from functools import cached_property
from typing import Any, Callable, Iterator, Literal, Optional, Union

from pydantic import BaseModel, Field

from ...data import Definition
from ...data.dtypes import dtype_str_to_torch_dtype


class RunnableMetadata(BaseModel):
    """Metadata about a runnable implementation.

    This class stores information about how a runnable was built, including the
    builder type, source definition/solution, and additional builder-specific data.
    """

    build_type: Union[Literal["torch", "python", "triton"], str]
    """The type of build that produced this runnable (e.g., 'python', 'torch', 'triton')."""
    definition_name: str
    """Name of the definition that specifies the expected interface."""
    solution_name: str
    """Name of the solution that was compiled into this runnable."""
    destination_passing_style: bool = True
    """Whether the runnable uses destination-passing style."""
    definition: Optional[Definition] = None
    """The full definition that was used to build the runnable. It's not necessary to be set, but
    required when calling in keyword passing style and value-returning style."""
    misc: dict[str, Any] = Field(default_factory=dict)
    """Miscellaneous metadata about the runnable. Contents vary by builder type."""


class RunnableInputs(BaseModel):
    """The inputs for a particular runnable. Concrete axes are stored for convenience."""

    callable_inputs: list[Any]
    """The inputs for the callable."""
    resolved_axes: dict[str, int]
    """The resolved axes values."""

    def __len__(self) -> int:
        return len(self.callable_inputs)

    def __getitem__(self, index: int) -> Any:
        return self.callable_inputs[index]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.callable_inputs)

    def clone_tensors(self) -> RunnableInputs:
        return RunnableInputs(
            callable_inputs=[
                v.clone() if isinstance(v, torch.Tensor) else v
                for v in self.callable_inputs
            ],
            resolved_axes=self.resolved_axes,
        )


class Runnable:
    """An executable wrapper around a compiled solution.

    A Runnable encapsulates a callable function along with metadata about how it was built
    and a cleanup function to release resources. It provides a uniform interface for
    executing solutions regardless of the build system or language used.
    """

    metadata: RunnableMetadata
    """Metadata about the build process and source solution."""

    _callable: Callable[..., Any]
    """The underlying callable function."""
    _cleaner: Optional[Callable[[], None]]
    """Optional cleanup function to release build artifacts and resources."""
    _gen_inputs_callable: Optional[Callable[..., Any]]
    """Optional callable to generate inputs. It should be a standalone method."""

    def __init__(
        self,
        callable: Callable[..., Any],
        metadata: RunnableMetadata,
        cleaner: Optional[Callable[[], None]] = None,
        gen_inputs_callable: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Constructor for the Runnable class.

        Parameters
        ----------
        callable : Callable[..., Any]
            The callable that is wrapped by the runnable.
        metadata : RunnableMetadata
            The metadata for the runnable.
        cleaner : Optional[Callable[[], None]]
            The cleaner function for the runnable. It will clean up the build artifacts/resources.
        """
        self._callable = callable
        self.metadata = metadata
        self._cleaner = cleaner
        self._gen_inputs_callable = gen_inputs_callable

    def gen_inputs(
        self, axes_and_scalars: dict[str, int], device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Generate inputs for the runnable.

        This method generates inputs for the runnable using the gen_inputs_callable.

        Parameters
        ----------
        axes_and_scalars : dict[str, int]
            The axes and scalar values for the inputs.
        device : torch.device
            The device to generate the inputs on.

        Returns
        -------
        dict[str, torch.Tensor]
            The generated inputs.

        """
        if self._gen_inputs_callable is None:
            raise ValueError("gen_inputs_callable is not set.")
        return self._gen_inputs_callable(axes_and_scalars, device)

    def __call__(self, *args: Any) -> Any:
        """Execute the runnable with positional arguments.

        This method calls the underlying compiled function with the provided inputs.
        If the function returns a single-element tuple, it is automatically unpacked
        to a scalar value for convenience.

        Parameters
        ----------
        args : Any
            Positional arguments for the underlying function.

        Returns
        -------
        Any
            The result of the underlying function. Single-element tuples are unpacked
            to scalar values.
        """
        ret = self._callable(*args)
        return self._revise_return_value(ret)

    def _revise_return_value(self, ret: Any) -> Any:
        """
        Revise the return value. Unpack 0-element tuple to None, 1-element tuple to its element.

        Parameters
        ----------
        ret : Any
            The return value to revise.

        Returns
        -------
        Any
            The revised return value.
        """
        if not isinstance(ret, tuple):
            return ret

        if len(ret) > 1:
            return ret
        elif len(ret) == 1:
            return ret[0]
        else:
            # len(ret) == 0
            return None

    @cached_property
    def _arg_names(self) -> list[str]:
        """Get the order of arguments for the underlying function."""
        if self.metadata.definition is None:
            raise ValueError(
                "When calling in keyword passing style, metadata.full_definition must "
                "be set."
            )
        definition: Definition = self.metadata.definition
        if self.metadata.destination_passing_style:
            return list(definition.inputs.keys()) + list(definition.outputs.keys())
        else:
            return list(definition.inputs.keys())

    def call_kwargs(self, **kwargs: Any) -> Any:
        """Call the runnable with keyword arguments.

        This method calls the underlying compiled function with the provided inputs.
        If the function returns a single-element tuple, it is automatically unpacked
        to a scalar value for convenience.
        """
        args = [kwargs[name] for name in self._arg_names]
        return self(*args)

    @cached_property
    def _output_dtype_list(self) -> list[torch.dtype]:
        """Get the list of output data types."""
        if self.metadata.definition is None:
            raise ValueError(
                "When calling in value-returning style, metadata.definition must "
                "be set."
            )
        definition = self.metadata.definition
        return [
            dtype_str_to_torch_dtype(output.dtype)
            for output in definition.outputs.values()
        ]

    def call_destination_passing(self, *args: Any) -> None:
        """Call the callable in destination-passing style (DPS). If the callable is already in DPS
        style, this method calls it directly. If the callable is in value-returning style,
        this method converts it to DPS style and calls it.

        Parameters
        ----------
        args : Any
            Positional arguments for the underlying function. Includes input tensors and output
            tensors.
        """
        import torch

        if self.metadata.destination_passing_style:
            self(*args)
            return

        # Convert value-returning style to destination-passing style
        if self.metadata.definition is None:
            raise ValueError(
                "When converting value-returning style to destination-passing style, "
                "metadata.definition must be set."
            )

        args_input = args[: len(self.metadata.definition.inputs)]
        args_output = args[len(self.metadata.definition.inputs) :]

        result = self._callable(*args_input)

        if len(args_output) == 0:
            return
        elif not isinstance(result, tuple):
            result = (result,)

        if len(result) != len(args_output):
            raise ValueError(
                "Value-returning style callable must return a tuple of the same length as "
                f"the number of outputs, got {len(result)} and {len(args_output)}"
            )

        for res_tensor, out_tensor in zip(result, args_output, strict=True):
            if not isinstance(res_tensor, torch.Tensor) or not isinstance(
                out_tensor, torch.Tensor
            ):
                raise ValueError(
                    "Destination-passing style callable must return a tuple of tensors, got "
                    f"{type(res_tensor)} and {type(out_tensor)}"
                )
            out_tensor.copy_(res_tensor)

    def _allocate_output_tensors(self, *args: Any) -> list[torch.Tensor]:
        """Allocate output tensors based on the definition and variable axis values."""
        import torch

        if self.metadata.definition is None:
            raise ValueError(
                "When converting destination-passing style to value-returning style, "
                "metadata.definition must be set."
            )
        definition = self.metadata.definition

        # Allocate output tensors first
        var_axes_values = definition.get_axes_values_from_inputs(args)
        output_shapes = list(definition.get_output_shapes(var_axes_values).values())

        # Determine device from input tensors
        device = next((v.device for v in args if hasattr(v, "device")), "cpu")
        dtype_list = self._output_dtype_list

        output_tensors: list[torch.Tensor] = []
        for shape, dtype in zip(output_shapes, dtype_list):
            shape = shape if shape is not None else ()
            output_tensors.append(torch.empty(shape, dtype=dtype, device=device))

        return output_tensors

    def call_value_returning(self, *args: Any) -> Any:
        """Call a destination-passing style (DPS) function in value-returning style.

        Some solutions use the destination-passing style,
        where output tensors are passed as arguments and the function modifies them in-place::

            function(**input_tensors, **output_tensors) -> None

        This method provides a value-returning interface by automatically allocating output
        tensors based on the definition, calling the DPS function, and returning the outputs::

            result = runnable.call_dps(**input_tensors)  # -> output_tensors

        Parameters
        ----------
        kwargs : Any
            Keyword arguments for input tensors matching the definition's input specification.

        Returns
        -------
        Any
            The output tensor(s). Single outputs are returned as-is, multiple outputs are
            returned as a tuple, and empty outputs return None.

        Raises
        ------
        ValueError
            If the metadata does not contain the full definition object needed for
            output tensor allocation.
        """
        if not self.metadata.destination_passing_style:
            return self(*args)

        # Convert destination-passing style to value-returning style
        output_tensors = self._allocate_output_tensors(*args)
        self._callable(*args, *output_tensors)
        return self._revise_return_value(tuple(output_tensors))

    def cleanup(self) -> None:
        """Clean up build artifacts and release resources.

        This method calls the cleaner function if one was provided during construction.
        It is idempotent: calling it multiple times is safe and has no additional effect
        after the first call.
        """
        if self._cleaner:
            try:
                self._cleaner()
            finally:
                self._cleaner = None
