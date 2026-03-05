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

"""Compiler subsystem package.

This package provides the infrastructure for building solutions into executable runnables.
It includes:
- Builder: Abstract base class for different language/build system implementations
- BuilderRegistry: Central registry for managing and dispatching builders
- Runnable: Executable wrapper around compiled solutions
- RunnableMetadata: Metadata about build process and source

The typical workflow is:
1. Get the singleton registry: registry = BuilderRegistry.get_instance()
2. Build a solution: runnable = registry.build(definition, solution)
3. Execute: result = runnable(**inputs)
"""

from .builder import Builder, BuildError
from .registry import BuilderRegistry
from .runnable import Runnable, RunnableMetadata, RunnableInputs

__all__ = [
    "Builder",
    "BuildError",
    "BuilderRegistry",
    "Runnable",
    "RunnableMetadata",
    "RunnableInputs",
]
