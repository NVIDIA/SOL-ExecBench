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
