"""Common utilities and base classes for data models."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

NonEmptyString = Annotated[str, Field(min_length=1)]
"""Type alias for non-empty strings with minimum length of 1."""

NonNegativeInt = Annotated[int, Field(ge=0)]
"""Type alias for non-negative integers."""


class BaseModelWithDocstrings(BaseModel):
    """Base model with the attribute docstrings being extracted to the model JSON schema."""

    model_config = ConfigDict(use_attribute_docstrings=True)
