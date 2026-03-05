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

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

NonEmptyString = Annotated[str, Field(min_length=1)]
"""Type alias for non-empty strings with minimum length of 1."""

NonNegativeInt = Annotated[int, Field(ge=0)]
"""Type alias for non-negative integers."""


class BaseModelWithDocstrings(BaseModel):
    """Base model with the attribute docstrings being extracted to the model JSON schema."""

    model_config = ConfigDict(use_attribute_docstrings=True)
