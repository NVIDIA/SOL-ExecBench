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

"""Logging utilities for sol_execbench evaluate."""

import logging
from typing import Optional, Union

_PACKAGE_LOGGER_NAME = "sol_execbench-evaluate"

logging.getLogger(_PACKAGE_LOGGER_NAME).addHandler(logging.NullHandler())


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger namespaced under the package root."""
    full_name = _PACKAGE_LOGGER_NAME if not name else f"{_PACKAGE_LOGGER_NAME}.{name}"
    return logging.getLogger(full_name)


def configure_logging(
    level: Union[int, str] = "INFO",
    *,
    handler: Optional[logging.Handler] = None,
    formatter: Optional[logging.Formatter] = None,
    propagate: bool = False,
) -> logging.Logger:
    """Configure the root package logger and return it."""
    logger = logging.getLogger(_PACKAGE_LOGGER_NAME)

    if isinstance(level, str):
        numeric_level = logging.getLevelName(level.upper())
        if isinstance(numeric_level, str):
            raise ValueError(f"Unknown log level: {level}")
        level = numeric_level

    logger.setLevel(level)

    if handler is None:
        handler = logging.StreamHandler()
    if formatter is None:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
        )
    handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = propagate

    return logger
