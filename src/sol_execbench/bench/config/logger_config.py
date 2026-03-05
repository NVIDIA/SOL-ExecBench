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
