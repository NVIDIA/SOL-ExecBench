from __future__ import annotations

from dataclasses import dataclass

from ...data import SupportedHardware


@dataclass(frozen=True)
class DeviceConfig:
    """Specification for a GPU hardware type.

    Attributes:
        hardware_type: SupportedHardware enum member
        locked_clock_speed: Graphics clock frequency in MHz for benchmarking (~80% of boost)
        sm_version: CUDA compute capability (e.g., "sm_89", "sm_100a")
    """

    hardware_type: SupportedHardware
    locked_clock_speed: int
    sm_version: str
