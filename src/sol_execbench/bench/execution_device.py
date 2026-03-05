"""Device management for GPU execution and clock control.

This module provides utilities for managing GPU devices, including:
- Device detection and properties
- GPU hardware identification
- Clock frequency locking for benchmarking
- Context managers for controlled execution

All clock management operations use GPU UUIDs instead of device indices
to ensure robustness across different CUDA_VISIBLE_DEVICES configurations.
"""

import logging
import subprocess
import time
from contextlib import contextmanager
from typing import Optional

import torch

from ..data import SupportedHardware
from .config import DeviceConfig

logger = logging.getLogger(__name__)


def _get_gpu_uuid(device_index: int) -> Optional[str]:
    """Get GPU UUID for nvidia-smi operations.

    Returns UUID with 'GPU-' prefix for nvidia-smi compatibility.

    Args:
        device_index: CUDA device index

    Returns:
        GPU UUID string with 'GPU-' prefix (e.g., "GPU-12345..."), or None if unavailable

    Examples:
        >>> uuid = _get_gpu_uuid(0)
        >>> if uuid:
        ...     # Use with nvidia-smi: nvidia-smi -i GPU-12345...
        ...     pass
    """
    if not torch.cuda.is_available():
        return None

    try:
        props = torch.cuda.get_device_properties(device_index)
        return f"GPU-{props.uuid}"
    except Exception as e:
        logger.warning(f"Failed to get GPU UUID for device {device_index}: {e}")
        return None


# GPU specifications for supported hardware
# Set to ~80% of boost clock for stability
_HARDWARE_TO_SPEC: dict[SupportedHardware, DeviceConfig] = {
    SupportedHardware.B200: DeviceConfig(
        hardware_type=SupportedHardware.B200,
        locked_clock_speed=1680,
        sm_version="sm_100a",
    ),
    SupportedHardware.LOCAL: DeviceConfig(
        hardware_type=SupportedHardware.LOCAL,
        locked_clock_speed=0,
        sm_version="",
    ),
}


def _detect_boost_clock(device_index: int = 0) -> Optional[int]:
    """Detect boost clock frequency for a local GPU.

    Args:
        device_index: CUDA device index

    Returns:
        Boost clock frequency in MHz, or None if detection fails

    Examples:
        >>> boost_clock = _detect_boost_clock(0)
        >>> if boost_clock:
        ...     print(f"Boost clock: {boost_clock} MHz")
        ...     print(f"Recommended lock: {int(boost_clock * 0.8)} MHz")
    """
    gpu_uuid = _get_gpu_uuid(device_index)
    if gpu_uuid is None:
        return None

    try:
        # Query max graphics clock using UUID
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.max.sm",
                "--format=csv,noheader,nounits",
                "-i",
                gpu_uuid,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        boost_clock = int(result.stdout.strip())
        logger.info(
            f"Detected boost clock for device {device_index}: {boost_clock} MHz"
        )
        return boost_clock
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"Failed to detect boost clock for device {device_index}: {e.stderr}"
        )
        return None
    except Exception as e:
        logger.warning(f"Failed to detect boost clock for device {device_index}: {e}")
        return None


def get_device_config(
    hardware_type: SupportedHardware, device_index: int = 0
) -> DeviceConfig:
    """Get GPU specification for a hardware type.

    For LOCAL hardware, automatically detects boost clock and sets locked_clock_speed
    to 80% of boost clock for stable benchmarking.

    Args:
        hardware_type: SupportedHardware enum member
        device_index: CUDA device index (used for LOCAL hardware detection)

    Returns:
        DeviceConfig object

    Raises:
        ValueError: If hardware type not supported
    """
    spec = _HARDWARE_TO_SPEC.get(hardware_type)
    if spec is None:
        raise ValueError(
            f"No GPU specification for {hardware_type}. "
            f"Supported hardware: {list(_HARDWARE_TO_SPEC.keys())}"
        )

    # Lazy load local GPU spec with boost clock detection
    if hardware_type == SupportedHardware.LOCAL:
        # Detect SM version
        if spec.sm_version == "":
            major, minor = torch.cuda.get_device_capability(device_index)
            sm_version = f"sm_{major}{minor}"
        else:
            sm_version = spec.sm_version

        # Detect boost clock and set to 80% for stability
        locked_clock = spec.locked_clock_speed
        if locked_clock == 0:
            boost_clock = _detect_boost_clock(device_index)
            if boost_clock:
                locked_clock = int(boost_clock * 0.8)
                logger.info(
                    f"Auto-detected LOCAL GPU: boost={boost_clock} MHz, "
                    f"setting locked_clock={locked_clock} MHz (80% of boost)"
                )
            else:
                logger.warning(
                    "Failed to detect boost clock for local GPU, clock locking may not work"
                )

        # Return updated spec
        spec = DeviceConfig(
            hardware_type=hardware_type,
            locked_clock_speed=locked_clock,
            sm_version=sm_version,
        )
        # Cache for future calls
        _HARDWARE_TO_SPEC[SupportedHardware.LOCAL] = spec

    return spec


class ExecutionDevice:
    """Represents a physical GPU device for execution with clock control capabilities.

    Provides properties for device identification and methods for clock
    frequency management. Uses GPU UUIDs for all nvidia-smi operations
    to ensure robustness across different CUDA_VISIBLE_DEVICES configurations.

    All properties are set upfront in __init__ (no lazy loading).

    Attributes:
        device_str: Device string (e.g., "cuda:0")
        device_index: PyTorch CUDA device index (None if not CUDA)
        hardware_name: Full GPU hardware name
        hardware_type: SupportedHardware enum member (None if not recognized)
        gpu_uuid: GPU UUID (None if unavailable)
        device_config: DeviceConfig with clock speed and SM version (None if not supported)
        sm_version: CUDA compute capability (e.g., "sm_89", "sm_100a")
        has_default_clock: Whether GPU has default clock frequency
        default_clock_mhz: Default clock frequency in MHz (None if not defined)
    """

    def __init__(self, device_config: DeviceConfig, device_str: str = "cuda:0"):
        """Initialize an execution device.

        All properties are computed upfront in this constructor.

        Args:
            device_config: GPU specification with clock speed and SM version
            device_str: Device string (e.g., "cuda:0", "cuda:1")

        Raises:
            AssertionError: If CUDA is not available
        """
        # Assert CUDA is available
        assert torch.cuda.is_available(), (
            "CUDA must be available to use ExecutionDevice"
        )

        # Store GPU spec
        self.device_config = device_config
        self.hardware_type = device_config.hardware_type
        self.default_clock_mhz = device_config.locked_clock_speed
        self.sm_version = device_config.sm_version
        self.has_default_clock = True

        # Parse device string
        self.device_str = device_str

        # Parse device index
        if ":" in device_str:
            try:
                self.device_index = int(device_str.split(":")[-1])
            except ValueError:
                self.device_index = 0
        else:
            # Default to device 0 if no index specified
            self.device_index = 0

        # Get hardware name from PyTorch
        try:
            props = torch.cuda.get_device_properties(self.device_index)
            self.hardware_name = props.name
        except Exception:
            self.hardware_name = "Unknown GPU"

        # Get GPU UUID (with "GPU-" prefix for nvidia-smi)
        self.gpu_uuid = _get_gpu_uuid(self.device_index)

    def _run_nvidia_smi(self, args: list[str]) -> str:
        """Run nvidia-smi command with the given arguments.

        Args:
            args: List of command-line arguments for nvidia-smi

        Returns:
            Command output as string

        Raises:
            RuntimeError: If command fails
        """
        cmd = ["nvidia-smi"] + args
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"nvidia-smi command failed: {e.stdout}\n{e.stderr}"
            ) from e

    def _get_current_clock(self) -> Optional[int]:
        """Get current graphics clock frequency in MHz using UUID.

        Returns:
            Current graphics clock in MHz, or None if unavailable
        """
        if self.gpu_uuid is None:
            return None

        try:
            result = self._run_nvidia_smi(
                [
                    "--query-gpu=clocks.sm",
                    "--format=csv,noheader,nounits",
                    "-i",
                    self.gpu_uuid,
                ]
            )
            return int(result.strip())
        except Exception:
            return None

    def _set_clock_lock(self, freq_mhz: int) -> None:
        """Lock graphics clocks to specified frequency using UUID.

        Verifies the clock was successfully set after locking.

        Args:
            freq_mhz: Frequency in MHz to lock clocks to

        Raises:
            RuntimeError: If clock locking fails or verification fails
        """
        if self.gpu_uuid is None:
            raise RuntimeError(
                "Cannot lock clocks on non-CUDA device or device without UUID"
            )

        # Set the clock lock
        self._run_nvidia_smi(["-i", self.gpu_uuid, "-lgc", str(freq_mhz)])

        # wait for nvidia-smi to update
        time.sleep(2.0)

        # Verify the clock was set correctly
        current_clock = self._get_current_clock()
        if current_clock is None:
            logger.warning(
                f"Could not verify clock lock to {freq_mhz} MHz (unable to read current clock)"
            )
        else:
            # Allow 2% tolerance for verification
            tolerance_mhz = freq_mhz * 0.02
            diff_mhz = abs(current_clock - freq_mhz)
            if diff_mhz > tolerance_mhz:
                raise RuntimeError(
                    f"Clock lock verification failed: requested {freq_mhz} MHz, "
                    f"but current clock is {current_clock} MHz "
                    f"(diff: {diff_mhz:.1f} MHz, tolerance: {tolerance_mhz:.1f} MHz)"
                )
            else:
                logger.info(
                    f"Successfully locked clock to {current_clock} MHz "
                    f"(requested {freq_mhz} MHz, diff: {diff_mhz:.1f} MHz)"
                )

    def _reset_clock_lock(self) -> None:
        """Reset graphics clocks to default using UUID.

        Raises:
            RuntimeError: If clock reset fails
        """
        if self.gpu_uuid is None:
            raise RuntimeError(
                "Cannot reset clocks on non-CUDA device or device without UUID"
            )

        self._run_nvidia_smi(["-i", self.gpu_uuid, "-rgc"])

    @contextmanager
    def lock_clocks(self, freq_mhz: Optional[int] = None):
        """Context manager to lock GPU clocks during execution.

        Automatically uses GPU's default clock if freq_mhz is None and
        the GPU has a default clock defined.

        Args:
            freq_mhz: Frequency in MHz to lock clocks to, or None to use
                     GPU's default clock if available

        Raises:
            RuntimeError: If clock locking fails or no frequency specified
                         for GPU without default clock

        Example:
            >>> device = ExecutionDevice("cuda:0")
            >>> with device.lock_clocks():  # Uses default for B200
            ...     # Run benchmarks with locked clocks
            ...     pass
            >>> with device.lock_clocks(1500):  # Use specific frequency
            ...     # Run benchmarks at 1500 MHz
            ...     pass
        """
        # Determine frequency to use
        if freq_mhz is None:
            if self.has_default_clock:
                freq_mhz = self.default_clock_mhz
            else:
                raise RuntimeError(
                    f"No frequency specified and {self.hardware_name} has no default clock. "
                    "Please specify freq_mhz explicitly."
                )

        # Lock clocks
        try:
            self._set_clock_lock(freq_mhz)
            yield
        finally:
            # Always reset clocks on exit
            try:
                self._reset_clock_lock()
            except Exception:
                # Best effort reset
                pass

    def set_as_current(self) -> None:
        """Set this device as the current PyTorch device.

        Raises:
            RuntimeError: If setting device fails
        """
        torch.cuda.set_device(self.device_index)

    def __str__(self) -> str:
        """String representation of the device."""
        return f"{self.device_str} ({self.hardware_name})"

    def __repr__(self) -> str:
        """Detailed string representation of the device."""
        return f"ExecutionDevice('{self.device_str}')"


def is_cuda_available() -> bool:
    """Check if CUDA is available on this system.

    Returns:
        True if CUDA is available, False otherwise
    """
    return torch.cuda.is_available()


def list_cuda_devices() -> list[str]:
    """List all available CUDA device strings.

    Returns:
        List of device strings (e.g., ['cuda:0', 'cuda:1'])
    """
    if not is_cuda_available():
        return []

    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]


__all__ = [
    "ExecutionDevice",
    "is_cuda_available",
    "list_cuda_devices",
]
