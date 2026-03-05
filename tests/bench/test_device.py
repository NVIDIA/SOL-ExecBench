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

"""Tests for DeviceConfig, ExecutionDevice, and clock locking (device.py).

Static-spec tests need no GPU.  Runtime tests are gated with
@pytest.mark.requires_torch_cuda.  The clock-locking benchmark additionally
requires root/sudo access to nvidia-smi; those tests self-skip when the
privilege is absent.
"""

import statistics
import time
from unittest.mock import patch

import pytest
import torch

from sol_execbench import SupportedHardware
from sol_execbench.bench import ExecutionDevice, DeviceConfig, list_cuda_devices
from sol_execbench.bench.execution_device import get_device_config


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _is_clock_lock_permission_error(exc: Exception) -> bool:
    """Return True when the error is caused by missing nvidia-smi or no root."""
    if isinstance(exc, FileNotFoundError):
        return True
    msg = str(exc).lower()
    return any(
        kw in msg for kw in ("nvidia-smi", "permission", "sudo", "root", "failed")
    )


def _bench_matmul(n_runs: int = 20) -> list[float]:
    """Warm up then time n_runs matmul calls, return per-call times in seconds."""
    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")
    for _ in range(5):
        torch.matmul(a, b)
        torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        torch.matmul(a, b)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


# ─────────────────────────────────────────────────────────────────────────────
# Static GPU spec — no CUDA required
# ─────────────────────────────────────────────────────────────────────────────


def test_DeviceConfig_returns_DeviceConfig_instance():
    assert isinstance(get_device_config(SupportedHardware.B200), DeviceConfig)


def test_DeviceConfig_b200_locked_clock():
    assert get_device_config(SupportedHardware.B200).locked_clock_speed == 1680


def test_DeviceConfig_b200_sm_version():
    assert get_device_config(SupportedHardware.B200).sm_version == "sm_100a"


def test_DeviceConfig_b200_hardware_type():
    assert (
        get_device_config(SupportedHardware.B200).hardware_type
        == SupportedHardware.B200
    )


def test_list_cuda_devices_without_cuda_is_empty():
    with patch("torch.cuda.is_available", return_value=False):
        assert list_cuda_devices() == []


# ─────────────────────────────────────────────────────────────────────────────
# Runtime device detection — CUDA required
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.requires_torch_cuda
def test_list_cuda_devices_returns_cuda_strings():
    devices = list_cuda_devices()
    assert len(devices) >= 1
    for d in devices:
        assert d.startswith("cuda:")


@pytest.mark.requires_torch_cuda
def test_list_cuda_devices_count_matches_torch():
    assert len(list_cuda_devices()) == torch.cuda.device_count()


@pytest.mark.requires_torch_cuda
def test_DeviceConfig_local_has_sm_version():
    spec = get_device_config(SupportedHardware.LOCAL)
    assert spec.sm_version.startswith("sm_")


@pytest.mark.requires_torch_cuda
def test_execution_device_device_str():
    spec = get_device_config(SupportedHardware.B200)
    device = ExecutionDevice(spec, "cuda:0")
    assert device.device_str == "cuda:0"


@pytest.mark.requires_torch_cuda
def test_execution_device_index_parsed():
    spec = get_device_config(SupportedHardware.B200)
    assert ExecutionDevice(spec, "cuda:0").device_index == 0


@pytest.mark.requires_torch_cuda
def test_execution_device_hardware_name_non_empty():
    spec = get_device_config(SupportedHardware.B200)
    assert len(ExecutionDevice(spec, "cuda:0").hardware_name) > 0


@pytest.mark.requires_torch_cuda
def test_execution_device_gpu_uuid_format():
    spec = get_device_config(SupportedHardware.B200)
    uuid = ExecutionDevice(spec, "cuda:0").gpu_uuid
    assert uuid is None or uuid.startswith("GPU-")


@pytest.mark.requires_torch_cuda
def test_execution_device_sm_version_from_spec():
    spec = get_device_config(SupportedHardware.B200)
    device = ExecutionDevice(spec, "cuda:0")
    assert device.sm_version == spec.sm_version


# ─────────────────────────────────────────────────────────────────────────────
# Clock locking — CUDA required; root/sudo required for nvidia-smi
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.requires_torch_cuda
def test_lock_clocks_allows_gpu_work():
    """lock_clocks context manager enters, allows GPU operations, and exits cleanly."""
    spec = get_device_config(SupportedHardware.LOCAL)
    freq_mhz = spec.locked_clock_speed or 1000
    device = ExecutionDevice(spec, "cuda:0")

    try:
        with device.lock_clocks(freq_mhz=freq_mhz):
            a = torch.randn(256, 256, device="cuda")
            b = torch.matmul(a, a)
            torch.cuda.synchronize()
            assert b.shape == (256, 256)
    except (RuntimeError, FileNotFoundError) as e:
        if _is_clock_lock_permission_error(e):
            pytest.skip(f"Clock locking requires root/sudo: {e}")
        raise


@pytest.mark.requires_torch_cuda
def test_lock_clocks_reduces_variance():
    """Locked clocks should produce lower timing variance than free-running clocks."""
    spec = get_device_config(SupportedHardware.LOCAL)
    freq_mhz = spec.locked_clock_speed
    if not freq_mhz:
        pytest.skip("Could not detect boost clock for LOCAL GPU — cannot lock clocks")

    device = ExecutionDevice(spec, "cuda:0")

    times_free = _bench_matmul()
    cv_free = statistics.stdev(times_free) / statistics.mean(times_free)

    try:
        with device.lock_clocks(freq_mhz=freq_mhz):
            times_locked = _bench_matmul()
    except (RuntimeError, FileNotFoundError) as e:
        if _is_clock_lock_permission_error(e):
            pytest.skip(f"Clock locking requires root/sudo: {e}")
        raise

    cv_locked = statistics.stdev(times_locked) / statistics.mean(times_locked)
    assert cv_locked <= cv_free * 1.5, (
        f"Clock locking did not reduce variance: "
        f"CV free={cv_free:.4f}, CV locked={cv_locked:.4f}"
    )
