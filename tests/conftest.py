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

from pathlib import Path
from typing import List

import pytest


def _gpu_sm_version() -> int:
    """Return the SM version of the current GPU (e.g. 90, 100), or 0 if unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor
    except ImportError:
        return 0


requires_sm100 = pytest.mark.skipif(
    _gpu_sm_version() < 100,
    reason=f"Requires sm_100+ (detected sm_{_gpu_sm_version()})",
)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "timing_serial: GPU timing tests (skipped by default; run with: pytest tests -m timing_serial -n 0)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: List[pytest.Item]
) -> None:
    """Skip tests based on hardware availability.

    Also skips timing_serial tests unless explicitly selected with -m.
    """
    sm_version = _gpu_sm_version()
    skip_timing = pytest.mark.skip(
        reason="timing_serial tests skipped by default; run with: pytest tests -m timing_serial -n 0"
    )

    # If the user passed -m that includes timing_serial, don't auto-skip them.
    markexpr = config.getoption("-m", default="")
    timing_selected = "timing_serial" in markexpr

    for item in items:
        if sm_version < 100 and any(item.iter_markers(name="requires_cutile")):
            item.add_marker(
                pytest.mark.skip(
                    reason=f"cuTile requires sm_100+ (detected sm_{sm_version})"
                )
            )
        if "timing_serial" in item.keywords and not timing_selected:
            item.add_marker(skip_timing)


@pytest.fixture
def tmp_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated temporary cache directory for each test.

    Sets SOLEXECBENCH_CACHE_PATH so every builder writes build artifacts into a
    fresh temp directory, preventing pollution between tests.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SOLEXECBENCH_CACHE_PATH", str(cache_dir))
    return cache_dir
