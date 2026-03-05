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

"""Defines the environment variables used in sol_execbench evaluate."""

import os
from pathlib import Path


def get_sol_execbench_dataset_path() -> Path:
    """Get the value of the SOLEXECBENCH_DATASET_PATH environment variable. It controls the path to the
    dataset to dump or to load.

    Returns
    -------
    Path
        The value of the SOLEXECBENCH_DATASET_PATH environment variable.
    """
    value = os.environ.get("SOLEXECBENCH_DATASET_PATH")
    if value:
        return Path(value).expanduser()
    return Path(Path.home() / ".cache" / "sol_execbench" / "dataset")


def get_sol_execbench_cache_path() -> Path:
    """Get the value of the SOLEXECBENCH_CACHE_PATH environment variable. It controls the path to the cache.

    Returns
    -------
    Path
        The value of the SOLEXECBENCH_CACHE_PATH environment variable.
    """
    value = os.environ.get("SOLEXECBENCH_CACHE_PATH")
    if value:
        return Path(value).expanduser()
    return Path.home() / ".cache" / "sol_execbench" / "cache"
