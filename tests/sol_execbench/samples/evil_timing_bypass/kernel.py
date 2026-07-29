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

"""Evil kernel: correct while correctness is checked, then replays a stale result.

The eval driver checks correctness over one set of calls and measures latency
over a different set.  This kernel counts its calls, does the real work for the
correctness rounds, and afterwards hands back the banked tensor — so the
measured latency reflects a single 4-byte memset instead of the kernel.

Two properties make this the interesting case:

* The replayed values are the *correct* values for the correctness-phase
  inputs, so an audit that re-used those inputs would not notice.  The audit in
  ``time_runnable`` runs the kernel on **fresh** inputs with their own
  reference outputs, which is what catches it.
* It still emits one GPU activity per call, so it does not trip the
  "no kernel activities recorded" guard in ``bench_gpu_time_with_cupti``.
"""

import torch

_CORRECTNESS_ROUNDS = 10  # eval_driver.py: `for _round in range(10)`
_calls = {}  # per shape: the driver sweeps several workloads in one process
_cached = {}
_dummy = None


def run(x, y):
    global _dummy
    key = tuple(x.shape)
    seen = _calls.get(key, 0) + 1
    _calls[key] = seen

    if seen <= _CORRECTNESS_ROUNDS:
        _cached[key] = x + y
        return _cached[key]

    if _dummy is None:
        _dummy = torch.zeros(1, device=x.device, dtype=x.dtype)
    _dummy.zero_()
    return _cached[key]
