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


"""Tests for sol_execbench.core.bench.reward_hack."""

import pytest
import torch

from sol_execbench.core.bench.reward_hack import (
    RewardHackDetected,
    check_eval_integrity,
    check_lazy_outputs,
    check_monkey_patch,
    check_thread_injection,
    check_timed_outputs,
    snapshot_critical_functions,
)
from sol_execbench.core.data.workload import ToleranceSpec

# ── check_monkey_patch ────────────────────────────────────────────────────────


class TestCheckMonkeyPatch:
    def test_passes_when_not_patched(self):
        """No exception when elapsed_time identity is unchanged."""
        check_monkey_patch()  # must not raise

    def test_raises_when_elapsed_time_replaced(self, monkeypatch):
        """RewardHackDetected when elapsed_time identity differs from the captured address."""
        from sol_execbench.core.bench import reward_hack

        if reward_hack._ELAPSED_TIME_ADDR is None:
            pytest.skip(
                "CUDA not available; _ELAPSED_TIME_ADDR was not captured at import"
            )

        monkeypatch.setattr(torch.cuda.Event, "elapsed_time", lambda self, other: 0.0)
        with pytest.raises(RewardHackDetected, match="monkey-patched"):
            check_monkey_patch()


# ── check_thread_injection ────────────────────────────────────────────────────


class TestCheckThreadInjection:
    def test_passes_when_count_unchanged(self):
        check_thread_injection(5, 5)

    def test_passes_when_count_decreases(self):
        check_thread_injection(5, 3)

    def test_raises_when_count_increases(self):
        with pytest.raises(RewardHackDetected, match="Thread injection"):
            check_thread_injection(3, 5)

    def test_error_message_includes_both_counts(self):
        with pytest.raises(RewardHackDetected) as exc:
            check_thread_injection(2, 7)
        msg = str(exc.value)
        assert "7" in msg and "2" in msg


# ── check_lazy_outputs ────────────────────────────────────────────────────────


class TestCheckLazyOutputs:
    def test_passes_for_real_tensors(self):
        check_lazy_outputs([torch.zeros(2), torch.ones(3)])

    def test_passes_for_empty_list(self):
        check_lazy_outputs([])

    def test_raises_for_non_tensor(self):
        with pytest.raises(RewardHackDetected, match="Lazy evaluation"):
            check_lazy_outputs([torch.zeros(2), 42])

    def test_raises_for_none(self):
        with pytest.raises(RewardHackDetected, match="NoneType"):
            check_lazy_outputs([None])

    def test_raises_for_tensor_subclass(self):
        """Tensor subclasses (e.g. FakeTensor views) are rejected by strict type() check."""

        class SubTensor(torch.Tensor):
            pass

        base = torch.zeros(2)
        sub = base.as_subclass(SubTensor)
        with pytest.raises(RewardHackDetected):
            check_lazy_outputs([sub])

    def test_error_message_includes_actual_type_name(self):
        with pytest.raises(RewardHackDetected) as exc:
            check_lazy_outputs(["a string"])
        assert "str" in str(exc.value)


# ── snapshot_critical_functions / check_eval_integrity ───────────────────────


class TestCheckTimedOutputs:
    TOL = ToleranceSpec()

    def test_passes_when_outputs_match(self):
        ref = [torch.ones(16), torch.zeros(16)]
        got = [torch.ones(16), torch.zeros(16)]
        check_timed_outputs(got, ref, self.TOL)  # must not raise

    def test_passes_within_tolerance(self):
        ref = [torch.ones(16)]
        got = [torch.ones(16) + 1e-6]
        check_timed_outputs(got, ref, self.TOL)  # must not raise

    def test_raises_when_values_diverge(self):
        """The kernel stopped computing once timing began."""
        ref = [torch.ones(16)]
        got = [torch.zeros(16)]
        with pytest.raises(RewardHackDetected, match="timing began"):
            check_timed_outputs(got, ref, self.TOL)

    def test_raises_on_output_count_mismatch(self):
        with pytest.raises(RewardHackDetected, match="expected 2"):
            check_timed_outputs([torch.ones(4)], [torch.ones(4)] * 2, self.TOL)

    def test_raises_on_shape_change(self):
        with pytest.raises(RewardHackDetected, match="changed from"):
            check_timed_outputs([torch.ones(8)], [torch.ones(16)], self.TOL)

    def test_raises_on_dtype_change(self):
        with pytest.raises(RewardHackDetected, match="changed from"):
            check_timed_outputs(
                [torch.ones(16, dtype=torch.float16)],
                [torch.ones(16, dtype=torch.float32)],
                self.TOL,
            )

    def test_raises_on_non_tensor_output(self):
        with pytest.raises(RewardHackDetected, match="not a Tensor"):
            check_timed_outputs([None], [torch.ones(16)], self.TOL)

    def test_error_message_reports_output_index(self):
        ref = [torch.ones(4), torch.ones(4)]
        got = [torch.ones(4), torch.zeros(4)]
        with pytest.raises(RewardHackDetected, match="output 1"):
            check_timed_outputs(got, ref, self.TOL)


# ── check_eval_integrity ──────────────────────────────────────────────────────


class TestEvalIntegrity:
    def test_snapshot_captures_present_names(self):
        ns = {"foo": lambda: None, "bar": 42}
        snap = snapshot_critical_functions(ns, ["foo", "bar", "missing"])
        assert "foo" in snap and "bar" in snap
        assert "missing" not in snap

    def test_passes_when_namespace_unchanged(self):
        fn_a = lambda: None  # noqa: E731
        fn_b = lambda: None  # noqa: E731
        ns = {"fn_a": fn_a, "fn_b": fn_b}
        snap = snapshot_critical_functions(ns, ["fn_a", "fn_b"])
        check_eval_integrity(snap, ns)  # must not raise

    def test_raises_when_function_replaced(self):
        original = lambda: None  # noqa: E731
        ns = {"time_runnable": original}
        snap = snapshot_critical_functions(ns, ["time_runnable"])
        ns["time_runnable"] = lambda *a, **kw: 0.001  # attacker replaces it
        with pytest.raises(RewardHackDetected, match="time_runnable.*monkey-patched"):
            check_eval_integrity(snap, ns)

    def test_raises_when_function_deleted(self):
        ns = {"compute_error_stats": lambda: None}
        snap = snapshot_critical_functions(ns, ["compute_error_stats"])
        del ns["compute_error_stats"]
        with pytest.raises(RewardHackDetected, match="compute_error_stats"):
            check_eval_integrity(snap, ns)

    def test_reports_first_violated_name(self):
        fn = lambda: None  # noqa: E731
        ns = {"a": fn, "b": fn}
        snap = snapshot_critical_functions(ns, ["a", "b"])
        ns["b"] = lambda: None  # replace only b
        with pytest.raises(RewardHackDetected, match="'b'"):
            check_eval_integrity(snap, ns)

    def test_empty_snapshot_always_passes(self):
        check_eval_integrity({}, {"anything": 123})

    def test_simulates_time_runnable_attack(self):
        """End-to-end: mimics the kernel.py attack via sys.modules['__main__']."""

        def real_time_runnable(fn, args, device, warmup=10, rep=50):
            return 1.0

        ns = {"time_runnable": real_time_runnable}
        snap = snapshot_critical_functions(ns, ["time_runnable"])

        # Simulate the attack: user code wraps time_runnable
        orig = ns["time_runnable"]

        def patched(fn, args, device, warmup=None, **kwargs):
            return orig(fn, args, device)

        ns["time_runnable"] = patched

        with pytest.raises(RewardHackDetected, match="time_runnable"):
            check_eval_integrity(snap, ns)
