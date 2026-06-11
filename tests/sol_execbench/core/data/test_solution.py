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


"""Tests for sol_execbench.core.data.solution — language and entry point validation."""

import pytest
from pydantic import ValidationError

from sol_execbench.core.data.solution import BuildSpec, SupportedLanguages


def _make_spec(**overrides):
    base = dict(
        languages=["triton"],
        target_hardware=["LOCAL"],
        entry_point="kernel.py::run",
    )
    base.update(overrides)
    return BuildSpec(**base)


# ── _validate_languages — mixing ─────────────────────────────────────────────


class TestLanguageMixingValidation:
    """BuildSpec must reject specs that mix C++ and Python languages."""

    # -- pure Python sets (should all pass) --

    @pytest.mark.parametrize(
        "langs",
        [
            ["pytorch"],
            ["triton"],
            ["tlx"],
            ["cute_dsl"],
            ["cutile"],
            ["cudnn_frontend"],
            ["pytorch", "triton"],
            ["pytorch", "triton", "tlx"],
            ["pytorch", "triton", "tlx", "cute_dsl", "cutile", "cudnn_frontend"],
        ],
    )
    def test_pure_python_languages_accepted(self, langs):
        spec = _make_spec(languages=langs)
        assert set(spec.languages) == {SupportedLanguages(lg) for lg in langs}

    # -- pure C++ sets (should all pass) --

    @pytest.mark.parametrize(
        "langs",
        [
            ["cuda_cpp"],
            ["cutlass"],
            ["cudnn"],
            ["cublas"],
            ["cuda_cpp", "cutlass"],
            ["cuda_cpp", "cutlass", "cudnn", "cublas"],
        ],
    )
    def test_pure_cpp_languages_accepted(self, langs):
        spec = _make_spec(languages=langs, entry_point="kernel.cu::run")
        assert set(spec.languages) == {SupportedLanguages(lg) for lg in langs}

    # -- mixed sets (should all fail) --

    @pytest.mark.parametrize(
        "langs",
        [
            ["pytorch", "cuda_cpp"],
            ["triton", "cutlass"],
            ["tlx", "cuda_cpp"],
            ["cute_dsl", "cudnn"],
            ["cutile", "cublas"],
            ["cudnn_frontend", "cuda_cpp"],
            ["pytorch", "triton", "cuda_cpp"],
            ["cuda_cpp", "cutlass", "pytorch"],
        ],
    )
    def test_mixed_languages_rejected(self, langs):
        with pytest.raises(ValidationError, match="C\\+\\+ and Python cannot be mixed"):
            _make_spec(languages=langs, entry_point="kernel.cu::run")

    # -- single language (every enum value should pass alone) --

    @pytest.mark.parametrize("lang", [lg.value for lg in SupportedLanguages])
    def test_every_language_accepted_alone(self, lang):
        ext = ".cu" if lang in ("cuda_cpp", "cutlass", "cudnn", "cublas") else ".py"
        spec = _make_spec(languages=[lang], entry_point=f"kernel{ext}::run")
        assert spec.languages == [SupportedLanguages(lang)]


# ── _validate_languages — entry point suffix ─────────────────────────────────


class TestEntryPointSuffixValidation:
    """BuildSpec must reject entry points whose suffix doesn't match the language category."""

    # -- Python languages with wrong suffix --

    @pytest.mark.parametrize(
        "lang", ["pytorch", "triton", "tlx", "cute_dsl", "cutile", "cudnn_frontend"]
    )
    def test_python_language_rejects_cu_entry(self, lang):
        with pytest.raises(ValidationError, match="require a .py entry point"):
            _make_spec(languages=[lang], entry_point="kernel.cu::run")

    @pytest.mark.parametrize(
        "lang", ["pytorch", "triton", "tlx", "cute_dsl", "cutile", "cudnn_frontend"]
    )
    def test_python_language_rejects_cpp_entry(self, lang):
        with pytest.raises(ValidationError, match="require a .py entry point"):
            _make_spec(languages=[lang], entry_point="kernel.cpp::run")

    # -- C++ languages with wrong suffix --

    @pytest.mark.parametrize("lang", ["cuda_cpp", "cutlass", "cudnn", "cublas"])
    def test_cpp_language_rejects_py_entry(self, lang):
        with pytest.raises(ValidationError, match="require a C\\+\\+/CUDA entry point"):
            _make_spec(languages=[lang], entry_point="kernel.py::run")

    # -- C++ languages with valid suffixes --

    @pytest.mark.parametrize(
        "suffix", [".cu", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".cuh"]
    )
    def test_cpp_language_accepts_valid_suffixes(self, suffix):
        spec = _make_spec(languages=["cuda_cpp"], entry_point=f"kernel{suffix}::run")
        assert spec.languages == [SupportedLanguages.CUDA_CPP]

    # -- Python languages with valid suffix --

    @pytest.mark.parametrize(
        "lang", ["pytorch", "triton", "tlx", "cute_dsl", "cutile", "cudnn_frontend"]
    )
    def test_python_language_accepts_py_entry(self, lang):
        spec = _make_spec(languages=[lang], entry_point="kernel.py::run")
        assert spec.languages == [SupportedLanguages(lang)]
