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

"""Unified JSON encoding/decoding utilities for Pydantic BaseModel objects."""

from pathlib import Path
from typing import Type, TypeVar, Union

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def save_json_file(object: BaseModel, path: Union[str, Path]) -> None:
    """
    Save a Pydantic BaseModel object to a JSON file.

    Parameters
    ----------
    object : BaseModel
        The Pydantic BaseModel instance to be serialized and saved.
    path : Union[str, Path]
        The file path where the JSON will be saved. Parent directories
        will be created if they don't exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(object.model_dump_json(indent=2, exclude_unset=True))


def load_json_file(model_cls: Type[T], path: Union[str, Path]) -> T:
    """
    Load a Pydantic BaseModel object from a JSON file.

    Parameters
    ----------
    model_cls : Type[BaseModel]
        The Pydantic BaseModel class to instantiate from the JSON data.
    path : Union[str, Path]
        The file path of the JSON file to load.

    Returns
    -------
    BaseModel
        An instance of the specified BaseModel class populated with
        data from the JSON file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValidationError
        If the JSON data doesn't match the BaseModel schema.
    """
    with open(Path(path), "r", encoding="utf-8") as f:
        return model_cls.model_validate_json(f.read())


def save_jsonl_file(objects: list[BaseModel], path: Union[str, Path]) -> None:
    """
    Save a list of Pydantic BaseModel objects to a JSONL file. Each object is serialized as a
    separate JSON object on its own line.

    Parameters
    ----------
    objects : list[BaseModel]
        A list of Pydantic BaseModel instances to be serialized and saved.
    path : Union[str, Path]
        The file path where the JSONL will be saved. Parent directories
        will be created if they don't exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        object_strs = [obj.model_dump_json(indent=None) for obj in objects]
        output_str = "\n".join(object_strs) + "\n"
        f.write(output_str)


def load_jsonl_file(model_cls: Type[T], path: Union[str, Path]) -> list[T]:
    """
    Load a list of Pydantic BaseModel objects from a JSONL file. Each line in the JSONL file should
    contain a valid JSON object that can be deserialized into the specified BaseModel class. Empty
    lines are skipped.

    Parameters
    ----------
    model_cls : Type[BaseModel]
        The Pydantic BaseModel class to instantiate for each JSON object.
    path : Union[str, Path]
        The file path of the JSONL file to load.

    Returns
    -------
    list[BaseModel]
        A list of instances of the specified BaseModel class, one for
        each valid JSON line in the file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValidationError
        If any JSON line doesn't match the BaseModel schema.
    """
    out = []
    with open(Path(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(model_cls.model_validate_json(line))
    return out


def append_jsonl_file(objects: list[BaseModel], path: Union[str, Path]) -> None:
    """
    Append a list of Pydantic BaseModel objects to a JSONL file. Each object is serialized as a
    separate JSON object and appended to the end of the file, one per line.

    Parameters
    ----------
    objects : list[BaseModel]
        A list of Pydantic BaseModel instances to be serialized and appended.
    path : Union[str, Path]
        The file path of the JSONL file to append to. Parent directories
        will be created if they don't exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    needs_newline_prefix = False
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            last_char = f.read(1)
            needs_newline_prefix = last_char != b"\n"

    with open(path, "a", encoding="utf-8") as f:
        if needs_newline_prefix:
            f.write("\n")

        object_strs = [obj.model_dump_json(indent=None) for obj in objects]
        output_str = "\n".join(object_strs) + "\n"
        f.write(output_str)
