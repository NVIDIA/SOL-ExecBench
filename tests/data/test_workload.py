import sys

import pytest

from sol_execbench.data import (
    CustomInput,
    RandomInput,
    SafetensorsInput,
    ScalarInput,
    Workload,
)


def test_random_input():
    inp = RandomInput()
    assert inp.type == "random"


def test_scalar_input():
    # Integer scalar
    inp = ScalarInput(value=42)
    assert inp.type == "scalar"
    assert inp.value == 42

    # Float scalar
    inp2 = ScalarInput(value=3.14)
    assert inp2.value == pytest.approx(3.14)

    # Boolean scalar
    inp3 = ScalarInput(value=True)
    assert inp3.value is True

    # Non-scalar types are rejected
    with pytest.raises(ValueError):
        ScalarInput(value="not_a_scalar")  # type: ignore[arg-type]


def test_safetensors_input():
    inp = SafetensorsInput(path="data/weights.safetensors", tensor_key="hidden_states")
    assert inp.type == "safetensors"
    assert inp.path == "data/weights.safetensors"
    assert inp.tensor_key == "hidden_states"

    # Empty path is rejected
    with pytest.raises(ValueError):
        SafetensorsInput(path="", tensor_key="k")

    # Empty tensor_key is rejected
    with pytest.raises(ValueError):
        SafetensorsInput(path="data/file.safetensors", tensor_key="")


def test_workload_valid():
    w = Workload(axes={"M": 4}, inputs={"A": RandomInput()}, uuid="w1")
    assert w.axes == {"M": 4}
    assert w.uuid == "w1"


def test_workload_zero_axis_allowed():
    # NonNegativeInt allows 0
    Workload(axes={"M": 0}, inputs={}, uuid="w_zero")


def test_workload_negative_axis_rejected():
    with pytest.raises(ValueError):
        Workload(axes={"M": -1}, inputs={"A": RandomInput()}, uuid="w_bad")


def test_workload_empty_uuid_rejected():
    with pytest.raises(ValueError):
        Workload(axes={"M": 4}, inputs={"A": RandomInput()}, uuid="")


def test_workload_all_input_types():
    w = Workload(
        axes={"M": 8},
        inputs={
            "tensor_rand": RandomInput(),
            "tensor_safe": SafetensorsInput(path="f.safetensors", tensor_key="k"),
            "scale": ScalarInput(value=2.0),
        },
        uuid="mixed",
    )
    assert w.inputs["tensor_rand"].type == "random"
    assert w.inputs["tensor_safe"].type == "safetensors"
    assert w.inputs["scale"].type == "scalar"


def test_workload_multiple_axes():
    w = Workload(
        axes={"batch": 16, "seq_len": 512, "hidden": 4096},
        inputs={"x": RandomInput()},
        uuid="multi_axis",
    )
    assert w.axes["batch"] == 16
    assert w.axes["seq_len"] == 512
    assert w.axes["hidden"] == 4096


def test_custom_input():
    inp = CustomInput()
    assert inp.type == "custom"


def test_workload_all_custom_inputs():
    w = Workload(
        axes={"M": 8},
        inputs={"A": CustomInput(), "B": CustomInput()},
        uuid="all_custom",
    )
    assert w.inputs["A"].type == "custom"
    assert w.inputs["B"].type == "custom"


def test_workload_mixed_custom_and_non_custom_rejected():
    with pytest.raises(ValueError, match="custom and non-custom"):
        Workload(
            axes={"M": 8},
            inputs={"A": CustomInput(), "B": RandomInput()},
            uuid="mixed_bad",
        )


def test_workload_mixed_custom_and_scalar_rejected():
    with pytest.raises(ValueError, match="custom and non-custom"):
        Workload(
            axes={"M": 8},
            inputs={"A": CustomInput(), "scale": ScalarInput(value=1.0)},
            uuid="mixed_bad2",
        )


def test_workload_mixed_custom_and_safetensors_rejected():
    with pytest.raises(ValueError, match="custom and non-custom"):
        Workload(
            axes={"M": 8},
            inputs={
                "A": CustomInput(),
                "B": SafetensorsInput(path="f.safetensors", tensor_key="k"),
            },
            uuid="mixed_bad3",
        )


def test_workload_empty_inputs_with_custom_allowed():
    """A workload with no inputs at all is fine (no mixing)."""
    w = Workload(axes={"M": 4}, inputs={}, uuid="empty")
    assert w.inputs == {}


def test_workload_single_custom_input():
    w = Workload(
        axes={"M": 4},
        inputs={"x": CustomInput()},
        uuid="single_custom",
    )
    assert len(w.inputs) == 1
    assert w.inputs["x"].type == "custom"


if __name__ == "__main__":
    pytest.main(sys.argv)
