import sys

import pytest

from sol_execbench import AxisConst, AxisVar, Definition, TensorSpec


@pytest.fixture
def sample_reference_code():
    # Minimal valid reference function matching make_minimal_definition's single input "A"
    return "def run(A):\n    return A\n"


def make_minimal_definition(ref_code: str) -> Definition:
    return Definition(
        name="def1",
        op_type="op",
        axes={"M": AxisVar(), "N": AxisConst(value=16)},
        inputs={"A": TensorSpec(shape=["M", "N"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference=ref_code,
    )


def test_axisconst_valid_and_invalid():
    AxisConst(value=1)
    # Zero is allowed
    AxisConst(value=0)
    with pytest.raises(ValueError):
        AxisConst(value=-3)


def test_axisvar_basic():
    ax = AxisVar()
    assert ax.type == "var"


def test_tensorspec_validation():
    TensorSpec(shape=["M"], dtype="int32")
    # shape must be a list, not a bare string
    with pytest.raises(ValueError):
        TensorSpec(shape="M", dtype="int32")  # type: ignore[arg-type]
    # dtype must be a valid DType value
    with pytest.raises(ValueError):
        TensorSpec(shape=["M"], dtype="not_a_dtype")  # type: ignore[arg-type]


def test_tensorspec_scalar():
    # shape=None represents a scalar input
    spec = TensorSpec(shape=None, dtype="float32")
    assert spec.shape is None


def test_definition_basic_validation(sample_reference_code):
    definition = make_minimal_definition(sample_reference_code)
    assert definition.name == "def1"
    assert set(definition.const_axes.keys()) == {"N"}
    assert definition.const_axes["N"] == 16
    assert set(definition.var_axes) == {"M"}


def test_definition_axis_reference_checks(sample_reference_code):
    # Input referencing undefined axis should raise
    with pytest.raises(ValueError):
        Definition(
            name="bad",
            op_type="op",
            axes={"M": AxisVar()},
            inputs={"A": TensorSpec(shape=["X"], dtype="float32")},  # X not defined
            outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
            reference=sample_reference_code,
        )


def test_definition_reference_must_define_run():
    with pytest.raises(ValueError):
        make_minimal_definition("def not_run():\n    pass\n")
    with pytest.raises(ValueError):
        make_minimal_definition("def run(:\n    pass\n")  # invalid syntax


def test_definition_run_inputs_match_valid():
    # run() params match inputs exactly
    Definition(
        name="ok",
        op_type="op",
        axes={"M": AxisVar()},
        inputs={
            "x": TensorSpec(shape=["M"], dtype="float32"),
            "y": TensorSpec(shape=["M"], dtype="float32"),
        },
        outputs={"z": TensorSpec(shape=["M"], dtype="float32")},
        reference="def run(x, y):\n    return x + y\n",
    )


def test_definition_run_inputs_wrong_count():
    with pytest.raises(ValueError, match="run\\(\\)"):
        Definition(
            name="bad_count",
            op_type="op",
            axes={"M": AxisVar()},
            inputs={"x": TensorSpec(shape=["M"], dtype="float32")},
            outputs={"z": TensorSpec(shape=["M"], dtype="float32")},
            # run() has 2 params but inputs only has 1
            reference="def run(x, y):\n    return x + y\n",
        )


def test_definition_run_inputs_wrong_names():
    with pytest.raises(ValueError, match="run\\(\\)"):
        Definition(
            name="bad_names",
            op_type="op",
            axes={"M": AxisVar()},
            inputs={"x": TensorSpec(shape=["M"], dtype="float32")},
            outputs={"z": TensorSpec(shape=["M"], dtype="float32")},
            # run() param is named "a" but input key is "x"
            reference="def run(a):\n    return a\n",
        )


def test_definition_run_inputs_positional_only_params():
    # Positional-only params (before /) should also match input names
    Definition(
        name="posonly",
        op_type="op",
        axes={"M": AxisVar()},
        inputs={"x": TensorSpec(shape=["M"], dtype="float32")},
        outputs={"z": TensorSpec(shape=["M"], dtype="float32")},
        reference="def run(x, /):\n    return x\n",
    )


def test_definition_constraints(sample_reference_code):
    # Empty constraint string is invalid
    with pytest.raises(ValueError):
        Definition(
            name="definition",
            op_type="op",
            axes={"M": AxisVar()},
            inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
            outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
            reference=sample_reference_code,
            constraints=[""],
        )

    # Invalid Python expression in constraint
    with pytest.raises(ValueError):
        Definition(
            name="definition",
            op_type="op",
            axes={"M": AxisVar()},
            inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
            outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
            reference=sample_reference_code,
            constraints=["M >"],  # invalid expression
        )


def test_definition_input_output_names_no_overlap(sample_reference_code):
    with pytest.raises(ValueError):
        Definition(
            name="overlapping",
            op_type="op",
            axes={"M": AxisVar()},
            inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
            outputs={"A": TensorSpec(shape=["M"], dtype="float32")},  # same name
            reference=sample_reference_code,
        )


def test_definition_scalar_input(sample_reference_code):
    # Scalar inputs have shape=None
    Definition(
        name="scalar_def",
        op_type="op",
        axes={"M": AxisVar()},
        inputs={
            "A": TensorSpec(shape=["M"], dtype="float32"),
            "scale": TensorSpec(shape=None, dtype="float32"),
        },
        outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
        reference="def run(A, scale):\n    return A * scale\n",
    )


def test_get_input_shapes_with_scalar():
    definition = Definition(
        name="scalar_test",
        op_type="op",
        axes={"M": AxisVar(), "N": AxisConst(value=4)},
        inputs={
            "A": TensorSpec(shape=["M", "N"], dtype="float32"),
            "scale": TensorSpec(shape=None, dtype="float32"),
            "B": TensorSpec(shape=["M"], dtype="float32"),
        },
        outputs={"C": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference="def run(A, scale, B):\n    return A * scale + B\n",
    )

    shapes_dict = definition.get_input_shapes({"M": 8})
    _, shapes = zip(*shapes_dict.items())
    assert len(shapes) == len(definition.inputs)
    assert shapes[0] == (8, 4)  # A: [M, N]
    assert shapes[1] is None  # scale: scalar
    assert shapes[2] == (8,)  # B: [M]


class TestGetAxesValues:
    @pytest.fixture
    def definition(self):
        return Definition(
            name="test_def",
            op_type="op",
            axes={"M": AxisVar(), "N": AxisConst(value=4)},
            inputs={
                "A": TensorSpec(shape=["M", "N"], dtype="float32"),
                "B": TensorSpec(shape=["M"], dtype="float32"),
            },
            outputs={"C": TensorSpec(shape=["M", "N"], dtype="float32")},
            reference="def run(A, B):\n    return A\n",
        )

    def test_from_shapes(self, definition):
        shapes = [(8, 4), (8,)]
        result = definition.get_axes_values(shapes)
        assert result == {"M": 8}

    def test_inconsistent_axis_values_raises(self, definition):
        shapes = [(8, 4), (10,)]  # M is 8 in A but 10 in B
        with pytest.raises(ValueError):
            definition.get_axes_values(shapes)


class TestGetAxesValuesFromInputs:
    @pytest.fixture
    def definition(self):
        return Definition(
            name="test_def",
            op_type="op",
            axes={"M": AxisVar(), "N": AxisConst(value=4)},
            inputs={
                "A": TensorSpec(shape=["M", "N"], dtype="float32"),
                "B": TensorSpec(shape=["M"], dtype="float32"),
            },
            outputs={"C": TensorSpec(shape=["M", "N"], dtype="float32")},
            reference="def run(A, B):\n    return A\n",
        )

    def test_with_tensors(self, definition):
        import torch

        A = torch.zeros((8, 4))
        B = torch.zeros((8,))
        result = definition.get_axes_values_from_inputs([A, B])
        assert result == {"M": 8}

    def test_with_scalar_input(self, definition):
        scalar_def = Definition(
            name="scalar_def",
            op_type="op",
            axes={"M": AxisVar()},
            inputs={
                "A": TensorSpec(shape=["M"], dtype="float32"),
                "scale": TensorSpec(shape=None, dtype="float32"),
            },
            outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
            reference="def run(A, scale):\n    return A * scale\n",
        )
        import torch

        A = torch.zeros((5,))
        scale = 2.0
        result = scalar_def.get_axes_values_from_inputs([A, scale])
        assert result == {"M": 5}


class TestCustomInputsEntrypoint:
    def test_valid_entrypoint(self):
        ref = (
            "def generate_inputs(axes_and_scalars, device):\n"
            "    return {}\n"
            "def run(A):\n"
            "    return A\n"
        )
        d = Definition(
            name="custom",
            op_type="op",
            axes={"M": AxisVar()},
            inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
            outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
            reference=ref,
            custom_inputs_entrypoint="generate_inputs",
        )
        assert d.custom_inputs_entrypoint == "generate_inputs"

    def test_none_entrypoint_is_default(self, sample_reference_code):
        d = make_minimal_definition(sample_reference_code)
        assert d.custom_inputs_entrypoint is None

    def test_invalid_identifier_raises(self, sample_reference_code):
        with pytest.raises(ValueError, match="valid Python identifier"):
            Definition(
                name="bad",
                op_type="op",
                axes={"M": AxisVar()},
                inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
                outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
                reference=sample_reference_code,
                custom_inputs_entrypoint="not-an-identifier",
            )

    def test_entrypoint_not_defined_in_reference_raises(self, sample_reference_code):
        with pytest.raises(ValueError, match="not defined as a top-level function"):
            Definition(
                name="missing_fn",
                op_type="op",
                axes={"M": AxisVar()},
                inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
                outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
                reference=sample_reference_code,
                custom_inputs_entrypoint="nonexistent_func",
            )

    def test_entrypoint_must_be_nonempty(self):
        ref = "def run(A):\n    return A\n"
        with pytest.raises(ValueError):
            Definition(
                name="empty_ep",
                op_type="op",
                axes={"M": AxisVar()},
                inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
                outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
                reference=ref,
                custom_inputs_entrypoint="",
            )

    def test_entrypoint_cannot_be_run(self):
        """The entrypoint can be 'run' since it's always defined—but it should work."""
        ref = "def run(A):\n    return A\n"
        d = Definition(
            name="run_ep",
            op_type="op",
            axes={"M": AxisVar()},
            inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
            outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
            reference=ref,
            custom_inputs_entrypoint="run",
        )
        assert d.custom_inputs_entrypoint == "run"

    def test_entrypoint_nested_function_not_found(self):
        """Only top-level functions should be detected."""
        ref = (
            "def run(A):\n"
            "    def make_inputs(axes, device):\n"
            "        return {}\n"
            "    return A\n"
        )
        with pytest.raises(ValueError, match="not defined as a top-level function"):
            Definition(
                name="nested",
                op_type="op",
                axes={"M": AxisVar()},
                inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
                outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
                reference=ref,
                custom_inputs_entrypoint="make_inputs",
            )


if __name__ == "__main__":
    pytest.main(sys.argv)
