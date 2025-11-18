"""
Tests for biorbd examples.
Run each example script to ensure they execute without errors.
"""

from .utils_examples import ExampleRunner


def test_main_double_pendulum():
    """
    Test the main.py example with double pendulum (using animate function).
    """
    example_path = ExampleRunner.example_folder() / "biorbd" / "main.py"
    assert example_path.exists(), f"Example file not found: {example_path}"

    # Run the example
    ExampleRunner.run_example_module(example_path)


def test_from_biorbd_model():
    """
    Test loading a biorbd model from biorbd object.
    """
    example_path = ExampleRunner.example_folder() / "biorbd" / "from_biorbd_model.py"
    assert example_path.exists(), f"Example file not found: {example_path}"

    # Run the example
    ExampleRunner.run_example_module(example_path)


def test_gait_reconstruction():
    """
    Test the gait reconstruction example.
    """
    example_path = ExampleRunner.example_folder() / "biorbd" / "gait_reconstruction.py"
    assert example_path.exists(), f"Example file not found: {example_path}"

    # Run the example
    ExampleRunner.run_example_module(example_path)


def test_marker_tracking():
    """
    Test the marker tracking example.
    """
    example_path = ExampleRunner.example_folder() / "biorbd" / "marker_tracking.py"
    assert example_path.exists(), f"Example file not found: {example_path}"

    # Run the example
    ExampleRunner.run_example_module(example_path)


def test_meshline_model():
    """
    Test the meshline model example.
    """
    example_path = ExampleRunner.example_folder() / "biorbd" / "meshline_model.py"
    assert example_path.exists(), f"Example file not found: {example_path}"

    # Run the example
    ExampleRunner.run_example_module(example_path)


def test_msk_model():
    """
    Test the MSK model with persistent markers example.
    """
    example_path = ExampleRunner.example_folder() / "biorbd" / "msk_model.py"
    assert example_path.exists(), f"Example file not found: {example_path}"

    # Run the example
    ExampleRunner.run_example_module(example_path)


def test_multi_models():
    """
    Test the multi-models example.
    """
    example_path = ExampleRunner.example_folder() / "biorbd" / "multi_models.py"
    assert example_path.exists(), f"Example file not found: {example_path}"

    # Run the example
    ExampleRunner.run_example_module(example_path)


def test_no_mesh():
    """
    Test the no mesh example.
    """
    example_path = ExampleRunner.example_folder() / "biorbd" / "no_mesh.py"
    assert example_path.exists(), f"Example file not found: {example_path}"

    # Run the example
    ExampleRunner.run_example_module(example_path)
