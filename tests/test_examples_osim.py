"""
Tests for OpenSim examples.
Run each example script to ensure they execute without errors.
"""

import pytest
from utils_examples import ExampleRunner


@pytest.mark.skipif(
    not pytest.importorskip("opensim", reason="opensim is not installed"),
    reason="opensim module not available",
)
def test_from_osim_model():
    """
    Test the OpenSim model example.
    """
    example_path = ExampleRunner.example_folder() / "osim" / "from_osim_model.py"
    assert example_path.exists(), f"Example file not found: {example_path}"

    # Run the example
    ExampleRunner.run_example_module(example_path)


def test_trc_reader():
    """
    Test the TRC reader example.
    """
    example_path = ExampleRunner.example_folder() / "osim" / "trc_reader.py"
    assert example_path.exists(), f"Example file not found: {example_path}"

    # Run the example
    ExampleRunner.run_example_module(example_path)
