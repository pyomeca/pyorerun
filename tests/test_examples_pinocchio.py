"""
Tests for Pinocchio examples.
Run each example script to ensure they execute without errors.
"""

import pytest
from .utils_examples import ExampleRunner


@pytest.mark.skipif(
    not pytest.importorskip("pinocchio", reason="pinocchio is not installed"),
    reason="pinocchio module not available",
)
def test_from_pinocchio_model():
    """
    Test the Pinocchio model example with the Baxter robot URDF.
    """
    example_path = ExampleRunner.example_folder() / "pinocchio" / "from_pinocchio_model.py"
    assert example_path.exists(), f"Example file not found: {example_path}"

    # Run the example
    ExampleRunner.run_example_module(example_path)
