"""
Tests for biobuddy examples.
Run each example script to ensure they execute without errors.
"""

import pytest
from .utils_examples import ExampleRunner


@pytest.mark.skipif(
    not pytest.importorskip("biobuddy", reason="biobuddy is not installed"),
    reason="biobuddy module not available",
)
def test_from_biobuddy_model():
    """
    Test the biobuddy example script.
    """
    example_path = ExampleRunner.example_folder() / "biobuddy" / "from_biobuddy_model.py"
    assert example_path.exists(), f"Example file not found: {example_path}"

    # Run the example
    ExampleRunner.run_example_module(example_path)
