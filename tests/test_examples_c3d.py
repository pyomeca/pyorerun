"""
Tests for c3d examples.
Run each example script to ensure they execute without errors.
"""

import pytest
from .utils_examples import ExampleRunner


def test_c3d_main():
    """
    Test the main.py C3D example.
    """
    example_path = ExampleRunner.example_folder() / "c3d" / "main.py"
    assert example_path.exists(), f"Example file not found: {example_path}"

    # Run the example
    ExampleRunner.run_example_module(example_path)


def test_c3d_gait():
    """
    Test the gait.py C3D example.
    """
    example_path = ExampleRunner.example_folder() / "c3d" / "gait.py"
    assert example_path.exists(), f"Example file not found: {example_path}"

    # Run the example
    ExampleRunner.run_example_module(example_path)


def test_c3d_running_gait():
    """
    Test the running_gait.py C3D example.
    """
    example_path = ExampleRunner.example_folder() / "c3d" / "running_gait.py"
    assert example_path.exists(), f"Example file not found: {example_path}"

    # Run the example
    ExampleRunner.run_example_module(example_path)
