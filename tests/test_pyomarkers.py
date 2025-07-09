"""
Tests for the custom PyoMarkers class.
"""

import numpy as np
import pytest
from pyorerun.pyomarkers import PyoMarkers


def test_pyomarkers_basic_init():
    """Test basic initialization of PyoMarkers."""
    # Create test data with shape (3, 2, 5) - 2 markers, 5 frames
    data = np.random.rand(3, 2, 5)

    markers = PyoMarkers(data)

    # Check shape - should be (4, 2, 5) after adding homogeneous coordinate
    assert markers.shape == (4, 2, 5)

    # Check data integrity - first 3 dimensions should match input
    np.testing.assert_array_equal(markers.to_numpy()[:3, :, :], data)

    # Check homogeneous coordinate (should be all ones)
    np.testing.assert_array_equal(markers.to_numpy()[3, :, :], np.ones((2, 5)))

    # Check default marker names
    assert markers.marker_names == ["marker_0", "marker_1"]

    # Check channel compatibility
    assert hasattr(markers, "channel")
    np.testing.assert_array_equal(markers.channel.values, ["marker_0", "marker_1"])

    # Check default time vector
    np.testing.assert_array_equal(markers.time, np.arange(5, dtype=float))

    # Check attrs
    assert isinstance(markers.attrs, dict)


def test_pyomarkers_with_custom_params():
    """Test PyoMarkers with custom parameters."""
    data = np.random.rand(3, 3, 10)
    time = np.linspace(0, 1, 10)
    marker_names = ["head", "shoulder", "elbow"]
    attrs = {"units": "m", "rate": 100}

    markers = PyoMarkers(data=data, time=time, marker_names=marker_names, show_labels=False, attrs=attrs)

    assert markers.shape == (4, 3, 10)
    assert markers.marker_names == marker_names
    np.testing.assert_array_equal(markers.time, time)
    assert markers.attrs == attrs
    assert markers.show_labels == False
    np.testing.assert_array_equal(markers.channel.values, marker_names)


def test_pyomarkers_4d_input():
    """Test PyoMarkers with 4D input data."""
    data = np.random.rand(4, 2, 5)

    markers = PyoMarkers(data)

    # Should preserve the input data as-is
    assert markers.shape == (4, 2, 5)
    np.testing.assert_array_equal(markers.to_numpy(), data)


def test_pyomarkers_validation_errors():
    """Test validation errors in PyoMarkers."""
    # Wrong number of dimensions
    with pytest.raises(ValueError, match="Data must be 3D array"):
        PyoMarkers(np.random.rand(3, 5))  # 2D array

    # Wrong first dimension size
    with pytest.raises(ValueError, match="First dimension must be 3 or 4"):
        PyoMarkers(np.random.rand(5, 2, 3))  # 5D first dimension

    # Mismatched marker names
    data = np.random.rand(3, 2, 5)
    with pytest.raises(ValueError, match="Number of marker names must match"):
        PyoMarkers(data, marker_names=["one", "two", "three"])  # 3 names for 2 markers

    # Mismatched time vector
    with pytest.raises(ValueError, match="Time vector length must match"):
        PyoMarkers(data, time=np.arange(3))  # 3 time points for 5 frames


def test_compatibility_with_existing_usage():
    """Test compatibility with existing usage patterns from the codebase."""
    # Create data similar to existing tests
    data = np.array([[[1000, 2000]], [[3000, 4000]], [[5000, 6000]]], dtype=float)
    markers = PyoMarkers(data)
    markers.attrs["units"] = "mm"

    # Test shape access patterns
    assert markers.shape[1] == 1  # number of markers
    assert markers.shape[2] == 2  # number of frames

    # Test to_numpy access pattern
    result = markers.to_numpy()[:3, :, :]
    np.testing.assert_array_equal(result, data)

    # Test channel.values access pattern
    marker_names_list = markers.channel.values.tolist()
    assert isinstance(marker_names_list, list)
    assert len(marker_names_list) == 1

    # Test attrs access pattern
    assert markers.attrs["units"] == "mm"


def test_max_xy_coordinate_span_compatibility():
    """Test compatibility with max_xy_coordinate_span function pattern."""
    # Create test data that matches the existing test pattern
    data = np.array([[[1, 2], [-1, -2]], [[0.5, 1], [-0.5, -1]], [[0, 0], [0, 0]]])  # X, Y, Z coords
    markers = PyoMarkers(data)

    # Test the actual function logic from max_xy_coordinate_span_by_markers
    marker_data = markers.to_numpy()
    min_markers = np.nanmin(np.nanmin(marker_data, axis=2), axis=1)
    max_markers = np.nanmax(np.nanmax(marker_data, axis=2), axis=1)
    x_absolute_max = np.nanmax(np.abs([min_markers[0], max_markers[0]]))
    y_absolute_max = np.nanmax(np.abs([min_markers[1], max_markers[1]]))
    result = np.max([x_absolute_max, y_absolute_max])

    assert result == 2.0  # Same as the original test expectation
