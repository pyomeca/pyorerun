import numpy as np
import pytest
from pyomeca import Markers as PyoMarkers

from pyorerun.rrc3d import (
    max_xy_coordinate_span_by_markers,
    adjust_position_unit_to_meters,
    adjust_pyomarkers_unit_to_meters,
    down_sample_force
)


def test_max_xy_coordinate_span():
    # Create mock PyoMarkers data
    data = np.array([
        [[1, 2], [-1, -2]], # X coords
        [[0.5, 1], [-0.5, -1]], # Y coords 
        [[0, 0], [0, 0]] # Z coords
    ])
    markers = PyoMarkers(data)
    
    max_span = max_xy_coordinate_span_by_markers(markers)
    assert max_span == 2.0


def test_adjust_position_unit():
    # Test mm to m
    arr = np.array([1000, 2000, 3000], dtype=float)
    adjusted = adjust_position_unit_to_meters(arr, "mm")
    np.testing.assert_array_equal(adjusted, [1, 2, 3])
    
    # Test cm to m
    arr = np.array([100, 200, 300], dtype=float)
    adjusted = adjust_position_unit_to_meters(arr, "cm")
    np.testing.assert_array_equal(adjusted, [1, 2, 3])
    
    # Test m to m
    arr = np.array([1, 2, 3], dtype=float)
    adjusted = adjust_position_unit_to_meters(arr, "m")
    np.testing.assert_array_equal(adjusted, [1, 2, 3])
    
    # Test invalid unit
    with pytest.raises(ValueError):
        adjust_position_unit_to_meters(arr, "invalid")


def test_adjust_pyomarkers_unit():
    data = np.array([[[1000, 2000]], [[3000, 4000]], [[5000, 6000]]], dtype=float)
    markers = PyoMarkers(data)
    markers.attrs["units"] = "mm"
    
    adjusted = adjust_pyomarkers_unit_to_meters(markers, "mm")
    expected = data / 1000
    np.testing.assert_array_equal(adjusted.to_numpy()[:3,:,:], expected)
    assert adjusted.attrs["units"] == "m"


def test_down_sample_force():
    # Mock force data with 2x the marker sampling rate
    force = {
        "force": np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]], dtype=float),
        "center_of_pressure": np.array([[10,20,30,40], [50,60,70,80], [90,100,110,120]], dtype=float)
    }
    t_span = np.array([0, 1])
    
    cop, force_vec = down_sample_force(force, t_span, "mm")
    
    expected_cop = np.array([[0.01, 0.03], [0.05, 0.07], [0.09, 0.11]])
    expected_force = np.array([[1,3], [5,7], [9,11]])
    
    np.testing.assert_array_equal(cop, expected_cop)
    np.testing.assert_array_equal(force_vec, expected_force)
    
    # Test non-integer ratio raises error
    t_span = np.array([0, 1, 2]) # 3 points vs 4 force samples
    with pytest.raises(NotImplementedError):
        down_sample_force(force, t_span, "mm")

