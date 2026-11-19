"""
Tests for MarkersUpdater and PersistentMarkersUpdater classes.
"""

import sys
import os
import numpy as np
import pytest
import rerun as rr
import shutil

# Create an isolated test environment to avoid package import issues
test_dir = '/tmp/test_pyorerun'
os.makedirs(f"{test_dir}/pyorerun/abstract", exist_ok=True)
os.makedirs(f"{test_dir}/pyorerun/model_components", exist_ok=True) 
os.makedirs(f"{test_dir}/pyorerun/xp_components", exist_ok=True)

# Copy necessary files directly
src_dir = '/home/runner/work/pyorerun/pyorerun/pyorerun'
dest_dir = f'{test_dir}/pyorerun'

try:
    shutil.copy(f'{src_dir}/abstract/abstract_class.py', f'{dest_dir}/abstract/')
    shutil.copy(f'{src_dir}/abstract/markers.py', f'{dest_dir}/abstract/')
    shutil.copy(f'{src_dir}/model_components/model_markers.py', f'{dest_dir}/model_components/')
    shutil.copy(f'{src_dir}/xp_components/persistent_marker_options.py', f'{dest_dir}/xp_components/')
except Exception:
    pass  # Files might already exist

# Create empty __init__.py files
for init_file in [
    f'{dest_dir}/__init__.py',
    f'{dest_dir}/abstract/__init__.py',
    f'{dest_dir}/model_components/__init__.py',
    f'{dest_dir}/xp_components/__init__.py'
]:
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            pass

# Add to path
sys.path.insert(0, test_dir)

# Mock the rerun.log function to avoid actual logging during tests
def mock_log(*args, **kwargs):
    pass
rr.log = mock_log

# Now import what we need
from pyorerun.abstract.abstract_class import Component, PersistentComponent
from pyorerun.abstract.markers import MarkerProperties
from pyorerun.xp_components.persistent_marker_options import PersistentMarkerOptions
from pyorerun.model_components.model_markers import (
    MarkersUpdater, 
    PersistentMarkersUpdater,
    compute_markers,
    from_pyo_to_rerun
)


class TestMarkersUpdater:
    """Test cases for MarkersUpdater class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create mock marker properties
        self.marker_names = ["marker1", "marker2", "marker3"]
        self.marker_properties = MarkerProperties(
            marker_names=self.marker_names,
            radius=0.02,
            color=np.array([255, 0, 0]),
            show_labels=True
        )
        
        # Create a simple callable that returns marker positions
        def mock_callable_markers(q):
            # Return 3 markers with 3D positions based on input q
            # The function should return (nb_markers, 3) so when transposed it becomes (3, nb_markers)
            return np.array([
                [q[0], q[1], q[2]],      # marker1
                [q[0]+1, q[1]+1, q[2]+1],  # marker2  
                [q[0]+2, q[1]+2, q[2]+2]   # marker3
            ])
        
        self.callable_markers = mock_callable_markers
        
        # Create MarkersUpdater instance
        self.updater = MarkersUpdater(
            name="test",
            marker_properties=self.marker_properties,
            callable_markers=self.callable_markers
        )

    def test_init(self):
        """Test MarkersUpdater initialization."""
        assert self.updater.name == "test/model_markers"
        assert self.updater.marker_properties == self.marker_properties
        assert self.updater.callable_markers == self.callable_markers

    def test_nb_markers(self):
        """Test nb_markers property."""
        assert self.updater.nb_markers == 3

    def test_nb_components(self):
        """Test nb_components property."""
        assert self.updater.nb_components == 1

    def test_compute_markers(self):
        """Test compute_markers method."""
        # Test with single frame
        q = np.array([[1.0], [2.0], [3.0]])
        markers = self.updater.compute_markers(q)
        
        # Should return (3, nb_markers, nb_frames)
        assert markers.shape == (3, 3, 1)
        
        # Check values for first frame
        expected = np.array([
            [[1.0], [2.0], [3.0]],       # x-coords for markers 1,2,3
            [[2.0], [3.0], [4.0]],       # y-coords for markers 1,2,3  
            [[3.0], [4.0], [5.0]]        # z-coords for markers 1,2,3
        ])
        
        np.testing.assert_array_equal(markers, expected)

    def test_compute_markers_multiple_frames(self):
        """Test compute_markers with multiple frames."""
        # Test with multiple frames
        q = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        markers = self.updater.compute_markers(q)
        
        # Should return (3, nb_markers, nb_frames)
        assert markers.shape == (3, 3, 2)

    def test_to_component(self):
        """Test to_component method."""
        q = np.array([1.0, 2.0, 3.0])
        component = self.updater.to_component(q)
        
        # Should return rr.Points3D
        assert isinstance(component, rr.Points3D)

    def test_to_rerun(self):
        """Test to_rerun method (should not raise exception)."""
        q = np.array([1.0, 2.0, 3.0])
        
        # This should not raise an exception
        self.updater.to_rerun(q)

    def test_to_chunk(self):
        """Test to_chunk method."""
        q = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])  # 2 frames
        chunk = self.updater.to_chunk(q)
        
        # Should return a dictionary
        assert isinstance(chunk, dict)
        assert self.updater.name in chunk
        assert isinstance(chunk[self.updater.name], list)
        
        # Should have the right number of elements
        assert len(chunk[self.updater.name]) == 6  # Indicator + 5 component types


class TestPersistentMarkersUpdater:
    """Test cases for PersistentMarkersUpdater class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create mock persistent marker options
        self.marker_names = ["marker1", "marker2"]
        self.persistent_options = PersistentMarkerOptions(
            marker_names=self.marker_names,
            radius=0.03,
            color=np.array([0, 255, 0]),
            show_labels=True,
            nb_frames=5
        )
        
        # Create a simple callable that returns marker positions
        def mock_callable_markers(q):
            # Return 2 markers with 3D positions based on input q
            # The function should return (nb_markers, 3) so when transposed it becomes (3, nb_markers)
            return np.array([
                [q[0], q[1], q[2]],      # marker1
                [q[0]*2, q[1]*2, q[2]*2]  # marker2  
            ])
        
        self.callable_markers = mock_callable_markers
        
        # Create PersistentMarkersUpdater instance
        self.updater = PersistentMarkersUpdater(
            name="test",
            callable_markers=self.callable_markers,
            persistent_options=self.persistent_options
        )

    def test_init(self):
        """Test PersistentMarkersUpdater initialization."""
        assert self.updater.name == "test/persistent_model_markers"
        assert self.updater.callable_markers == self.callable_markers
        assert self.updater.persistent_options == self.persistent_options

    def test_nb_components(self):
        """Test nb_components property."""
        assert self.updater.nb_components == 1

    def test_nb_markers(self):
        """Test nb_markers property."""
        assert self.updater.nb_markers == 2

    def test_marker_names(self):
        """Test marker_names property."""
        assert self.updater.marker_names == self.marker_names

    def test_nb_frames(self):
        """Test nb_frames property."""
        assert self.updater.nb_frames == 5

    def test_compute_markers(self):
        """Test compute_markers method."""
        # Test with single frame
        q = np.array([[1.0], [2.0], [3.0]])
        markers = self.updater.compute_markers(q)
        
        # Should return (3, nb_markers, nb_frames)
        assert markers.shape == (3, 2, 1)
        
        # Check values for first frame
        expected = np.array([
            [[1.0], [2.0]],       # x-coords for markers 1,2
            [[2.0], [4.0]],       # y-coords for markers 1,2
            [[3.0], [6.0]]        # z-coords for markers 1,2
        ])
        
        np.testing.assert_array_equal(markers, expected)

    def test_to_component(self):
        """Test to_component method."""
        # Create test data with multiple frames to work with persistent frames
        q = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])  # 3 frames
        frame = 2  # Test frame 2
        
        component = self.updater.to_component(q, frame)
        
        # Should return rr.Points3D
        assert isinstance(component, rr.Points3D)

    def test_to_rerun(self):
        """Test to_rerun method (should not raise exception)."""
        q = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        frame = 1
        
        # This should not raise an exception
        self.updater.to_rerun(q, frame)

    def test_to_chunk(self):
        """Test to_chunk method."""
        q = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])  # 3 frames
        chunk = self.updater.to_chunk(q)
        
        # Should return a dictionary
        assert isinstance(chunk, dict)
        assert self.updater.name in chunk
        assert isinstance(chunk[self.updater.name], list)
        
        # Should have the right number of elements
        assert len(chunk[self.updater.name]) == 6  # Indicator + 5 component types


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_compute_markers_single_marker(self):
        """Test compute_markers function with single marker."""
        def mock_callable(q):
            # Single marker - return (1, 3) shape which becomes (3, 1) after transpose  
            return np.array([[q[0], q[1], q[2]]])
        
        q = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 2 frames
        result = compute_markers(q, nb_markers=1, callable_markers=mock_callable)
        
        # Should return (3, 1, 2) - xyz, 1 marker, 2 frames
        assert result.shape == (3, 1, 2)
        
        # Check first frame
        np.testing.assert_array_equal(result[:, :, 0], [[1.0], [3.0], [5.0]])
        # Check second frame
        np.testing.assert_array_equal(result[:, :, 1], [[2.0], [4.0], [6.0]])

    def test_compute_markers_multiple_markers(self):
        """Test compute_markers function with multiple markers."""
        def mock_callable(q):
            # Return (2, 3) shape which becomes (3, 2) after transpose
            return np.array([
                [q[0], q[1], q[2]],      # marker1 
                [q[0]+1, q[1]+1, q[2]+1] # marker2
            ])
        
        q = np.array([[1.0], [2.0], [3.0]])  # 1 frame
        result = compute_markers(q, nb_markers=2, callable_markers=mock_callable)
        
        # Should return (3, 2, 1) - xyz, 2 markers, 1 frame
        assert result.shape == (3, 2, 1)
        
        expected = np.array([
            [[1.0], [2.0]],  # x-coords for both markers
            [[2.0], [3.0]],  # y-coords for both markers
            [[3.0], [4.0]]   # z-coords for both markers
        ])
        
        np.testing.assert_array_equal(result, expected)

    def test_from_pyo_to_rerun(self):
        """Test from_pyo_to_rerun function."""
        # Create test data in pyomeca format [3 x N]
        pyomeca_data = np.array([
            [1.0, 2.0, 3.0],  # x-coords
            [4.0, 5.0, 6.0],  # y-coords
            [7.0, 8.0, 9.0]   # z-coords
        ])
        
        result = from_pyo_to_rerun(pyomeca_data)
        
        # Should return [N x 3] format
        expected = np.array([
            [1.0, 4.0, 7.0],
            [2.0, 5.0, 8.0], 
            [3.0, 6.0, 9.0]
        ])
        
        assert result.shape == (3, 3)
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__])