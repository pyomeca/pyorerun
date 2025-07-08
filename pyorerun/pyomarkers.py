"""
Custom Pyomarkers class to replace pyomeca dependency.
"""

from typing import Optional, List

import ezc3d
import numpy as np


class PyoMarkers:
    """
    A class to handle 3D marker data, designed to replace pyomeca.Markers.

    This class provides compatibility with the existing pyomeca.Markers API
    while being self-contained and removing the external dependency.

    Attributes
    ----------
    data : np.ndarray
        The marker data with shape (4, n_markers, n_frames) where the 4th dimension
        includes homogeneous coordinates (x, y, z, 1)
    time : np.ndarray
        Time vector for each frame
    marker_names : list of str
        Names/labels of the markers
    attrs : dict
        Metadata attributes (e.g., units)
    show_labels : bool
        Whether labels should be displayed
    """

    def __init__(
        self,
        data: np.ndarray = None,
        time: Optional[np.ndarray] = None,
        marker_names: Optional[List[str]] = None,
        channels: Optional[List[str]] = None,  # Alternative name for marker_names for compatibility
        show_labels: bool = True,
        attrs: Optional[dict] = None,
    ):
        """
        Initialize PyoMarkers instance.

        Parameters
        ----------
        data : np.ndarray
            Marker data. Can be (3, n_markers, n_frames) or (4, n_markers, n_frames)
        time : np.ndarray, optional
            Time vector. If None, creates a simple index-based time vector
        marker_names : list of str, optional
            Marker names. If None, creates default names
        channels : list of str, optional
            Alternative name for marker_names for compatibility with original API
        show_labels : bool, default True
            Whether to show marker labels
        attrs : dict, optional
            Metadata attributes
        """
        if data is None:
            raise ValueError("Data must be provided")

        # Handle data shape - ensure we have 4D (homogeneous coordinates)
        if data.ndim != 3:
            raise ValueError("Data must be 3D array with shape (3 or 4, n_markers, n_frames)")

        if data.shape[0] == 3:
            # Add homogeneous coordinate (w=1)
            ones = np.ones((1, data.shape[1], data.shape[2]))
            self.data = np.vstack([data, ones])
        elif data.shape[0] == 4:
            self.data = data.copy()
        else:
            raise ValueError("First dimension must be 3 or 4 (x,y,z or x,y,z,w)")

        # Set up time vector
        if time is None:
            self.time = np.arange(self.data.shape[2], dtype=float)
        else:
            self.time = np.array(time)

        # Set up marker names - handle both marker_names and channels parameters
        if channels is not None:
            marker_names = channels  # Use channels if provided (for compatibility)

        if marker_names is None:
            self.marker_names = [f"marker_{i}" for i in range(self.data.shape[1])]
        else:
            self.marker_names = list(marker_names)

        # Validate dimensions
        if len(self.marker_names) != self.data.shape[1]:
            raise ValueError("Number of marker names must match number of markers")
        if len(self.time) != self.data.shape[2]:
            raise ValueError("Time vector length must match number of frames")

        # Set attributes
        self.attrs = attrs if attrs is not None else {}
        self.show_labels = show_labels

        # Create a mock channel object for compatibility
        self.channel = MockChannel(self.marker_names)

    @property
    def shape(self) -> tuple:
        """Return the shape of the data."""
        return self.data.shape

    @property
    def units(self) -> str:
        """Return the units of the markers."""
        return self.attrs.get("units", "mm")

    @property
    def rate(self) -> float:
        """Return the sampling rate of the markers."""
        return self.attrs.get("rate")

    @property
    def first_frame(self) -> int:
        """Return the index of the first frame."""
        return self.attrs.get("first_frame")

    @property
    def last_frame(self) -> int:
        """Return the index of the last frame."""
        return self.attrs.get("last_frame")

    def to_numpy(self) -> np.ndarray:
        """Return the data as a numpy array."""
        return self.data.copy()

    def __truediv__(self, other):
        """Support division for unit conversion."""
        new_data = self.data / other
        return PyoMarkers(
            data=new_data,
            time=self.time.copy(),
            marker_names=self.marker_names.copy(),
            show_labels=self.show_labels,
            attrs=self.attrs.copy(),
        )

    def __itruediv__(self, other):
        """Support in-place division for unit conversion."""
        self.data /= other
        return self

    @classmethod
    def from_c3d(
        cls, filename: str, prefix_delimiter: str = ":", suffix_delimiter: str = None, show_labels: bool = True
    ) -> "PyoMarkers":
        """
        Create PyoMarkers from a C3D file.

        Parameters
        ----------
        filename : str
            Path to the C3D file
        prefix_delimiter : str, default ":"
            Delimiter for prefix in marker names
        suffix_delimiter : str, optional
            Delimiter for suffix in marker names
        show_labels : bool, default True
            Whether to show marker labels

        Returns
        -------
        PyoMarkers
            A new PyoMarkers instance
        """
        c3d = ezc3d.c3d(filename)

        # Get marker data
        points = c3d["data"]["points"]  # Shape: (4, n_markers, n_frames)

        # Get marker names
        marker_names = c3d["parameters"]["POINT"]["LABELS"]["value"]

        # Clean up marker names (remove empty strings and strip whitespace)
        marker_names = [name.strip() for name in marker_names if name.strip()]

        # Get time vector
        point_rate = c3d["parameters"]["POINT"]["RATE"]["value"][0]
        n_frames = points.shape[2]
        time = np.arange(n_frames) / point_rate

        # Get units
        units = "mm"  # Default C3D unit
        if "UNITS" in c3d["parameters"]["POINT"]:
            units = c3d["parameters"]["POINT"]["UNITS"]["value"][0].strip()

        attrs = {
            "units": units,
            "rate": point_rate,
            "filename": filename,
            "first_frame": c3d.c3d_swig.header().firstFrame(),
            "last_frame": c3d.c3d_swig.header().lastFrame(),
        }

        return cls(data=points, time=time, marker_names=marker_names, show_labels=show_labels, attrs=attrs)


class MockChannel:
    """
    Mock channel object to provide compatibility with pyomeca.Markers.channel API.
    """

    def __init__(self, marker_names: List[str]):
        self.values = np.array(marker_names)
        self.data = marker_names
