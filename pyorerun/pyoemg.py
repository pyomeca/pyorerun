"""
Custom Pyoemg class to display muscle activation.
"""

from typing import Optional, List

import ezc3d
import numpy as np


class Pyoemg:
    """
    A class to handle emg data.
    """

    def __init__(
        self,
        data: np.ndarray = None,
        time: Optional[np.ndarray] = None,
        muscle_names: Optional[List[str]] = None,
        mvc: Optional[np.ndarray] = None,
        colormap: Optional["matplotlib.colors.Colormap"] = None,
        attrs: Optional[dict] = None,
    ):
        """
        Initialize Pyoemg instance.
        WARNING: The order in which the muscles are provided must be the same as the MVC values.

        Parameters
        ----------
        data : np.ndarray
            The emg data with shape (n_emg, n_frames)
        time : np.ndarray
            Time vector for each frame
        muscle_names : list of str
            Names/labels of the emg/muscles
        mvc : np.ndarray
            The maximal voluntary contraction values for each muscle. If None, the default is the maximal value across all frames for each muscle independently.
        colormap: matplotlblib.colors.Colormap, optional
        attrs : dict
            Metadata attributes (e.g., units)
        """
        if data is None:
            raise ValueError("Data must be provided")

        # Handle data shape
        if data.ndim != 2:
            raise ValueError("Data must be 2D array with shape (n_emg, n_frames)")
        self.data = data

        # Set up time vector
        if time is None:
            self.time = np.arange(self.data.shape[1], dtype=float)
        else:
            if time.shape[0] != self.data.shape[1]:
                raise ValueError("Time vector length must match number of frames provided in the data.")
            self.time = np.array(time)

        if muscle_names is None:
            self.muscle_names = [f"muscle_{i}" for i in range(self.data.shape[0])]
        else:
            self.muscle_names = list(muscle_names)

        if mvc is None:
            self.mvc = np.nanmax(self.data, axis=1)
        else:
            if mvc.shape[0] != self.data.shape[0]:
                raise ValueError(f"MVC values must be provided for each muscle. There were {mvc.shape[0]} mvc values and {self.data.shape[0]} muscle values provided.")
            self.mvc = mvc

        # Validate dimensions
        if len(self.muscle_names) != self.data.shape[0]:
            raise ValueError("Number of marker names must match number of markers")
        if len(self.time) != self.data.shape[2]:
            raise ValueError("Time vector length must match number of frames")

        # Set attributes
        self.attrs = attrs if attrs is not None else {}


    @property
    def shape(self) -> tuple:
        """Return the shape of the data."""
        return self.data.shape

    @property
    def units(self) -> str:
        """Return the units of the emg."""
        return self.attrs.get("units", "V")

    @property
    def rate(self) -> float:
        """Return the sampling rate of the emg."""
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
        return Pyoemg(
            data=new_data,
            time=self.time.copy(),
            muscle_names=self.muscle_names.copy(),
            mvc=self.mvc.copy(),
            colormap=self.colormap.copy(),
            attrs=self.attrs.copy(),
        )

    def __itruediv__(self, other):
        """Support in-place division for unit conversion."""
        self.data /= other
        return self


    # .....
    @classmethod
    def from_c3d(
        cls, filename: str, prefix_delimiter: str = ":", suffix_delimiter: str = None, show_labels: bool = True
    ) -> "Pyomarkers":
        """
        Create Pyomarkers from a C3D file.

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
        Pyomarkers
            A new Pyomarkers instance
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
