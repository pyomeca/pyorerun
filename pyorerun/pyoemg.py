"""
Custom PyoMuscles class to display muscle activation.
"""

from typing import Optional, List

import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap


class PyoMuscles:
    """
    A class to handle emg data.
    """

    def __init__(
        self,
        data: np.ndarray = None,
        time: Optional[np.ndarray] = None,
        muscle_names: Optional[List[str]] = None,
        mvc: Optional[np.ndarray] = None,
        colormap: Optional[ListedColormap] | str = get_cmap("magma"),
        attrs: Optional[dict] = None,
    ):
        """
        Initialize PyoMuscles instance.
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
        colormap: matplotlib.cm.get_cmap() instance, optional
            The colormap to use when displaying the emg data. If None, the default is "magma".
        attrs : dict
            Metadata attributes (e.g., units)
        """
        self.data = self.check_and_rectify_data(data)
        self.time = self.initialize_time_vector(time)
        self.muscle_names = self.initialize_muscle_names(muscle_names)
        self.mvc = self.initialize_mvc(mvc)
        self.colormap = self.initialize_colormap(colormap)
        self.check_dimensions()
        self.attrs = attrs if attrs is not None else {}

    @staticmethod
    def check_and_rectify_data(data: np.ndarray) -> np.ndarray:
        """Check if the data is valid and return the absolute value of the muscle activation (useful in the case of raw EMG)."""
        if data is None:
            raise ValueError("Data must be provided")

        # Handle data shape
        if data.ndim != 2:
            raise ValueError("Data must be 2D array with shape (n_emg, n_frames)")
        # Rectify the emg signal
        rectified_data = np.abs(data)

        return rectified_data

    def initialize_time_vector(self, time: Optional[np.ndarray | list | tuple] = None) -> np.ndarray:
        """Set up the time vector as a np.ndarray."""
        if time is None:
            time_vector = np.arange(self.data.shape[1], dtype=float)
        else:
            if time.shape[0] != self.data.shape[1]:
                raise ValueError("Time vector length must match number of frames provided in the data.")
            time_vector = np.array(time)
        return time_vector

    def initialize_muscle_names(self, muscle_names: Optional[List[str]] = None) -> List[str]:
        """Check if muscle names are provided and set default names if not."""
        if muscle_names is None:
            muscle_names = [f"muscle_{i}" for i in range(self.data.shape[0])]
        else:
            muscle_names = list(muscle_names)
        return muscle_names

    def initialize_mvc(self, mvc: Optional[np.ndarray] = None) -> np.ndarray:
        """Check that the mvc values provided are correct."""
        if mvc is None:
            mvc = np.nanmax(self.data, axis=1)
        else:
            if mvc.shape[0] != self.data.shape[0]:
                raise ValueError(
                    f"MVC values must be provided for each muscle. There were {mvc.shape[0]} mvc values and {self.data.shape[0]} muscle values provided."
                )
            if np.any(mvc <= 0.0):
                raise ValueError("MVC values must be strictly positive.")
        return mvc

    @staticmethod
    def initialize_colormap(colormap: Optional[ListedColormap | str] = None) -> ListedColormap:
        """Check that the colormap provided is correct"""
        if colormap is not None:
            if isinstance(colormap, str):
                colormap = get_cmap(colormap)
            if not isinstance(colormap, ListedColormap):
                raise TypeError("colormap must be a matplotlib.cm.get_cmap instance or the name of the colormap (str).")
        return colormap

    def check_dimensions(self):
        """Validate that the dimensions match."""
        if len(self.muscle_names) != self.data.shape[0]:
            raise ValueError("Number of marker names must match number of markers")
        if len(self.time) != self.data.shape[1]:
            raise ValueError("Time vector length must match number of frames")

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
        """Return the data as a numpy array and normalize by MVC."""
        data = self.data.copy()
        for i_muscle in range(self.data.shape[0]):
            data[i_muscle, :] /= self.mvc[i_muscle]
        return data

    def to_colors(self) -> np.ndarray:
        """Return a np.array of RGB values for each muscle."""
        data = self.to_numpy()
        return self.colormap(data)[:, :, :3]

    def __truediv__(self, other):
        """Support division for unit conversion."""
        new_data = self.data / other
        return PyoMuscles(
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

    @classmethod
    def from_c3d(cls, filename: str) -> "PyoMarkers":
        raise NotImplementedError("Defining muscle activation from a c3d file is not implmented yet.")
