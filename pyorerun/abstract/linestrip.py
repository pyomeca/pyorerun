from abc import abstractmethod

import numpy as np

from .abstract_class import Component


class LineStrips(Component):
    @abstractmethod
    def nb_strips(self):
        pass


class LineStripProperties:
    """
    A class used to represent the properties of a line strip.


    Attributes
    ----------
    strip_names : list[str]
        a list of names for the markers
    radius : float
        the radius of the markers
    color : np.ndarray
        the color of the markers, in RGB format from 0 to 255, e.g. [0, 0, 255] for blue

    Methods
    -------
    nb_strips():
        Returns the number of markers.
    radius_to_rerun():
        Returns a numpy array with the radius of each marker.
    color_to_rerun():
        Returns a numpy array with the color of each marker.
    """

    def __init__(self, strip_names: list[str, ...] | tuple[str, ...], radius: float | np.ndarray, color: np.ndarray, show_labels: bool = True):
        """
        Constructs all the necessary attributes for the MarkerProperties object.

        Parameters
        ----------
        strip_names : list[str]
            a list of names for the markers
        radius : float
            the radius of the markers
        color : np.ndarray
            the color of the markers
        show_labels : bool
            whether to show the labels of the markers (this can be changed by checking the appropriate box in the GUI)
        """
        self.strip_names = strip_names
        self.radius = radius
        self.color = color
        self.show_labels = show_labels

    @property
    def nb_strips(self):
        """
        Returns the number of markers.

        Returns
        -------
        int
            The number of markers.
        """
        return len(self.strip_names)

    def radius_to_rerun(self) -> np.ndarray:
        """
        Returns a numpy array with the radius of each marker.

        Returns
        -------
        np.ndarray
            A numpy array with the radius of each marker.
        """
        if isinstance(self.radius, float):
            return np.ones(self.nb_strips) * self.radius
        else:
            return self.radius

    def color_to_rerun(self) -> np.ndarray:
        """
        Returns a numpy array with the color of each marker.

        Returns
        -------
        np.ndarray
            A numpy array with the color of each marker.
        """
        if self.color.ndim == 1:
            return np.tile(self.color, (self.nb_strips, 1))
        else:
            return self.color
