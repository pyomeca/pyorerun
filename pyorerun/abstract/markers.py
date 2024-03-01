from abc import abstractmethod

import numpy as np

from .abstract_class import Component


class Markers(Component):
    @abstractmethod
    def nb_markers(self):
        pass


class MarkerProperties:
    """
    A class used to represent the properties of a marker.


    Attributes
    ----------
    markers_names : list[str]
        a list of names for the markers
    radius : float
        the radius of the markers
    color : np.ndarray
        the color of the markers, in RGB format from 0 to 255, e.g. [0, 0, 255] for blue

    Methods
    -------
    nb_markers():
        Returns the number of markers.
    radius_to_rerun():
        Returns a numpy array with the radius of each marker.
    color_to_rerun():
        Returns a numpy array with the color of each marker.
    """

    def __init__(self, markers_names: list[str], radius: float, color: np.ndarray):
        """
        Constructs all the necessary attributes for the MarkerProperties object.

        Parameters
        ----------
            markers_names : list[str]
                a list of names for the markers
            radius : float
                the radius of the markers
            color : np.ndarray
                the color of the markers
        """
        self.markers_names = markers_names
        self.radius = radius
        self.color = color

    @property
    def nb_markers(self):
        """
        Returns the number of markers.

        Returns
        -------
        int
            The number of markers.
        """
        return len(self.markers_names)

    def radius_to_rerun(self) -> None:
        """
        Returns a numpy array with the radius of each marker.

        Returns
        -------
        np.ndarray
            A numpy array with the radius of each marker.
        """
        return np.ones(self.nb_markers) * self.radius

    def color_to_rerun(self) -> None:
        """
        Returns a numpy array with the color of each marker.

        Returns
        -------
        np.ndarray
            A numpy array with the color of each marker.
        """
        return np.tile(self.color, (self.nb_markers, 1))
