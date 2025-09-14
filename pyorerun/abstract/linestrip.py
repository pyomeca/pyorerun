from abc import abstractmethod

import numpy as np

from .abstract_class import Component, PersistentComponent


class LineStrips(Component):
    @abstractmethod
    def nb_strips(self):
        pass

class PersistentLineStrips(PersistentComponent):
    @abstractmethod
    def nb_strips(self):
        pass

class LineStripProperties:
    """
    A class used to represent the properties of a line strip.


    Attributes
    ----------
    strip_names : list[str]
        a list of names for the lines
    radius : float
        the radius of the lines
    color : np.ndarray
        the color of the lines, in RGB format from 0 to 255, e.g. [0, 0, 255] for blue
    show_labels : bool | list[bool]
        whether to show the labels of the lines (this can be changed by checking the appropriate box in the GUI)

    Methods
    -------
    nb_strips():
        Returns the number of lines.
    radius_to_rerun():
        Returns a numpy array with the radius of each line.
    color_to_rerun():
        Returns a numpy array with the color of each line.
    show_labels_to_rerun():
        Returns a list of booleans indicating if the label of each line should be displayed.
    """

    def __init__(
        self,
        strip_names: list[str, ...] | tuple[str, ...],
        radius: float | np.ndarray,
        color: np.ndarray,
        show_labels: bool | list[bool] = True,
    ):
        """
        Constructs all the necessary attributes for the MarkerProperties object.

        Parameters
        ----------
        strip_names : list[str]
            a list of names for the lines
        radius : float
            the radius of the lines
        color : np.ndarray
            the color of the lines
        show_labels : bool | list[bool]
            whether to show the labels of the lines (this can be changed by checking the appropriate box in the GUI)
        """
        self.strip_names = strip_names
        self.radius = radius
        self.color = color
        self.show_labels = show_labels

    @property
    def nb_strips(self):
        """
        Returns the number of lines.

        Returns
        -------
        int
            The number of lines.
        """
        return len(self.strip_names)

    def radius_to_rerun(self) -> np.ndarray:
        """
        Returns a numpy array with the radius of each line.

        Returns
        -------
        np.ndarray
            A numpy array with the radius of each line.
        """
        if isinstance(self.radius, float):
            return np.ones(self.nb_strips) * self.radius
        else:
            return self.radius

    def color_to_rerun(self, nb_frames: int) -> np.ndarray:
        """
        Returns a numpy array with the color of each line.

        Returns
        -------
        np.ndarray
            A numpy array with the color of each line.
        """
        nb_strips = len(self.strip_names)
        if len(self.color.shape) == 3:
            colors = [self.color[s, f, :] for f in range(nb_frames) for s in range(nb_strips)]
        elif len(self.color.shape) == 2:
            colors = [self.color[s, :, :] for _ in range(nb_frames) for s in range(nb_strips)]
        else:
            colors = [self.color for _ in range(nb_frames * nb_strips)]
        return colors

    def show_labels_to_rerun(self) -> list[bool]:
        """
        Returns a list of booleans indicating if the label of each line should be displayed.
        """
        if isinstance(self.show_labels, bool):
            return [self.show_labels] * self.nb_strips
        elif isinstance(self.show_labels, list):
            return self.show_labels
        else:
            raise ValueError("The show_labels attribute must be a boolean or a list of booleans.")
