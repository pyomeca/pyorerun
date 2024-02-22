from abc import ABC

import numpy as np


class Component(ABC):
    def __init__(self):
        self.__name = None
        self.__color = np.array([0, 0, 255])

    @property
    def name(self) -> str:
        return self.__name

    @property
    def color(self) -> np.ndarray:
        return self.__color

    def set_color(self, color: np.ndarray) -> None:
        self.__color = color

    def set_name(self, name: str) -> None:
        self.__name = name


class Object3D(Component):
    def __init__(self, name: str):
        super().__init__()
        self.name = name


class MarkerSet(Component):
    def __init__(self, positions: np.ndarray, labels: tuple[str]) -> None:
        """
        A class to store a set of markers.

        Parameters
        ----------
        positions: np.ndarray
            The markers' positions [n_frames x n_markers x 3]
        labels: tuple[str]
            The markers' labels.
        """
        super().__init__()
        self.positions = positions
        self.labels = labels
        self.set_color(np.array([0, 0, 255]))
        self.__size = 0.01

    @property
    def nb_markers(self) -> int:
        return self.positions.shape[1]

    @property
    def size(self) -> float:
        return self.__size

    def set_size(self, size: float) -> None:
        self.__size = size
