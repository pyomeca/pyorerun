from abc import ABC
from functools import partial
from typing import Callable

import numpy as np
import rerun as rr


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

    def to_rerun(self, show_labels) -> Callable[[np.ndarray], rr.Points3D]:
        return partial(
            rr.Points3D,
            colors=np.tile(self.color, (self.nb_markers, 1)),
            radii=np.ones(self.nb_markers) * self.size,
            labels=self.labels if show_labels else None,
        )
