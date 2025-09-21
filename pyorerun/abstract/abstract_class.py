from abc import ABC, abstractmethod

import numpy as np


class TimelessComponent(ABC):
    @abstractmethod
    def to_rerun(self):
        pass

    @abstractmethod
    def nb_components(self):
        pass


class Component(ABC):
    @abstractmethod
    def to_rerun(self, q: np.ndarray):
        pass

    @abstractmethod
    def nb_components(self):
        pass

    @abstractmethod
    def to_chunk(self, q: np.ndarray):
        pass


class PersistentComponent(ABC):

    def list_frames_to_keep(self, nb_frames_in_trial: int) -> list[list[int]]:
        """
        For each frame in the trial, it returns a list of the frame numbers that must be displayed.
        Example: A trial composed of 5 frames with a self.nb_frames of 3 frames would get the following output:
        [
            [0],
            [0, 1],
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
        ]
        Parameters
        ----------
        nb_frames_in_trial: int
            The number of trames that the trial contains
        """
        # Deal with the case where nb_frames is None
        if self.nb_frames is None:
            self.nb_frames = nb_frames_in_trial

        list_frames_to_keep = []
        for i in range(nb_frames_in_trial):
            if i < self.nb_frames:
                list_frames_to_keep.append(list(range(i + 1)))
            else:
                list_frames_to_keep.append(list(range(i - self.nb_frames + 1, i + 1)))
        return list_frames_to_keep

    @abstractmethod
    def to_rerun(self, q: np.ndarray, frame_bounds: tuple[int, int]):
        pass

    @abstractmethod
    def nb_components(self):
        pass

    @abstractmethod
    def to_chunk(self, q: np.ndarray):
        pass


class Components(ABC):
    @abstractmethod
    def to_rerun(self, q: np.ndarray):
        pass

    @abstractmethod
    def nb_components(self):
        pass

    @abstractmethod
    def components(self):
        pass


class ExperimentalData(Component):
    @abstractmethod
    def nb_frames(self):
        pass

    @abstractmethod
    def to_chunk(self, **kwargs) -> dict[str, list]:
        pass
