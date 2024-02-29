from abc import ABC, abstractmethod

import numpy as np


class Component(ABC):
    @abstractmethod
    def to_rerun(self, q: np.ndarray):
        pass

    @abstractmethod
    def nb_components(self):
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


class Markers(Component):
    @abstractmethod
    def nb_markers(self):
        pass


class ExperimentalData(Component):
    @abstractmethod
    def nb_frames(self):
        pass
