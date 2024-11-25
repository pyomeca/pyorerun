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
