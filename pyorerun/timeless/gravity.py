import numpy as np
import rerun as rr

from ..abstract.abstract_class import TimelessComponent


class Gravity(TimelessComponent):

    def __init__(self, name, vector: np.ndarray):
        self.name = name + "/gravity"
        self.vector = vector / 20

    @property
    def nb_components(self):
        return 1

    def to_rerun(self) -> None:
        rr.log(
            self.name,
            rr.Arrows3D(origins=np.zeros(3), vectors=self.vector, colors=np.array([255, 255, 255])),
        )
