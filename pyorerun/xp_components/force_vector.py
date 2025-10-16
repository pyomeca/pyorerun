from abc import ABC

import numpy as np
import rerun as rr

from ..abstract.abstract_class import ExperimentalData
from ..abstract.markers import rgb255_to_hex_rgba

VECTOR_COLOR = rgb255_to_hex_rgba(np.array([201, 219, 227]))


class Vector(ExperimentalData, ABC):
    def __init__(self, name: str, num: int, vector_origins: np.ndarray, vector_magnitudes: np.ndarray):
        self.name = name + f"/force_vector_{num}"

        assert vector_origins.shape == vector_magnitudes.shape

        self.vector_origins = vector_origins
        self.vector_magnitude = vector_magnitudes

    @property
    def nb_components(self):
        return 1

    @property
    def nb_frames(self):
        return self.vector_origins.shape[1]

    def to_component(self, frame: int) -> np.ndarray:
        return rr.Arrows3D(
            origins=self.vector_origins[:, frame],
            vectors=self.vector_magnitude[:, frame],
            colors=VECTOR_COLOR,
        )

    def initialize(self):
        pass

    def to_rerun(self, frame) -> None:
        rr.log(
            self.name,
            self.to_component(frame),
        )

    def to_chunk(self, **kwargs) -> dict[str, list]:

        return {
            self.name: [
                *rr.Arrows3D.columns(
                    origins=self.vector_origins.T.tolist(),
                    vectors=self.vector_magnitude.T.tolist(),
                    colors=[VECTOR_COLOR for _ in range(self.nb_frames)],
                )
            ]
        }


class ForceVector(Vector):
    """
    Display a force vector in rerun, and apply a scaling factor to the vector magnitude to make it readable.
    """

    def __init__(self, name: str, num: int, vector_origins: np.ndarray, vector_magnitudes: np.ndarray):
        super().__init__(name, num, vector_origins, vector_magnitudes)
        self.vector_magnitude = vector_magnitudes / 200


class VectorXp(Vector):
    """
    Display a vector in rerun.
    """

    def __init__(self, name: str, num: int, vector_origin: np.ndarray, vector_endpoint: np.ndarray):
        vector_magnitude = vector_endpoint - vector_origin
        super().__init__(name, num, vector_origin, vector_magnitude)
