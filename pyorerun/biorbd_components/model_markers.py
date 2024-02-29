import numpy as np
import rerun as rr

from .abstract_class import Component


class BiorbdModelMarkers(Component):
    def __init__(self, name, markers_names: list[str], radius, color, callable: callable):
        self.name = name + "/model_markers"
        self.markers_names = markers_names
        self.callable_markers = callable
        self.radius = radius if radius is not None else 0.01
        self.color = color if color is not None else np.array([0, 0, 255])

    @property
    def nb_markers(self):
        return len(self.markers_names)

    @property
    def nb_components(self):
        return 1

    def to_rerun(self, q: np.ndarray) -> None:
        rr.log(
            self.name,
            rr.Points3D(
                positions=self.callable_markers(q),
                radii=np.ones(self.nb_markers) * self.radius,
                colors=np.tile(self.color, (self.nb_markers, 1)),
                labels=self.markers_names,
            ),
        )


def from_pyo_to_rerun(maker_positions: np.ndarray) -> np.ndarray:
    """[3 x N] to [N x 3]"""
    return np.transpose(maker_positions, (1, 0))
