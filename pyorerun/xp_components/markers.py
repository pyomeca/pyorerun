import numpy as np
import rerun as rr
from pyomeca import Markers as PyoMarkers

from ..abstract.abstract_class import Markers, ExperimentalData


class MarkersXp(Markers, ExperimentalData):
    def __init__(self, name, markers: PyoMarkers, radius: float = None, color: float = None):
        self.name = name + "/markers"
        self.markers = markers
        self.radius = radius if radius is not None else 0.01
        self.color = color if color is not None else np.array([0, 0, 255])

    @property
    def nb_markers(self):
        return len(self.markers_names)

    @property
    def markers_names(self):
        return self.markers.channel.values.tolist()

    @property
    def nb_frames(self):
        return len(self.callable_markers(q))

    @property
    def nb_components(self):
        return 1

    def to_rerun(self, frame: int) -> None:
        rr.log(
            self.name,
            rr.Points3D(
                positions=from_pyomeca_to_rerun(self.markers[:3, :, frame].to_numpy()),
                radii=np.ones(self.nb_markers) * self.radius,
                colors=np.tile(self.color, (self.nb_markers, 1)),
                labels=self.markers_names,
            ),
        )


def from_pyomeca_to_rerun(marker_positions: np.ndarray) -> np.ndarray:
    """
    Pyomeca as a standard of [3 x N x frames] for markers positions.
    Rerun as a standard of [N x 3] for markers positions.

    The goal is to convert from [3 x N] to [N x 3]
    """
    return np.transpose(marker_positions, (1, 0))
