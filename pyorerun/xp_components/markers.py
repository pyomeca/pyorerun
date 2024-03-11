import numpy as np
import rerun as rr
from pyomeca import Markers as PyoMarkers

from ..abstract.abstract_class import ExperimentalData
from ..abstract.markers import Markers, MarkerProperties


class MarkersXp(Markers, ExperimentalData):
    def __init__(self, name, markers: PyoMarkers):
        self.name = name + "/markers"
        self.markers = markers
        self.markers_numpy = markers.to_numpy()
        self.markers_properties = MarkerProperties(
            markers_names=markers.channel.values.tolist(),
            radius=0.01,
            color=np.array([255, 255, 255]),
        )

    @property
    def nb_markers(self):
        return len(self.markers_names)

    @property
    def markers_names(self):
        return self.markers.channel.values.tolist()

    @property
    def nb_frames(self):
        return len(self.markers.shape[2])

    @property
    def nb_components(self):
        return 1

    def to_rerun(self, frame: int) -> None:
        rr.log(
            self.name,
            rr.Points3D(
                positions=from_pyomeca_to_rerun(self.markers_numpy[:3, :, frame]),
                radii=self.markers_properties.radius_to_rerun(),
                colors=self.markers_properties.color_to_rerun(),
                labels=self.markers_names,
            ),
        )

    def to_rerun_curve(self, frame) -> None:
        """todo:  should it be a MarkerCurve type?"""
        positions_f = from_pyomeca_to_rerun(self.markers_numpy[:3, :, frame])
        markers_names = self.markers_names
        for m in markers_names:
            for j, axis in enumerate(["X", "Y", "Z"]):
                rr.log(
                    f"markers_graphs/{m}/{axis}",
                    rr.Scalar(
                        positions_f[markers_names.index(m), j],
                    ),
                )


def from_pyomeca_to_rerun(marker_positions: np.ndarray) -> np.ndarray:
    """
    Pyomeca as a standard of [3 x N x frames] for markers positions.
    Rerun as a standard of [N x 3] for markers positions.

    The goal is to convert from [3 x N] to [N x 3]
    """
    return np.transpose(marker_positions, (1, 0))
