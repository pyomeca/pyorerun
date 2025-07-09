import numpy as np
import rerun as rr
from ..pyomarkers import PyoMarkers

from ..abstract.abstract_class import ExperimentalData
from ..abstract.markers import Markers, MarkerProperties


class MarkersXp(Markers, ExperimentalData):

    _counter = 0  # counter for each instance of the class to change the color of the markers
    _MARKERS_COLORS = [
        np.array([255, 255, 255]),
        np.array([245, 66, 53]),
        np.array([232, 30, 99]),
        np.array([33, 149, 245]),
        np.array([76, 176, 79]),
        np.array([103, 56, 182]),
    ]

    def __init__(self, name, markers: PyoMarkers):

        self.name = name + "/markers"
        self.markers = markers
        self.markers_numpy = markers.to_numpy()
        self.markers_properties = MarkerProperties(
            markers_names=markers.channel.values.tolist(),
            radius=0.01,
            color=MarkersXp._MARKERS_COLORS[MarkersXp._counter],
            show_labels=markers.show_labels,
        )

        MarkersXp._counter = (MarkersXp._counter + 1) % len(MarkersXp._MARKERS_COLORS)

    @property
    def nb_markers(self):
        return len(self.markers_names)

    @property
    def markers_names(self):
        return self.markers.channel.values.tolist()

    @property
    def nb_frames(self):
        return self.markers.shape[2]

    @property
    def nb_components(self):
        return 1

    def initialize(self):
        pass

    def to_rerun(self, frame: int) -> None:
        rr.log(
            self.name,
            rr.Points3D(
                positions=from_pyomeca_to_rerun(self.markers_numpy[:3, :, frame]),
                radii=self.markers_properties.radius_to_rerun(),
                colors=self.markers_properties.color_to_rerun(),
                labels=self.markers_names,
                show_labels=self.markers_properties.show_labels_to_rerun(),
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

    def to_component(self, frame: int) -> rr.Points3D:
        rr.Points3D(
            positions=from_pyomeca_to_rerun(self.markers_numpy[:3, :, frame]),
            radii=self.markers_properties.radius_to_rerun(),
            colors=self.markers_properties.color_to_rerun(),
            labels=self.markers_names,
            show_labels=self.markers_properties.show_labels_to_rerun(),
        )

    def to_chunk(self, **kwargs) -> dict[str, list]:
        # flatten the markers to 3 x (nb_markers * nb_frames)
        flattened_markers = self.markers_numpy[:3, :, :].transpose(2, 1, 0).reshape(-1, 3)
        markers_names = self.markers_names * self.nb_frames
        partition = [self.nb_markers for _ in range(self.nb_frames)]
        return {
            self.name: [
                rr.Points3D.indicator(),
                rr.components.Position3DBatch(flattened_markers).partition(partition),
                rr.components.ColorBatch([self.markers_properties.color for _ in range(self.nb_frames)]),
                rr.components.RadiusBatch([self.markers_properties.radius for _ in range(self.nb_frames)]),
                rr.components.TextBatch(markers_names).partition(partition),
                rr.components.ShowLabelsBatch([self.markers_properties.show_labels for _ in range(self.nb_frames)]),
            ]
        }


def from_pyomeca_to_rerun(marker_positions: np.ndarray) -> np.ndarray:
    """
    Pyomeca as a standard of [3 x N x frames] for markers positions.
    Rerun as a standard of [N x 3] for markers positions.

    The goal is to convert from [3 x N] to [N x 3]
    """
    return np.transpose(marker_positions, (1, 0))
