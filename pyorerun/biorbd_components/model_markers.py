import numpy as np
import rerun as rr

from ..abstract.abstract_class import Component
from ..abstract.markers import MarkerProperties


class MarkersUpdater(Component):
    def __init__(self, name, marker_properties: MarkerProperties, callable_markers: callable):
        self.name = name + "/model_markers"
        self.marker_properties = marker_properties
        self.callable_markers = callable_markers

    @property
    def nb_markers(self) -> int:
        return self.marker_properties.nb_markers

    @property
    def nb_components(self) -> int:
        return 1

    def to_rerun(self, q: np.ndarray) -> None:
        rr.log(
            self.name,
            self.to_component(q),
        )

    def to_component(self, q: np.ndarray) -> rr.Points3D:
        return rr.Points3D(
            positions=self.callable_markers(q),
            radii=self.marker_properties.radius_to_rerun(),
            colors=self.marker_properties.color_to_rerun(),
            labels=self.marker_properties.markers_names,
            show_labels=False,
        )

    def compute_markers(self, q: np.ndarray) -> np.ndarray:
        nb_frames = q.shape[1]
        markers = np.zeros((3, self.nb_markers, nb_frames))
        if self.nb_markers > 1:
            for f in range(q.shape[1]):
                markers[:, :, f] = self.callable_markers(q[:, f]).squeeze().T
        else:
            for f in range(q.shape[1]):
                markers[:, :, f] = self.callable_markers(q[:, f]).T

        return markers

    def to_chunk(self, q) -> dict[str, list]:
        nb_frames = q.shape[1]
        markers = self.compute_markers(q).transpose(2, 1, 0).reshape(-1, 3)
        markers_names = [name for _ in range(nb_frames) for name in self.marker_properties.markers_names]
        partition = [self.nb_markers for _ in range(nb_frames)]
        return {
            self.name: [
                rr.Points3D.indicator(),
                rr.components.Position3DBatch(markers).partition(partition),
                rr.components.ColorBatch([self.marker_properties.color for _ in range(nb_frames)]),
                rr.components.RadiusBatch([self.marker_properties.radius for _ in range(nb_frames)]),
                rr.components.TextBatch(markers_names).partition(partition),
                rr.components.ShowLabelsBatch([False for _ in range(nb_frames)]),
            ]
        }


def from_pyo_to_rerun(maker_positions: np.ndarray) -> np.ndarray:
    """[3 x N] to [N x 3]"""
    return np.transpose(maker_positions, (1, 0))
