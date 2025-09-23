import numpy as np
import rerun as rr

from ..abstract.abstract_class import Component
from ..abstract.markers import MarkerProperties
from ..xp_components.persistent_marker_options import PersistentMarkerOptions


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
            show_labels=self.marker_properties.show_labels_to_rerun(),
        )

    def compute_markers(self, q: np.ndarray) -> np.ndarray:
        nb_frames = q.shape[1]
        markers = np.zeros((3, self.nb_markers, nb_frames))

        for f in range(q.shape[1]):
            computed_markers = self.callable_markers(q[:, f])
            computed_markers = computed_markers.squeeze().T if self.nb_markers > 1 else computed_markers.T
            markers[:, :, f] = computed_markers

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
                rr.components.ShowLabelsBatch([self.marker_properties.show_labels for _ in range(nb_frames)]),
            ]
        }


class PersistentMarkersUpdater(MarkersUpdater):
    def __init__(
        self,
        name,
        marker_properties: MarkerProperties,
        callable_markers: callable,
        persistent_markers: PersistentMarkerOptions,
    ):
        super().__init__(name, marker_properties, callable_markers)
        self.persistent_markers = persistent_markers

    @property
    def nb_marker_to_keep(self) -> int:
        return len(self.persistent_markers.marker_names)

    def get_markers_to_keep(self, q: np.ndarray) ->  tuple[np.ndarray, list[str]]:
        """ From all markers, keep only the markers to compute a marker trajectory for """
        model_markers = self.compute_markers(q)
        model_markers_names = self.marker_properties.markers_names
        markers_to_keep, markers_to_keep_names = self.persistent_markers.marker_to_keep(
            model_markers, model_markers_names
        )
        return markers_to_keep, markers_to_keep_names

    def to_rerun(self, q: np.ndarray) -> None:
        rr.log(
        self.name,
        self.to_persistent_component(q),
    )

    def to_persistent_component(self, q: np.ndarray) -> rr.Points3D:
        nb_frames = q.shape[1]
        frames_to_keep = self.persistent_markers.list_frames_to_keep(nb_frames)[-1]
        markers_to_keep, markers_to_keep_names = self.get_markers_to_keep(q)
        markers = markers_to_keep[:, :, frames_to_keep].transpose(2, 1, 0).reshape(-1, 3)

        return rr.Points3D(
            positions=markers,
            radii=self.marker_properties.radius_to_rerun(),
            colors=self.marker_properties.color_to_rerun(),
            labels=markers_to_keep_names,
            show_labels=self.marker_properties.show_labels_to_rerun(),
        )

    def to_chunk(self, q: np.ndarray) -> dict[str, list]:
        """
        Parameters
        ----------
        q: np.ndarray
            The generalized coordinates (N_markers x N_frames)
        """
        nb_frames = q.shape[1]
        list_frames_to_keep = self.persistent_markers.list_frames_to_keep(nb_frames)
        markers_to_keep, markers_to_keep_names = self.get_markers_to_keep(q)

        # Repeat the markers to keep for each frame
        markers = np.empty((0, 3))
        for frames_to_keep in list_frames_to_keep:
            markers_to_display = self.compute_markers_to_display(markers_to_keep, frames_to_keep)
            markers = np.vstack((markers, markers_to_display.transpose(2, 1, 0).reshape(-1, 3)))

        # Get the partitions
        partition = [self.nb_marker_to_keep * len(frames_to_keep) for frames_to_keep in list_frames_to_keep]
        partition_marker_names = []
        for frames_to_keep in list_frames_to_keep:
            partition_marker_names += markers_to_keep_names * len(frames_to_keep)
        return {
            self.name: [
                rr.Points3D.indicator(),
                rr.components.Position3DBatch(markers).partition(partition),
                rr.components.ColorBatch([self.marker_properties.color for _ in range(nb_frames)]),
                rr.components.RadiusBatch([self.marker_properties.radius for _ in range(nb_frames)]),
                rr.components.TextBatch(partition_marker_names).partition(partition),
                rr.components.ShowLabelsBatch([self.marker_properties.show_labels for _ in range(nb_frames)]),
            ]
        }

def from_pyo_to_rerun(maker_positions: np.ndarray) -> np.ndarray:
    """[3 x N] to [N x 3]"""
    return np.transpose(maker_positions, (1, 0))
