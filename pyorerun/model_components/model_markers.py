import numpy as np
import rerun as rr

from ..abstract.abstract_class import Component, PersistentComponent
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
            labels=self.marker_properties.marker_names,
            show_labels=self.marker_properties.show_labels_to_rerun(),
        )

    def compute_markers(self, q: np.ndarray) -> np.ndarray:
        return compute_markers(q, self.nb_markers, self.callable_markers)

    def to_chunk(self, q) -> dict[str, list]:
        nb_frames = q.shape[1]
        markers = self.compute_markers(q).transpose(2, 1, 0).reshape(-1, 3)
        marker_names = [name for _ in range(nb_frames) for name in self.marker_properties.marker_names]
        partition = [self.nb_markers for _ in range(nb_frames)]
        return {
            self.name: [
                rr.Points3D.indicator(),
                rr.components.Position3DBatch(markers).partition(partition),
                rr.components.ColorBatch([self.marker_properties.color for _ in range(nb_frames)]),
                rr.components.RadiusBatch([self.marker_properties.radius for _ in range(nb_frames)]),
                rr.components.TextBatch(marker_names).partition(partition),
                rr.components.ShowLabelsBatch([self.marker_properties.show_labels for _ in range(nb_frames)]),
            ]
        }


class PersistentMarkersUpdater(PersistentComponent):
    def __init__(
        self,
        name,
        marker_properties: MarkerProperties,
        callable_markers: callable,
        persistent_options: PersistentMarkerOptions,
    ):
        self.name = name + "/persistent_model_markers"
        self.marker_properties = marker_properties
        self.callable_markers = callable_markers
        self.persistent_options = persistent_options

    @property
    def nb_components(self) -> int:
        return 1

    @property
    def nb_markers(self) -> int:
        return len(self.persistent_options.marker_names)

    @property
    def marker_names(self):
        return self.persistent_options.marker_names

    @property
    def nb_frames(self) -> int:
        return self.persistent_options.nb_frames

    def compute_markers(self, q: np.ndarray) -> np.ndarray:
        return compute_markers(q, self.nb_markers, self.callable_markers)

    def to_rerun(self, q: np.ndarray, frame: int) -> None:
        rr.log(
            self.name,
            self.to_component(q, frame),
        )

    def to_component(self, q: np.ndarray, frame: int) -> rr.Points3D:
        frames_to_keep = self.persistent_options.frames_to_keep(frame)
        markers = self.compute_markers(q)
        markers = markers.transpose(2, 1, 0).reshape(-1, 3)

        return rr.Points3D(
            positions=markers,
            radii=self.marker_properties.radius_to_rerun(),
            colors=self.marker_properties.color_to_rerun(),
            labels=self.persistent_options.marker_names * len(frames_to_keep),
            show_labels=self.marker_properties.show_labels_to_rerun(),
        )

    def to_chunk(self, q: np.ndarray) -> dict[str, list]:
        """
        Parameters
        ----------
        q: np.ndarray
            The generalized coordinates (N_markers x N_frames)
        """
        nb_frames_trials = q.shape[1]

        markers = np.empty((0, 3))
        for frame in range(nb_frames_trials):
            frames_to_keep = self.persistent_options.frames_to_keep(frame)
            markers_to_display = self.compute_markers(q[:, frames_to_keep])
            markers = np.vstack((markers, markers_to_display.transpose(2, 1, 0).reshape(-1, 3)))

        # Get the partitions
        list_frames_to_keep = self.persistent_options.all_frames_to_keep(nb_frames_trials)
        partition = [self.nb_markers * len(frames_to_keep) for frames_to_keep in list_frames_to_keep]

        partition_marker_names = []
        for frame in range(nb_frames_trials):
            frames_to_keep = self.persistent_options.frames_to_keep(frame)
            partition_marker_names += self.marker_names * len(frames_to_keep)

        return {
            self.name: [
                rr.Points3D.indicator(),
                rr.components.Position3DBatch(markers).partition(partition),
                rr.components.ColorBatch([self.marker_properties.color for _ in range(nb_frames_trials)]),
                rr.components.RadiusBatch([self.marker_properties.radius for _ in range(nb_frames_trials)]),
                rr.components.TextBatch(partition_marker_names).partition(partition),
                rr.components.ShowLabelsBatch([self.marker_properties.show_labels for _ in range(nb_frames_trials)]),
            ]
        }


def compute_markers(q: np.ndarray, nb_markers, callable_markers) -> np.ndarray:
    nb_frames = q.shape[1]
    markers = np.zeros((3, nb_markers, nb_frames))

    for f in range(q.shape[1]):
        computed_markers = callable_markers(q[:, f])
        computed_markers = computed_markers.squeeze().T if nb_markers > 1 else computed_markers.T
        markers[:, :, f] = computed_markers

    return markers


def from_pyo_to_rerun(maker_positions: np.ndarray) -> np.ndarray:
    """[3 x N] to [N x 3]"""
    return np.transpose(maker_positions, (1, 0))
