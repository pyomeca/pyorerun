import numpy as np
import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!

from .biorbd_interface import BiorbdModel
from .markerset import MarkerSet
from .rr_utils import display_frame, display_meshes, display_markers


def rr_biorbd(biomod: str, q: np.ndarray, tspan: np.ndarray) -> None:
    """
    Display a biorbd model in rerun.

    Parameters
    ----------
    biomod: str
        The biomod file to display.
    q: np.ndarray
        The generalized coordinates of the model.
    tspan: np.ndarray
        The time span of the animation, such as the time instant of each frame.
    """
    model = BiorbdModel(biomod)

    rerun_biorbd = RerunBiorbd(model)
    rerun_biorbd.set_tspan(tspan)
    rerun_biorbd.set_q(q)
    rerun_biorbd.rerun("animation")


class RerunBiorbd:
    """
    A class to animate a biorbd model in rerun.
    """

    def __init__(self, biomod: BiorbdModel) -> None:
        self.model = biomod
        self.homogenous_matrices = None
        self.model_markers = None
        self.tspan = None
        self.__model_markers_color = np.array([0, 0, 255])
        self.__model_markers_size = 0.01
        self._markers = []
        self.__show_marker_labels = False

    def show_labels(self, show: bool) -> None:
        self.__show_marker_labels = show

    def set_marker_color(self, color: np.ndarray) -> None:
        self.__model_markers_color = color

    def set_marker_size(self, size: float) -> None:
        self.__model_markers_size = size

    def add_marker_set(
        self, positions: np.ndarray, name: str, color, labels: tuple[str] = None, size: float = None
    ) -> None:
        if positions.shape[0] != self.nb_frames:
            raise ValueError(
                "The number of frames in the markers must be the same as the number of frames in the animation."
            )
        marker_set = MarkerSet(positions, labels)
        marker_set.set_name(name)
        marker_set.set_color(np.array([0, 0, 0]) if color is None else color)
        marker_set.set_size(size)
        self._markers.append(marker_set)

    def set_q(self, q: np.ndarray) -> None:
        self.homogenous_matrices = self.model.all_frame_homogeneous_matrices(q)
        self.add_marker_set(
            positions=self.model.all_frame_markers(q),
            name="model",
            labels=self.model.marker_names,
            color=self.__model_markers_color,
            size=self.__model_markers_size,
        )

    def set_tspan(self, tspan: np.ndarray) -> None:
        self.tspan = tspan

    @property
    def nb_frames(self) -> int:
        return len(self.tspan)

    def rerun(self, name: str = "animation_id") -> None:

        rr.init(self.model.path, spawn=True)

        for i, t in enumerate(self.tspan):
            rr.set_time_seconds("stable_time", t)

            display_frame(name)
            display_meshes(name, self.model.meshes, self.homogenous_matrices[i, :, :, :])

            for markers in self._markers:
                display_markers(
                    name,
                    name=markers.name,
                    positions=markers.positions[i, :, :3],
                    colors=np.tile(markers.color, (markers.nb_markers, 1)),
                    radii=np.ones(markers.nb_markers) * markers.size,
                    labels=markers.labels if self.__show_marker_labels else None,
                )
