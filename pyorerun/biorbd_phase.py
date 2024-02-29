import numpy as np
import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!

from .biorbd_components.model_interface import BiorbdModel
from .biorbd_components.segments import BiorbdModelSegments


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

    rerun_biorbd = RerunBiorbdPhase(model)
    rerun_biorbd.set_q_and_t_span(q, tspan)
    rerun_biorbd.rerun("animation", clear_last_node=False)


class RerunBiorbdPhase:
    """
    A class to animate a biorbd model in rerun.
    """

    def __init__(self, biomod: BiorbdModel, phase: int = 0):
        self.phase = phase
        self.__window = "animation"
        self.model = biomod
        self.rerun_model = BiorbdModelSegments(name=biomod.name, model=biomod)
        self.t_span = None
        self.q = None
        self.__model_markers_color = np.array([0, 0, 255])
        self.__model_markers_size = 0.01
        self.__show_marker_labels = False
        self.__show_local_frames = True

    def set_window(self, window: str) -> None:
        self.__window = window

    @property
    def window(self) -> str:
        return self.__window

    def show_local_frames(self, show: bool) -> None:
        self.__show_local_frames = show

    def show_labels(self, show: bool) -> None:
        self.__show_marker_labels = show

    def set_marker_color(self, color: np.ndarray) -> None:
        self.__model_markers_color = color

    def set_marker_size(self, size: float) -> None:
        self.__model_markers_size = size

    # def add_marker_set(
    #     self, positions: np.ndarray, name: str, color, labels: tuple[str] = None, size: float = None
    # ) -> None:
    #     if positions.shape[0] != self.nb_frames:
    #         raise ValueError(
    #             f"The number of frames in the markers ({positions.shape[0]}) "
    #             f"must be the same as the number of frames in the animation ({self.nb_frames}). "
    #             f"For phase {self.phase}."
    #         )
    #     marker_set = MarkerSet(positions, labels)
    #     marker_set.set_name(name)
    #     marker_set.set_color(np.array([0, 0, 0]) if color is None else color)
    #     if size is not None:
    #         marker_set.set_size(size)
    #     self._markers.append(marker_set)

    def set_q_and_t_span(self, q: np.ndarray, t_span: np.ndarray) -> None:
        """
        Set the generalized coordinates q and the time span of the animation.

        Parameters
        ----------
        q: np.ndarray
            The generalized coordinates of the model.
        t_span: np.ndarray
            The time span of the animation, such as the time instant of each frame.

        """
        if q.shape[1] != t_span.shape[0]:
            raise ValueError(
                f"The shapes of q and tspan are inconsistent. "
                f"They must have the same length."
                f"Current shapes are q: {q.shape} and tspan: {t_span.shape}."
            )

        self.q = q
        self.t_span = t_span

    @property
    def nb_frames(self) -> int:
        return len(self.t_span)

    def rerun(self, name: str = "animation_id", init: bool = True, clear_last_node: bool = False) -> None:
        full_name = f"{name}_{self.phase}"
        if init:
            rr.init(self.model.path, spawn=True)

        for i, t in enumerate(self.t_span):
            rr.set_time_seconds("stable_time", t)
            self.rerun_model.to_rerun(self.q[:, i])

        if clear_last_node:
            self.clear(full_name)

    def clear(self, name) -> None:
        """remove the displayed components by the end of the animation phase"""
        for i, mesh in enumerate(self.model.meshes):
            rr.log(name + f"/{mesh.name}_{i}", rr.Clear(recursive=False))

        for markers in self._markers:
            rr.log(name + f"/{markers.name}_markers", rr.Clear(recursive=False))
