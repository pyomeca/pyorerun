from functools import partial

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

    rerun_biorbd = RerunBiorbdPhase(model)
    rerun_biorbd.set_tspan(tspan)
    rerun_biorbd.set_q(q)
    rerun_biorbd.rerun("animation", clear_last_node=False)


class RerunBiorbdPhase:
    """
    A class to animate a biorbd model in rerun.
    """

    def __init__(self, biomod: BiorbdModel, phase: int = 0):
        self.phase = phase
        self.__window = "animation"
        self.model = biomod
        self.homogenous_matrices = None
        self.model_markers = None
        self.tspan = None
        self.__model_markers_color = np.array([0, 0, 255])
        self.__model_markers_size = 0.01
        self._markers = []
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

    def add_marker_set(
        self, positions: np.ndarray, name: str, color, labels: tuple[str] = None, size: float = None
    ) -> None:
        if positions.shape[0] != self.nb_frames:
            raise ValueError(
                f"The number of frames in the markers ({positions.shape[0]}) "
                f"must be the same as the number of frames in the animation ({self.nb_frames}). "
                f"For phase {self.phase}."
            )
        marker_set = MarkerSet(positions, labels)
        marker_set.set_name(name)
        marker_set.set_color(np.array([0, 0, 0]) if color is None else color)
        if size is not None:
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

    def rerun(self, name: str = "animation_id", init: bool = True, clear_last_node: bool = False) -> None:
        full_name = f"{name}_{self.phase}"
        if init:
            rr.init(self.model.path, spawn=True)

        for i, t in enumerate(self.tspan):
            rr.set_time_seconds("stable_time", t)
            self.display_components(full_name, i)

        if clear_last_node:
            self.clear(full_name)

    def display_components(self, full_name: str, frame) -> None:
        """Display the components of the model at a specific frame"""
        display_frame(full_name)
        display_meshes(full_name, self.model.meshes, self.homogenous_matrices[frame, :, :, :], self.__show_local_frames)

        for markers in self._markers:
            display_markers(
                full_name,
                name=markers.name,
                positions=markers.positions[frame, :, :3],
                point3d=markers.to_rerun(self.__show_marker_labels),
            )

    def clear(self, name) -> None:
        """remove the displayed components by the end of the animation phase"""
        for i, mesh in enumerate(self.model.meshes):
            rr.log(name + f"/{mesh.name}_{i}", rr.Clear(recursive=False))

        for markers in self._markers:
            rr.log(name + f"/{markers.name}_markers", rr.Clear(recursive=False))


class RerunBiorbd:
    """
    A class to animate a biorbd model in rerun with multiple phases.
    """

    def __init__(self) -> None:
        self.rerun_biorbd_phases = [[]]

    def add_phase(
        self, biomod: BiorbdModel, t_span: np.ndarray, q: np.ndarray, phase: int = None, window: str = "animation"
    ) -> None:

        if self.nb_phase - phase < 0:
            raise ValueError(
                f"You must add the phases in order.", f"Add phase {self.nb_phase} before adding phase {phase}."
            )

        if self.nb_phase - phase == 0:
            self.rerun_biorbd_phases.append([])

        rerun_biorbd = RerunBiorbdPhase(biomod, phase=self.next_phase if phase is None else phase)
        rerun_biorbd.set_tspan(t_span)
        rerun_biorbd.set_q(q)
        rerun_biorbd.set_window(window)

        self.rerun_biorbd_phases[phase].append(rerun_biorbd)

    def add_marker_set(
        self,
        positions: np.ndarray,
        name: str,
        color,
        labels: tuple[str] = None,
        size: float = None,
        phase: int = None,
    ) -> None:
        phase = phase if phase is not None else self.next_phase
        self.rerun_biorbd_phases[phase][0].add_marker_set(positions, name, color, labels, size)

    @property
    def next_phase(self) -> int:
        return len(self.rerun_biorbd_phases)

    @property
    def nb_phase(self) -> int:
        return len(self.rerun_biorbd_phases)

    def rerun(self, server_name: str = "multi_phase_animation") -> None:
        rr.init(server_name, spawn=True)
        for i, phase in enumerate(self.rerun_biorbd_phases):
            for j, rerun_biorbd in enumerate(phase):

                partial_rerun = partial(
                    rerun_biorbd.rerun, name=f"{rerun_biorbd.window}/phase_{i}/element_{j}", init=False
                )

                if i < self.nb_phase - 1:
                    partial_rerun(clear_last_node=True)
                else:
                    partial_rerun(clear_last_node=False)
