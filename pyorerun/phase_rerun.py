import numpy as np
import rerun as rr
from pyomeca import Markers as PyoMarkers

from .biorbd_components.model_interface import BiorbdModel
from .biorbd_phase import BiorbdRerunPhase
from .xp_components.markers import MarkersXp
from .xp_phase import XpRerunPhase


class PhaseRerun:
    """
    A class to animate a biorbd model in rerun.
    """

    def __init__(self, t_span: np.ndarray, phase: int = 0, window: str = None):
        """
        Parameters
        ----------
        t_span: np.ndarray
            The time span of the animation, such as the time instant of each frame.
        phase: int
            The phase number of the animation, zero by default.
        """
        self.phase = phase
        self.name = f"animation_phase_{self.phase}"
        if window:
            self.name = f"{window}/{self.name}"

        # same t_span for the phase
        self.t_span = t_span

        self.biorbd_models = BiorbdRerunPhase(name=self.name, phase=phase)
        self.xp_data = XpRerunPhase(name=self.name, phase=phase)

        self.__window = "animation"
        self.__model_markers_color = np.array([0, 0, 255])
        self.__model_markers_size = 0.01
        self.__show_marker_labels = False
        self.__show_local_frames = True

    def add_animated_model(self, biomod: BiorbdModel, q: np.ndarray) -> None:
        """
        Add an animated model to the phase.

        Parameters
        ----------
        biomod: BiorbdModel
            The biorbd model to display.
        q: np.ndarray
            The generalized coordinates of the model.
        """
        if q.shape[1] != self.t_span.shape[0]:
            raise ValueError(
                f"The shapes of q and tspan are inconsistent. "
                f"They must have the same length."
                f"Current shapes are q: {q.shape[1]} and tspan: {self.t_span.shape}."
            )

        self.biorbd_models.add_animated_model(biomod, q)

    def add_xp_markers(self, name, markers: PyoMarkers) -> None:
        """
        Add an animated model to the phase.

        Parameters
        ----------
        name: str
            The name of the markers set.
        markers: PyoMarkers
            The experimental data to display.
        """
        if markers.shape[2] != self.t_span.shape[0]:
            raise ValueError(
                f"The shapes of q and tspan are inconsistent. "
                f"They must have the same length."
                f"Current shapes are q: {markers.shape[1]} and tspan: {self.t_span.shape}."
            )

        self.xp_data.add_data(MarkersXp(name=f"{self.name}/{name}", markers=markers))

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

    def rerun(self, name: str = "animation_phase", init: bool = True, clear_last_node: bool = False) -> None:
        if init:
            rr.init(f"{name}_{self.phase}", spawn=True)

        for frame, t in enumerate(self.t_span):
            rr.set_time_seconds("stable_time", t)
            self.biorbd_models.to_rerun(frame)
            self.xp_data.to_rerun(frame)

        if clear_last_node:
            for component in [*self.biorbd_models.component_names, *self.xp_data.component_names]:
                rr.log(component, rr.Clear(recursive=False))
