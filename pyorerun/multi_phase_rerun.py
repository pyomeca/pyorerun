import numpy as np
import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!
import rerun.blueprint as rrb
from pyomeca import Markers as PyoMarkers

from .biorbd_components.model_interface import BiorbdModel
from .phase_rerun import PhaseRerun


class MultiPhaseRerun:
    """
    A class to animate a biorbd model in rerun with multiple phases.
    """

    def __init__(self) -> None:
        self.rerun_biorbd_phases: list[dict[str, PhaseRerun], ...] = []

    def add_phase(self, t_span: np.ndarray, phase: int = 0, window: str = "animation") -> None:

        if self.nb_phase - phase < 0:
            raise ValueError(
                f"You must add the phases in order.", f"Add phase {self.nb_phase} before adding phase {phase}."
            )

        if self.nb_phase - phase == 0:
            self.rerun_biorbd_phases.append(dict())

        self.rerun_biorbd_phases[phase][window] = PhaseRerun(t_span, phase, window)

    def add_animated_model(
        self,
        biomod: BiorbdModel,
        q: np.ndarray,
        tracked_markers: np.ndarray = None,
        phase: int = 0,
        window: str = "animation",
    ) -> None:
        self.rerun_biorbd_phases[phase][window].add_animated_model(biomod, q, tracked_markers)

    def add_xp_markers(self, name: str, markers: PyoMarkers, phase: int = 0, window: str = "animation") -> None:
        self.rerun_biorbd_phases[phase][window].add_xp_markers(name, markers)

    def add_q(
        self,
        name: str,
        q: np.ndarray,
        ranges: tuple[tuple[float, float], ...],
        dof_names: tuple[str, ...],
        phase: int = 0,
        window: str = "animation",
    ) -> None:
        self.rerun_biorbd_phases[phase][window].add_q(name, q, ranges, dof_names)

    def add_floor(
        self,
        square_width: float = None,
        height_offset: float = None,
        subsquares: int = None,
        phase: int = 0,
        window: str = "animation",
    ) -> None:
        self.rerun_biorbd_phases[phase][window].add_floor(square_width, height_offset, subsquares)

    def add_force_plate(self, num: int, corners: np.ndarray, phase: int = 0, window: str = "animation") -> None:
        self.rerun_biorbd_phases[phase][window].add_force_plate(num, corners)

    def add_force_data(
        self, num: int, force_origin: np.ndarray, force_vector: np.ndarray, phase: int = 0, window: str = "animation"
    ) -> None:
        self.rerun_biorbd_phases[phase][window].add_force_data(num, force_origin, force_vector)

    @property
    def nb_phase(self) -> int:
        return len(self.rerun_biorbd_phases)

    @property
    def windows(self, phase: int = 0) -> list[str]:
        return list(self.rerun_biorbd_phases[phase].keys())

    @property
    def all_windows(self) -> list[str]:
        return [windows for phase in self.rerun_biorbd_phases for windows in phase.keys()]

    def rerun_by_frame(self, server_name: str = "multi_phase_animation", notebook=False) -> None:
        rr.init(server_name, spawn=True if not notebook else False)
        for i, phase in enumerate(self.rerun_biorbd_phases):
            for j, (window, rr_phase) in enumerate(phase.items()):

                rrb.Spatial3DView(
                    origin="/",
                    contents=f"{window}/**",
                )

                more_phases_after_this_one = i < self.nb_phase - 1
                rr_phase.rerun_by_frame(init=False, clear_last_node=more_phases_after_this_one)

    def rerun(self, server_name: str = "multi_phase_animation", notebook=False) -> None:
        rr.init(server_name, spawn=True if not notebook else False)
        for i, phase in enumerate(self.rerun_biorbd_phases):
            for j, (window, rr_phase) in enumerate(phase.items()):

                rrb.Spatial3DView(
                    origin="/",
                    contents=f"{window}/**",
                )

                more_phases_after_this_one = i < self.nb_phase - 1
                rr_phase.rerun(init=False, clear_last_node=False)
