import numpy as np
import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!
from pyomeca import Markers as PyoMarkers

from .biorbd_components.model_interface import BiorbdModel
from .phase_rerun import PhaseRerun


class MultiPhaseRerun:
    """
    A class to animate a biorbd model in rerun with multiple phases.
    """

    def __init__(self) -> None:
        self.rerun_biorbd_phases: list[dict] = []

    def add_phase(self, t_span: np.ndarray, phase: int = 0, window: str = "animation") -> None:

        if self.nb_phase - phase < 0:
            raise ValueError(
                f"You must add the phases in order.", f"Add phase {self.nb_phase} before adding phase {phase}."
            )

        if self.nb_phase - phase == 0:
            self.rerun_biorbd_phases.append(dict())

        rerun_biorbd = PhaseRerun(t_span, phase, window)
        self.rerun_biorbd_phases[phase][window] = rerun_biorbd

    def add_animated_model(self, biomod: BiorbdModel, q: np.ndarray, phase: int = 0, window: str = "animation") -> None:
        self.rerun_biorbd_phases[phase][window].add_animated_model(biomod, q)

    def add_xp_markers(self, name: str, markers: PyoMarkers, phase: int = 0, window: str = "animation") -> None:
        self.rerun_biorbd_phases[phase][window].add_xp_markers(name, markers)

    @property
    def nb_phase(self) -> int:
        return len(self.rerun_biorbd_phases)

    @property
    def windows(self, phase: int = 0) -> list[str]:
        return list(self.rerun_biorbd_phases[phase].keys())

    @property
    def all_windows(self) -> list[str]:
        return [windows for phase in self.rerun_biorbd_phases for windows in phase.keys()]

    def rerun(self, server_name: str = "multi_phase_animation") -> None:
        rr.init(server_name, spawn=True)
        for i, phase in enumerate(self.rerun_biorbd_phases):
            for j, rr_phase in enumerate(phase.values()):
                more_phases_after_this_one = i < self.nb_phase - 1
                rr_phase.rerun(init=False, clear_last_node=more_phases_after_this_one)
