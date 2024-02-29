from functools import partial

import numpy as np
import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!

from .biorbd_components.model_interface import BiorbdModel
from .biorbd_phase import RerunBiorbdPhase


class Biorbd:
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
