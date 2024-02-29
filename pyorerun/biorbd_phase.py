import numpy as np

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

    rerun_biorbd = BiorbdRerunPhase(model)
    rerun_biorbd.set_q_and_t_span(q, tspan)
    rerun_biorbd.rerun("animation", clear_last_node=False)


class BiorbdRerunPhase:
    """
    A class to animate a biorbd model in rerun.
    """

    def __init__(self, name, phase: int = 0):
        self.name = name
        self.phase = phase
        self.models = []
        self.rerun_models = []
        self.q = []

    def add_animated_model(self, biomod: BiorbdModel, q: np.ndarray):
        self.models.append(biomod)
        self.rerun_models.append(BiorbdModelSegments(name=f"{self.name}/{self.nb_models}_{biomod.name}", model=biomod))
        self.q.append(q)

    def to_rerun(self, frame: int):
        for i, rr_model in enumerate(self.rerun_models):
            rr_model.to_rerun(self.q[i][:, frame])

    @property
    def nb_models(self):
        return len(self.models)
