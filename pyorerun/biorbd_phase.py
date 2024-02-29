import numpy as np

from .biorbd_components.model_interface import BiorbdModel
from .biorbd_components.segments import BiorbdModelSegments


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
