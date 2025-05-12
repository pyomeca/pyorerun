import numpy as np

from .model_interfaces.biorbd_model_interface import BiorbdModel
from .phase_rerun import PhaseRerun


def rr_biorbd(biomod: str, q: np.ndarray, tspan: np.ndarray) -> None:
    """
    Display a biorbd model in rerun.

    Parameters
    ----------
    biomod: str
        The biomod file path to display.
    q: np.ndarray
        The generalized coordinates of the model.
    tspan: np.ndarray
        The time span of the animation, such as the time instant of each frame.
    """
    model = BiorbdModel(biomod)

    phase_rerun = PhaseRerun(tspan)
    phase_rerun.add_animated_model(model, q)
    phase_rerun.rerun()
