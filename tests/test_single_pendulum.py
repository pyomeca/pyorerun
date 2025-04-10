import platform
from pathlib import Path

import numpy as np

from pyorerun import PhaseRerun, BiorbdModel


class TestUtils:
    @staticmethod
    def example_folder() -> str:
        return TestUtils._capitalize_folder_drive(str(Path(__file__).parent / "../examples/"))

    @staticmethod
    def _capitalize_folder_drive(folder: str) -> str:
        if platform.system() == "Windows" and folder[1] == ":":
            # Capitilize the drive letter if it is windows
            folder = folder[0].upper() + folder[1:]
        return folder


def test_single_pendulum():
    """
    Test the single pendulum model with a simple animation.
    """

    # building some time components
    nb_frames = 200
    nb_seconds = 1
    t_span = np.linspace(0, nb_seconds, nb_frames)

    # building some generalized coordinates
    q = np.zeros((1, nb_frames))
    q[0, :] = np.linspace(0, 0.1, nb_frames)

    # Load the model path
    model_path = TestUtils.example_folder() + "/biorbd/models/single_pendulum.bioMod"

    model = BiorbdModel(model_path)

    phase_rerun = PhaseRerun(t_span)
    phase_rerun.add_animated_model(model, q)
    phase_rerun.rerun(notebook=True)
