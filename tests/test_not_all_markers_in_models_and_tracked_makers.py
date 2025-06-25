import platform
from pathlib import Path

import numpy as np
import pytest

from pyorerun import PhaseRerun, BiorbdModel, Pyomarkers


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


def test_not_all_markers_in_models_and_tracked_makers():
    """
    Test that the rerun works when not all markers in the model are tracked
    """
    biorbd_model_path = TestUtils.example_folder() + "/biorbd/models/double_pendulum.bioMod"

    # building some time components
    nb_frames = 200
    nb_seconds = 1
    t_span = np.linspace(0, nb_seconds, nb_frames)

    # building some generalized coordinates
    q = np.zeros((2, nb_frames))
    q[0, :] = np.linspace(0, 0.1, nb_frames)
    q[1, :] = np.linspace(0, 0.3, nb_frames)

    # loading biorbd model
    biorbd_model = BiorbdModel(biorbd_model_path)
    all_q = q + 0.3 * np.random.rand(2, nb_frames)
    noisy_markers = np.zeros((3, biorbd_model.nb_markers - 1, nb_frames))
    for i in range(nb_frames):
        noisy_markers[:, :, i] = biorbd_model.markers(all_q[:, i]).T[:, :-1]

    # running the animation
    rerun_biorbd = PhaseRerun(t_span)
    markers = Pyomarkers(data=noisy_markers, channels=list(biorbd_model.marker_names)[::-1][:-1])
    with pytest.raises(
        ValueError,
        match=r"The markers of the model and the tracked markers are inconsistent. They must have the same names and shape.\nCurrent markers are \('marker_1', 'marker_2', 'marker_3', 'marker_4'\) and\n tracked markers: \('marker_4', 'marker_3', 'marker_2'\).",
    ):
        rerun_biorbd.add_animated_model(biorbd_model, q, tracked_markers=markers)
