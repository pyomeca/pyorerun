import platform
from pathlib import Path

import numpy as np
import pytest
import numpy.testing as npt

from pyorerun import PhaseRerun, BiorbdModel, PyoMarkers
from pyorerun.utils.markers_utils import check_and_adjust_markers, sort_markers_to_a_predefined_order


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
    markers = PyoMarkers(data=noisy_markers, channels=list(biorbd_model.marker_names)[::-1][:-1])
    with pytest.raises(
        ValueError,
        match=r"The markers of the model and the tracked markers are inconsistent. They must have the same names and shape.\nCurrent markers are \('marker_1', 'marker_2', 'marker_3', 'marker_4'\) and\n tracked markers: \('marker_4', 'marker_3', 'marker_2'\).",
    ):
        rerun_biorbd.add_animated_model(biorbd_model, q, tracked_markers=markers)


def test_tracked_markers_not_in_order():
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
    noisy_markers = np.zeros((3, biorbd_model.nb_markers, nb_frames))
    for i in range(nb_frames):
        noisy_markers[:, :, i] = biorbd_model.markers(all_q[:, i]).T

    # running the animation
    rerun_biorbd = PhaseRerun(t_span)
    markers = PyoMarkers(data=noisy_markers, channels=list(biorbd_model.marker_names)[::-1])
    rerun_biorbd.add_animated_model(biorbd_model, q, tracked_markers=markers)

    # Test that the markers are in the right order
    npt.assert_almost_equal(rerun_biorbd.xp_data.xp_data[0].markers_numpy[:3, 0, 100], noisy_markers[:3, -1, 100])
    npt.assert_almost_equal(rerun_biorbd.xp_data.xp_data[0].markers_numpy[:3, -1, 100], noisy_markers[:3, 0, 100])


def test_check_and_adjust_markers():

    # Creating a model
    biorbd_model_path = TestUtils.example_folder() + "/biorbd/models/double_pendulum.bioMod"
    biorbd_model = BiorbdModel(biorbd_model_path)

    # Creating fake markers
    np.random.seed(42)
    noisy_markers = np.random.random((3, biorbd_model.nb_markers, 2))

    # Shape is not consistent
    with pytest.raises(
        ValueError,
        match=r"The markers of the model and the tracked markers are inconsistent. "
        r"They must have the same names and shape.\nCurrent markers are \('marker_1', 'marker_2', "
        r"'marker_3', 'marker_4'\) and\n tracked markers: \('marker_0', 'marker_1', 'marker_2'\).",
    ):
        check_and_adjust_markers(biorbd_model, PyoMarkers(data=noisy_markers[:, :-1, :]))

    # Marker names are not of the right length
    with pytest.raises(ValueError, match="Number of marker names must match number of markers"):
        pyomarkers = PyoMarkers(data=noisy_markers, marker_names=["marker_2", "marker_3", "marker_4"])

    # Marker names are not all in the model
    with pytest.raises(
        ValueError,
        match=r"The markers of the model and the tracked markers are inconsistent. Tracked markers "
        r"\('tata', 'marker_2', 'marker_3', 'marker_4'\) must contain all the markers in the model "
        r"\('marker_1', 'marker_2', 'marker_3', 'marker_4'\).",
    ):
        pyomarkers = PyoMarkers(data=noisy_markers, marker_names=["tata", "marker_2", "marker_3", "marker_4"])
        check_and_adjust_markers(biorbd_model, pyomarkers)

    # The markers were in the wrong order, but were reordered
    pyomarkers = PyoMarkers(data=noisy_markers, marker_names=["marker_2", "marker_1", "marker_3", "marker_4"])
    tracked_markers = check_and_adjust_markers(biorbd_model, pyomarkers)
    npt.assert_almost_equal(tracked_markers.to_numpy()[:3, 0, 0], noisy_markers[:3, 1, 0])
    npt.assert_almost_equal(tracked_markers.to_numpy()[:3, 1, 0], noisy_markers[:3, 0, 0])
    npt.assert_almost_equal(tracked_markers.to_numpy()[:3, 0, 1], noisy_markers[:3, 1, 1])
    npt.assert_almost_equal(tracked_markers.to_numpy()[:3, 1, 1], noisy_markers[:3, 0, 1])

    # Everything was as it should have been
    pyomarkers = PyoMarkers(data=noisy_markers, marker_names=["marker_1", "marker_2", "marker_3", "marker_4"])
    tracked_markers = check_and_adjust_markers(biorbd_model, pyomarkers)
    npt.assert_almost_equal(tracked_markers.to_numpy()[:3, 0, 0], noisy_markers[:3, 0, 0])
    npt.assert_almost_equal(tracked_markers.to_numpy()[:3, 0, 1], noisy_markers[:3, 0, 1])


def test_sort_markers_to_a_predefined_order():
    """
    Test the sort_markers_to_a_predefined_order function
    """
    # Creating a model
    biorbd_model_path = TestUtils.example_folder() + "/biorbd/models/double_pendulum.bioMod"
    biorbd_model = BiorbdModel(biorbd_model_path)

    # Creating fake markers
    np.random.seed(42)
    noisy_markers = np.random.random((3, biorbd_model.nb_markers, 2))

    # The markers were in the wrong order, but were reordered
    pyomarkers = PyoMarkers(data=noisy_markers, marker_names=["marker_2", "marker_1", "marker_3", "marker_4"])
    reordered_pyomarkers = sort_markers_to_a_predefined_order(
        pyomarkers, pyomarkers.marker_names, biorbd_model.marker_names
    )
    npt.assert_almost_equal(reordered_pyomarkers.to_numpy()[:3, 0, 0], noisy_markers[:3, 1, 0])
    npt.assert_almost_equal(reordered_pyomarkers.to_numpy()[:3, 1, 0], noisy_markers[:3, 0, 0])
    npt.assert_almost_equal(reordered_pyomarkers.to_numpy()[:3, 0, 1], noisy_markers[:3, 1, 1])
    npt.assert_almost_equal(reordered_pyomarkers.to_numpy()[:3, 1, 1], noisy_markers[:3, 0, 1])

    # Everything was as it should have been
    pyomarkers = PyoMarkers(data=noisy_markers, marker_names=["marker_1", "marker_2", "marker_3", "marker_4"])
    reordered_pyomarkers = sort_markers_to_a_predefined_order(
        pyomarkers, pyomarkers.marker_names, biorbd_model.marker_names
    )
    npt.assert_almost_equal(reordered_pyomarkers.to_numpy()[:3, 0, 0], noisy_markers[:3, 0, 0])
    npt.assert_almost_equal(reordered_pyomarkers.to_numpy()[:3, 0, 1], noisy_markers[:3, 0, 1])
