import numpy as np
import pytest

from pyorerun import BiorbdModel, PhaseRerun, PyoMarkers
from pyorerun.model_phase import ModelRerunPhase
from pyorerun.timeless_components import TimelessRerunPhase
from pyorerun.xp_phase import XpRerunPhase


def test_phase_rerun_init():
    t_span = np.linspace(0, 1, 50)
    phase_rerun = PhaseRerun(t_span, phase=1, window="test_window")
    assert phase_rerun.phase == 1
    assert phase_rerun.name == "test_window/animation_phase_1"
    np.testing.assert_array_equal(phase_rerun.t_span, t_span)
    assert isinstance(phase_rerun.models, ModelRerunPhase)
    assert isinstance(phase_rerun.xp_data, XpRerunPhase)
    assert isinstance(phase_rerun.timeless_components, TimelessRerunPhase)


def test_add_animated_model(tmp_path):
    # Create temporary biomod file
    biomod_content = """version 4
    gravity 0 0 -9.81
    segment Seg1
        translations xyz
        rotations xyz
        mass 1
        inertia
            1 0 0
            0 1 0
            0 0 1
        com 0 0 0
        mesh 0 0 0
    endsegment
    
    marker marker1
        parent Seg1
        translation 0 0 0
    endmarker
    
    """
    biomod_file = tmp_path / "test.bioMod"
    biomod_file.write_text(biomod_content)

    t_span = np.linspace(0, 1, 50)
    phase_rerun = PhaseRerun(t_span)
    model = BiorbdModel(str(biomod_file))
    q = np.zeros((model.model.nbQ(), 50))

    # Test basic model addition
    phase_rerun.add_animated_model(model, q)
    assert len(phase_rerun.models.models) == 1

    # Test with tracked markers
    markers_data = np.random.rand(3, model.nb_markers, 50)
    markers = PyoMarkers(data=markers_data, channels=list(model.marker_names))
    phase_rerun.add_animated_model(model, q, markers)

    # Test shape mismatch error
    wrong_q = np.zeros((model.nb_q, 40))
    with pytest.raises(ValueError, match="The shapes of q and tspan are inconsistent"):
        phase_rerun.add_animated_model(model, wrong_q)

    # Test marker mismatch error
    wrong_markers = PyoMarkers(data=np.random.rand(3, 5, 50))
    with pytest.raises(ValueError, match="The markers of the model and the tracked markers are inconsistent"):
        phase_rerun.add_animated_model(model, q, wrong_markers)


def test_add_xp_markers():
    t_span = np.linspace(0, 1, 50)
    phase_rerun = PhaseRerun(t_span)
    markers_data = np.random.rand(3, 10, 50)
    markers = PyoMarkers(data=markers_data)

    # Test valid markers
    phase_rerun.add_xp_markers("test_markers", markers)
    assert len(phase_rerun.xp_data.xp_data) == 1

    # Test shape mismatch error
    wrong_markers = PyoMarkers(data=np.random.rand(3, 10, 40))
    with pytest.raises(ValueError, match="The shapes of markers and tspan are inconsistent"):
        phase_rerun.add_xp_markers("wrong_markers", wrong_markers)


def test_add_floor():
    t_span = np.linspace(0, 1, 50)
    phase_rerun = PhaseRerun(t_span)

    # Test with default parameters
    phase_rerun.add_floor(square_width=1.0)
    assert len(phase_rerun.timeless_components.timeless_components) == 1

    # Test with custom parameters
    phase_rerun.add_floor(square_width=1.0, height_offset=0.1, subsquares=10)
    assert len(phase_rerun.timeless_components.timeless_components) == 2


def test_add_force_plate():
    t_span = np.linspace(0, 1, 50)
    phase_rerun = PhaseRerun(t_span)
    corners = np.random.rand(4, 3)
    phase_rerun.add_force_plate(num=0, corners=corners)
    assert len(phase_rerun.timeless_components.timeless_components) == 1


def test_add_force_data():
    t_span = np.linspace(0, 1, 50)
    phase_rerun = PhaseRerun(t_span)
    force_origin = np.random.rand(3, 50)
    force_vector = np.random.rand(3, 50)

    # Test valid force data
    phase_rerun.add_force_data(num=0, force_origin=force_origin, force_vector=force_vector)
    assert len(phase_rerun.xp_data.xp_data) == 1

    # Test shape mismatch error
    wrong_force = np.random.rand(3, 40)
    with pytest.raises(ValueError, match="The shapes of force_origin/force_vector and tspan are inconsistent"):
        phase_rerun.add_force_data(num=0, force_origin=wrong_force, force_vector=force_vector)

def test_add_vector():
    t_span = np.linspace(0, 1, 50)
    phase_rerun = PhaseRerun(t_span)
    vector_origin = np.random.rand(3, 50)
    vector_endpoint = np.random.rand(3, 50)

    # Test valid force data
    phase_rerun.add_xp_vector(name="vector", num=0, vector_origin=vector_origin, vector_endpoint=vector_endpoint)
    assert len(phase_rerun.xp_data.xp_data) == 1

    # Test shape mismatch error
    wrong_force = np.random.rand(3, 40)
    with pytest.raises(ValueError, match=r"The shapes of vector and tspan are inconsistent. They must have the same length. Current shapes are vector_origin: \(3, 40\), vector_endpoint: \(3, 50\), and tspan: \(50,\)."):
        phase_rerun.add_xp_vector(name="vector", num=0, vector_origin=wrong_force, vector_endpoint=vector_endpoint)
    with pytest.raises(ValueError, match=r"The shapes of vector and tspan are inconsistent. They must have the same length. Current shapes are vector_origin: \(3, 50\), vector_endpoint: \(3, 40\), and tspan: \(50,\)."):
        phase_rerun.add_xp_vector(name="vector", num=0, vector_origin=vector_origin, vector_endpoint=wrong_force)

    wrong_force = np.random.rand(2, 50)
    with pytest.raises(ValueError, match=r"The shapes of vector_origin and vector_endpoint must be \(3, nb_frames\). Current shapes are vector_origin: \(2, 50\) and vector_endpoint: \(3, 50\)."):
        phase_rerun.add_xp_vector(name="vector", num=0, vector_origin=wrong_force, vector_endpoint=vector_endpoint)
    with pytest.raises(ValueError, match=r"The shapes of vector_origin and vector_endpoint must be \(3, nb_frames\). Current shapes are vector_origin: \(3, 50\) and vector_endpoint: \(2, 50\)."):
        phase_rerun.add_xp_vector(name="vector", num=0, vector_origin=vector_origin, vector_endpoint=wrong_force)


def test_add_video():
    t_span = np.linspace(0, 1, 50)
    phase_rerun = PhaseRerun(t_span)
    video_array = np.random.randint(0, 255, size=(50, 10, 10, 3), dtype=np.uint8)

    # Test valid video
    phase_rerun.add_video("test_video", video_array)
    assert len(phase_rerun.xp_data.xp_data) == 1

    # Test shape mismatch error
    wrong_video = np.random.randint(0, 255, size=(40, 10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="The video array and tspan are inconsistent"):
        phase_rerun.add_video("wrong_video", wrong_video)
