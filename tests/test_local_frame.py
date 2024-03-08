import numpy as np

from pyorerun.biorbd_components.local_frame import LocalFrame, Axis


def dummy_callable_transform(q):
    return np.eye(4)


def test_local_frame():
    local_frame = LocalFrame("test", dummy_callable_transform)

    # Test initialization
    assert local_frame.name == "test"
    assert local_frame.transform_callable == dummy_callable_transform
    assert local_frame.scale == 0.3
    assert isinstance(local_frame.x_axis, Axis)
    assert isinstance(local_frame.y_axis, Axis)
    assert isinstance(local_frame.z_axis, Axis)

    # Test components property
    assert len(local_frame.components) == 3

    # Test nb_components property
    assert local_frame.nb_components == 3

    local_frame.to_rerun(np.array([0, 0, 0]))

    assert local_frame.component_names == ["test/X", "test/Y", "test/Z"]
