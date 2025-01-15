import numpy as np

from pyorerun.biorbd_components.local_frame import LocalFrameUpdater


def dummy_callable_transform(q):
    return np.eye(4)


def test_local_frame():
    local_frame = LocalFrameUpdater("test", dummy_callable_transform)

    # Test initialization
    assert local_frame.name == "test"
    assert local_frame.transform_callable == dummy_callable_transform
    assert local_frame.scale == 0.1

    # Test nb_components property
    assert local_frame.nb_components == 1

    local_frame.to_rerun(np.array([0, 0, 0]))
