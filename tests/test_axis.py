import numpy as np

from pyorerun.biorbd_components.axis import Axis


def dummy_transform(q):
    return np.eye(4)


def test_axis():
    axis_x = Axis("x", dummy_transform, 0)
    axis_y = Axis("y", dummy_transform, 1)
    axis_z = Axis("z", dummy_transform, 2)

    assert axis_x.nb_components() == 1
    assert axis_y.nb_components() == 1
    assert axis_z.nb_components() == 1

    assert axis_x.color == [255, 0, 0]
    assert axis_y.color == [0, 255, 0]
    assert axis_z.color == [0, 0, 255]
