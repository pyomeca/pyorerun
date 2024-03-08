import numpy as np

from pyorerun.biorbd_components.axisupdater import AxisUpdater


def dummy_transform(q):
    return np.eye(4)


def test_axis():
    axis_x = AxisUpdater("x", dummy_transform, 0)
    axis_y = AxisUpdater("y", dummy_transform, 1)
    axis_z = AxisUpdater("z", dummy_transform, 2)

    assert axis_x.nb_components() == 1
    assert axis_y.nb_components() == 1
    assert axis_z.nb_components() == 1

    assert axis_x.color == [255, 0, 0]
    assert axis_y.color == [0, 255, 0]
    assert axis_z.color == [0, 0, 255]
