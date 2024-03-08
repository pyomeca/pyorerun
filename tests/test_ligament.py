import numpy as np

from pyorerun.biorbd_components.ligaments import BiorbdModelLigaments, LineStripProperties


def dummy_callable_ligaments(q):
    return np.array([[0, 0, 0], [1, 1, 1]])


def test_biorbd_model_ligaments():
    ligament_properties = LineStripProperties(
        strip_names=["strip1", "strip2"],
        radius=0.5,
        color=np.array([255, 0, 0]),
    )

    np.testing.assert_almost_equal(ligament_properties.color_to_rerun(), np.array([[255, 0, 0], [255, 0, 0]]))
    np.testing.assert_almost_equal(ligament_properties.radius_to_rerun(), np.array([0.5, 0.5]))

    ligaments = BiorbdModelLigaments("test", ligament_properties, dummy_callable_ligaments)

    # Test initialization
    assert ligaments.name == "test/ligaments"
    assert ligaments.ligament_properties == ligament_properties
    assert ligaments.callable_ligament == dummy_callable_ligaments

    # Test nb_strips property
    assert ligaments.nb_strips == 2

    # Test nb_components property
    assert ligaments.nb_components == 1
