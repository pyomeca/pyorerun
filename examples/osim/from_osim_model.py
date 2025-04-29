import opensim
import numpy as np

from pyorerun import OsimModel, PhaseRerun


def main():
    # building some time components
    nb_frames = 50
    nb_seconds = 1
    t_span = np.linspace(0, nb_seconds, nb_frames)

    biorbd_model = opensim.Model("models/Wu_Shoulder_Model_kinova_scaled_adjusted_2.bioMod")
    prr_model = OsimModel.from_osim_object(biorbd_model)

    # building some generalized coordinates
    q = np.zeros((biorbd_model.getNumCoordinates(), nb_frames))
    q[10, :] = np.linspace(0, np.pi / 8, nb_frames)
    q[12, :] = np.linspace(0, np.pi / 3, nb_frames)
    q[11, :] = np.linspace(0, np.pi / 4, nb_frames)
    q[13, :] = np.linspace(0, np.pi / 8, nb_frames)
    q[14, :] = np.linspace(0, np.pi / 8, nb_frames)
    q[15, :] = np.linspace(0, np.pi / 8, nb_frames)

    viz = PhaseRerun(t_span)
    viz.add_animated_model(prr_model, q)
    viz.rerun("msk_model")


if __name__ == "__main__":
    main()
