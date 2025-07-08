import numpy as np

from pyorerun import BiorbdModel, PhaseRerun, PyoMarkers


def main():
    biorbd_model_path = "models/double_pendulum.bioMod"

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
    markers = PyoMarkers(data=noisy_markers, channels=list(biorbd_model.marker_names))
    rerun_biorbd.add_animated_model(biorbd_model, q, tracked_markers=markers)

    # rerun_biorbd.add_xp_markers(
    #     name="noisy_markers",
    #     markers=markers,
    # )
    rerun_biorbd.rerun("animation")


if __name__ == "__main__":
    main()
