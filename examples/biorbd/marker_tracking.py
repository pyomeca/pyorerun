import numpy as np

from pyorerun import BiorbdModel, RerunBiorbdPhase


def main():
    biorbd_model_path = "double_pendulum.bioMod"

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
    noisy_markers = biorbd_model.all_frame_markers(q + 0.1 * np.random.rand(2, nb_frames))

    # running the animation
    rerun_biorbd = RerunBiorbdPhase(biorbd_model)
    rerun_biorbd.set_tspan(t_span)
    rerun_biorbd.set_q(q)
    rerun_biorbd.add_marker_set(
        positions=noisy_markers,
        name="noisy_markers",
        labels=biorbd_model.marker_names,
        size=0.01,
        color=np.array([0, 0, 0]),
    )
    rerun_biorbd.rerun("animation")


if __name__ == "__main__":
    main()
