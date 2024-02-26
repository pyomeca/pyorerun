import numpy as np

from pyorerun import BiorbdModel, RerunBiorbd


def building_some_q_and_t_span(nb_frames: int, nb_seconds: int) -> tuple[np.ndarray, np.ndarray]:
    t_span = np.linspace(0, nb_seconds, nb_frames)

    # building some generalized coordinates
    q = np.zeros((2, nb_frames))
    q[0, :] = np.linspace(0, 0.1, nb_frames)
    q[1, :] = np.linspace(0, 0.3, nb_frames)
    return q, t_span


def main():
    biorbd_model_path = "double_pendulum.bioMod"

    # building some time components
    nb_frames = 200
    nb_seconds = 1
    q0, t_span0 = building_some_q_and_t_span(nb_frames, nb_seconds)
    q1, t_span1 = building_some_q_and_t_span(20, 0.5)

    # loading biorbd model
    biorbd_model = BiorbdModel(biorbd_model_path)
    noisy_markers = biorbd_model.all_frame_markers(q0 + 0.1 * np.random.rand(2, nb_frames))

    # running the animation
    rerun_biorbd = RerunBiorbd()
    rerun_biorbd.add_phase(biorbd_model, t_span0, q0, phase=0)
    rerun_biorbd.add_phase(biorbd_model, t_span0, q0 + 0.2, phase=0)
    rerun_biorbd.add_phase(biorbd_model, t_span0[-1] + t_span1, q1, phase=1)
    rerun_biorbd.add_phase(biorbd_model, t_span0[-1] + t_span1, q1, phase=1, window="split_animation")

    rerun_biorbd.add_marker_set(noisy_markers, "noisy_markers", color=np.array([255, 0, 0]), phase=0)

    rerun_biorbd.rerun()


if __name__ == "__main__":
    main()
