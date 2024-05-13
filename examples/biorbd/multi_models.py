"""
This example shows how to run multiple models in the same window. It also shows how to run multiple phases in different
windows. It also shows how to add experimental markers to the animation.
"""

import numpy as np
from numpy import random
from pyomeca import Markers

from pyorerun import BiorbdModel, MultiPhaseRerun


def building_some_q_and_t_span(nb_frames: int, nb_seconds: int) -> tuple[np.ndarray, np.ndarray]:
    t_span = np.linspace(0, nb_seconds, nb_frames)

    # building some generalized coordinates
    q = np.zeros((2, nb_frames))
    q[0, :] = np.linspace(0, 0.1, nb_frames)
    q[1, :] = np.linspace(0, 0.3, nb_frames)
    return q, t_span


def main():
    biorbd_model_path = "models/double_pendulum.bioMod"

    # building some time components
    nb_frames = 200
    nb_seconds = 1
    q0, t_span0 = building_some_q_and_t_span(nb_frames, nb_seconds)
    q1, t_span1 = building_some_q_and_t_span(20, 0.5)

    # loading biorbd model
    biorbd_model = BiorbdModel(biorbd_model_path)
    noisy_markers = np.zeros((3, biorbd_model.nb_markers, nb_frames))
    for i in range(nb_frames):
        noisy_markers[:, :, i] = biorbd_model.markers(q0[:, i]).T + random.random((3, biorbd_model.nb_markers)) * 0.1

    # running the animation
    rerun_biorbd = MultiPhaseRerun()

    rerun_biorbd.add_phase(t_span=t_span0, phase=0, window="animation")
    rerun_biorbd.add_animated_model(biorbd_model, q0, phase=0, window="animation")

    black_model = BiorbdModel(biorbd_model_path)
    black_model.options.mesh_color = (0, 0, 0)
    black_model.options.show_gravity = True

    rerun_biorbd.add_animated_model(black_model, q0 + 0.2, phase=0, window="animation")

    rerun_biorbd.add_phase(t_span=t_span0, phase=0, window="split_animation")
    rerun_biorbd.add_animated_model(biorbd_model, q0 + 0.2, phase=0, window="split_animation")

    rerun_biorbd.add_phase(t_span=t_span0[-1] + t_span1, phase=1)
    rerun_biorbd.add_animated_model(biorbd_model, q1, phase=1)

    rerun_biorbd.add_phase(t_span=t_span0[-1] + t_span1, phase=1, window="split_animation")
    rerun_biorbd.add_animated_model(biorbd_model, q1, phase=1, window="split_animation")

    markers = Markers(data=noisy_markers, channels=list(biorbd_model.marker_names))
    rerun_biorbd.add_xp_markers(
        name="noisy_markers",
        markers=markers,
        phase=0,
    )

    rerun_biorbd.rerun("multi_model_test")


if __name__ == "__main__":
    main()
