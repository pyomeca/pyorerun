"""
This example shows how to run multiple models in the same window. It also shows how to run multiple phases in different
windows. It also shows how to add experimental markers to the animation.
"""

import numpy as np
from numpy import random

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

    depth_image = 65535 * np.ones((200, 300), dtype=np.int64)
    depth_image[50:150, 50:150] = 20000
    depth_image[130:180, 100:280] = 45000
    # add noise and repear on nb_frames
    depth_image = np.tile(depth_image[:, :, np.newaxis], (1, 1, nb_frames))
    depth_image += random.random_integers(low=0, high=1000, size=depth_image.shape)

    # running the animation
    multi_rerun = MultiPhaseRerun()

    multi_rerun.add_phase(t_span=t_span0, phase=0, window="animation")
    multi_rerun.add_animated_model(biorbd_model, q0, phase=0, window="animation")
    multi_rerun.add_depth_image(name="depth", depth_image=depth_image, phase=0, window="animation")

    multi_rerun.rerun("multi_model_test")


if __name__ == "__main__":
    main()
