import numpy as np


def display_frame(rr, animation_id):
    """Display the world reference frame"""
    rr.log(
        animation_id + "/X",
        rr.Arrows3D(
            origins=np.zeros(3),
            vectors=np.array([1, 0, 0]),
            colors=np.array([255, 0, 0]),
        ),
    )
    rr.log(
        animation_id + "/Y",
        rr.Arrows3D(
            origins=np.zeros(3),
            vectors=np.array([0, 1, 0]),
            colors=np.array([0, 255, 0]),
        ),
    )
    rr.log(
        animation_id + "/Z",
        rr.Arrows3D(
            origins=np.zeros(3),
            vectors=np.array([0, 0, 1]),
            colors=np.array([0, 0, 255]),
        ),
    )
    return rr
