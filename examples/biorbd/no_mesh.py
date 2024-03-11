import numpy as np
from pyomeca import Markers as PyoMarkers

from pyorerun import BiorbdModelNoMesh, PhaseRerun


def main():
    biorbd_model_path = "models/no_mesh_no_marker.s2mMod"

    # building some time components
    nb_frames = 200
    nb_seconds = 1
    t_span = np.linspace(0, nb_seconds, nb_frames)

    # loading biorbd model
    biorbd_model = BiorbdModelNoMesh(biorbd_model_path)

    # building some noisy markers
    nb_random_markers = 10
    noisy_markers = np.zeros((3, nb_random_markers, nb_frames))
    for i in range(nb_frames):
        noisy_markers[:, :, i] = np.random.rand(3, nb_random_markers)

    # running the animation
    rerun_biorbd = PhaseRerun(t_span)
    q = np.zeros((biorbd_model.model.nbQ(), nb_frames))  # no movement
    rerun_biorbd.add_animated_model(biorbd_model, q)
    markers = PyoMarkers(data=noisy_markers, channels=list(biorbd_model.marker_names))
    rerun_biorbd.add_xp_markers(
        name="noisy_markers",
        markers=markers,
    )
    rerun_biorbd.rerun("animation")


if __name__ == "__main__":
    main()
