import numpy as np

from pyorerun import BiorbdModel, PhaseRerun, MarkerTrajectories


def main():
    # building some time components
    nb_frames = 50
    nb_seconds = 1
    t_span = np.linspace(0, nb_seconds, nb_frames)

    model = BiorbdModel("models/Wu_Shoulder_Model_kinova_scaled_adjusted_2.bioMod")
    model.options.transparent_mesh = False

    # building some generalized coordinates
    q = np.zeros((model.model.nbQ(), nb_frames))
    q[10, :] = np.linspace(0, np.pi / 8, nb_frames)
    q[12, :] = np.linspace(0, np.pi / 3, nb_frames)
    q[11, :] = np.linspace(0, np.pi / 4, nb_frames)
    q[13, :] = np.linspace(0, np.pi / 8, nb_frames)
    q[14, :] = np.linspace(0, np.pi / 8, nb_frames)
    q[15, :] = np.linspace(0, np.pi / 8, nb_frames)

    import time

    # Initialize the animation with a time vector
    viz = PhaseRerun(t_span)

    # Example of how to add a persistent marker
    marker_trajectory = MarkerTrajectories(marker_names=["ULNA"], nb_frames=20)
    marker_trajectory = MarkerTrajectories(marker_names=["ULNA", "RADIUS"], nb_frames=None)

    # viz.add_animated_model(model, q, display_q=True)
    viz.add_animated_model(model, q, display_q=False, marker_trajectories=marker_trajectory)

    tic = time.time()
    viz.rerun("msk_model with chunks")
    toc = time.time()
    print(f"Time to run: {toc - tic}")

    tic = time.time()
    # viz.rerun_by_frame("msk_model frame by frame")
    toc = time.time()
    print(f"Time to run with chunks: {toc - tic}")


if __name__ == "__main__":
    main()
