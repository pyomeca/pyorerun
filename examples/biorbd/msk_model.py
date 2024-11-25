import numpy as np

from pyorerun import BiorbdModel, PhaseRerun


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

    viz = PhaseRerun(t_span)
    # viz.add_animated_model(model, q, display_q=True)
    viz.add_animated_model(model, q, display_q=False)
    tic = time.time()
    viz.rerun("msk_model")
    toc = time.time()
    print(f"Time to run: {toc - tic}")

    tic = time.time()
    viz.rerun_with_chunks("msk_model with chunk")
    toc = time.time()
    print(f"Time to run with chunks: {toc - tic}")


if __name__ == "__main__":
    main()
