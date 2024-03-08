import numpy as np

from pyorerun import BiorbdModel, PhaseRerun


def main():
    # building some time components
    nb_frames = 10
    nb_seconds = 1
    t_span = np.linspace(0, nb_seconds, nb_frames)

    model = BiorbdModel("models/shoulder_model.bioMod")
    model2 = BiorbdModel("models/double_pendulum.bioMod")
    print(model.model)
    # building some generalized coordinates
    q = np.zeros((model.model.nbQ(), nb_frames))
    q[0, :] = np.linspace(0, 0.02, nb_frames)
    q[1, :] = np.linspace(0, 0.03, nb_frames)
    q[2, :] = np.linspace(0, 0.04, nb_frames)
    q[7, :] = np.linspace(0, 0.05, nb_frames)

    viz = PhaseRerun(t_span)
    # viz.add_animated_model(model2, q[:2, :])
    viz.add_animated_model(model, q)
    viz.rerun("msk_model")


if __name__ == "__main__":
    main()
