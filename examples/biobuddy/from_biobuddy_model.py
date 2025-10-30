"""
To run this example, you need to install biobuddy:
    `pip install biobuddy`
    or
    `conda-forge install -c conda-forge biobuddy`
"""

import numpy as np
import biobuddy

from pyorerun import PhaseRerun, BiobuddyModel, DisplayModelOptions


def main():

    # building some time components
    nb_frames = 50
    nb_seconds = 1
    t_span = np.linspace(0, nb_seconds, nb_frames)

    # Creating the model
    model_path = "../biorbd/models/Wu_Shoulder_Model_kinova_scaled_adjusted_2.bioMod"
    biobuddy_model = biobuddy.BiomechanicalModelReal().from_biomod(
        model_path
    )
    biobuddy_model.change_mesh_directories("../biorbd/models/Geometry_cleaned")
    display_options = DisplayModelOptions()
    prr_model = BiobuddyModel.from_biobuddy_object(biobuddy_model, options=display_options)

    # building some generalized coordinates
    q = np.zeros((biobuddy_model.nb_q, nb_frames))
    q[10, :] = np.linspace(0, np.pi / 8, nb_frames)
    q[12, :] = np.linspace(0, np.pi / 3, nb_frames)
    q[11, :] = np.linspace(0, np.pi / 4, nb_frames)
    q[13, :] = np.linspace(0, np.pi / 8, nb_frames)
    q[14, :] = np.linspace(0, np.pi / 8, nb_frames)
    q[15, :] = np.linspace(0, np.pi / 8, nb_frames)

    # Animate the model
    viz = PhaseRerun(t_span)
    viz.add_animated_model(prr_model, q)
    # viz.rerun("msk_model")
    viz.rerun_by_frame("msk_model")


if __name__ == "__main__":
    main()
