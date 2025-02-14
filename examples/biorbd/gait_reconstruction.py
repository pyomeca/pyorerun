import pickle
from pyomeca import Markers as PyoMarkers

from pyorerun import BiorbdModel, PhaseRerun


def main():

    # --- Load the experimental data from file --- #
    with open(f"data/reduced_exp_data.pkl", "rb") as file:
        data = pickle.load(file)
        q_reduced = data["q_reduced"]
        t_reduced = data["t_reduced"]
        force_plate_1_corners = data["force_plate_1_corners"]
        force_plate_2_corners = data["force_plate_2_corners"]
        f_ext_reduced_1 = data["f_ext_reduced_1"]
        f_ext_reduced_2 = data["f_ext_reduced_2"]
        markers_reduced = data["markers_reduced"]


    # --- Create the visualization --- #
    # Add the model
    biorbd_model_path = "models/walker.bioMod"
    model = BiorbdModel(biorbd_model_path)
    model.options.transparent_mesh = False
    viz = PhaseRerun(t_reduced)

    # Add experimental markers
    markers = PyoMarkers(data=markers_reduced, channels=list(model.marker_names))

    # Add force plates to the animation
    viz.add_force_plate(num=0, corners=force_plate_1_corners)
    viz.add_force_plate(num=1, corners=force_plate_2_corners)
    viz.add_force_data(
        num=0,
        force_origin=f_ext_reduced_1[:3, :],
        force_vector=f_ext_reduced_1[6:9, :],
    )
    viz.add_force_data(
        num=1,
        force_origin=f_ext_reduced_2[:3, :],
        force_vector=f_ext_reduced_2[6:9, :],
    )

    # Add the kinematics
    viz.add_animated_model(model, q_reduced, tracked_markers=markers)

    # Play
    viz.rerun("Experimental data with kinematics reconstruction")


if __name__ == "__main__":
    main()
