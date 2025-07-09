import pickle
import numpy as np

from pyorerun import BiorbdModel, PhaseRerun, PyoMuscles, PyoMarkers


def main():

    # --- Load the experimental data from file --- #
    with open(f"data/gait_reconstruction.pkl", "rb") as file:
        data = pickle.load(file)
        q = data["q"]
        t_span = data["t_span"]
        force_plate_1_corners = data["force_plate_1_corners"]
        force_plate_2_corners = data["force_plate_2_corners"]
        f_ext_1 = data["f_ext_1"]
        f_ext_2 = data["f_ext_2"]
        markers = data["markers"]

    # --- Create the visualization --- #
    # Add the model
    biorbd_model_path = "models/walker.bioMod"
    model = BiorbdModel(biorbd_model_path)
    model.options.transparent_mesh = False
    model.options.show_marker_labels = False
    model.options.show_contact_labels = False
    model.options.show_center_of_mass_labels = False
    model.options.show_muscle_labels = False
    model.options.show_ligament_labels = False

    viz = PhaseRerun(t_span)

    # Add experimental markers
    pyomarkers = PyoMarkers(data=markers, channels=list(model.marker_names), show_labels=False)

    # Add experimental emg
    nb_muscles = model.nb_muscles
    nb_frames = q.shape[1]
    fake_emg = np.ones((nb_muscles, nb_frames))  # Fake EMG data for demonstration
    for i_muscle in range(nb_muscles):
        fake_emg[i_muscle, :] = np.linspace(0.01, 1, nb_frames)
    pyoemg = PyoMuscles(
        data=fake_emg,
        muscle_names=list(model.muscle_names),
        mvc=np.ones((nb_muscles,)),  # Fake MVC values
        colormap="viridis",
    )

    # Add force plates to the animation
    viz.add_force_plate(num=0, corners=force_plate_1_corners)
    viz.add_force_plate(num=1, corners=force_plate_2_corners)
    viz.add_force_data(
        num=0,
        force_origin=f_ext_1[:3, :],
        force_vector=f_ext_1[6:9, :],
    )
    viz.add_force_data(
        num=1,
        force_origin=f_ext_2[:3, :],
        force_vector=f_ext_2[6:9, :],
    )

    # Add the kinematics
    viz.add_animated_model(
        model, q
    )  # This line is just to test the model without markers (but is not necessary for the example to work)
    viz.add_animated_model(model, q, tracked_markers=pyomarkers, muscle_activations_intensity=pyoemg)

    # Play
    viz.rerun("Experimental data with kinematics reconstruction")


if __name__ == "__main__":
    main()
