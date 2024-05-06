import tkinter as tk
from tkinter import ttk

import numpy as np
import rerun as rr

from pyorerun import BiorbdModel, PhaseRerun


def update_model(event, dof_index: int):
    global dof_sliders, dof_slider_values, counter, q_squeeze

    the_dof_idx = dof_index
    the_value = dof_sliders[the_dof_idx].get()
    print(f"q{the_dof_idx} = {the_value}")

    dof_slider_values[the_dof_idx].config(text=f"{the_value:.2f}")

    q_squeeze[the_dof_idx] = the_value

    counter += 1
    rr.set_time_sequence(timeline="step", sequence=counter)
    q = q_squeeze

    # Update the model
    for i, rr_model in enumerate(viz.biorbd_models.rerun_models):
        rr_model.to_rerun(q)
        biorbd_model = rr_model.model.model

    # Update the q trajectories
    q_ranges = [q_range for segment in biorbd_model.segments() for q_range in segment.QRanges()]
    for joint_idx in range(biorbd_model.nbQ()):
        name = f"q{joint_idx} - {rr_model.model.dof_names[joint_idx]}"
        rr.log(f"{name}/min", rr.SeriesLine(color=[255, 0, 0], name="min", width=0.5))
        rr.log(f"{name}/max", rr.SeriesLine(color=[255, 0, 0], name="max", width=0.5))
        rr.log(f"{name}/value", rr.SeriesLine(color=[0, 255, 0], name="q", width=0.5))

        q_range = q_ranges[joint_idx]
        to_serie_line(name=name, min=q_range.min(), max=q_range.max(), val=q_squeeze[joint_idx])


def to_serie_line(name, min, max, val):
    rr.log(f"{name}/min", rr.Scalar(min))
    rr.log(f"{name}/max", rr.Scalar(max))
    rr.log(f"{name}/value", rr.Scalar(val))


def main():
    # building some time components
    global model, viz, dof_sliders, dof_slider_values, counter, q_squeeze

    counter = 0
    nb_frames = 1
    nb_seconds = 0

    t_span = np.linspace(0, nb_seconds, nb_frames)

    model = BiorbdModel("models/Wu_Shoulder_Model_kinova_scaled_adjusted_2.bioMod")
    q_squeeze = np.zeros(model.model.nbQ())

    # building some generalized coordinates
    q = np.zeros((model.model.nbQ(), nb_frames))

    viz = PhaseRerun(t_span)
    viz.add_animated_model(model, q)
    rr.init(f"hello_{0}", spawn=True)

    # Create the Tkinter window
    root = tk.Tk()
    root.title("Degree of Freedom q Sliders")

    dof_sliders = []
    dof_slider_values = []
    update_functions = []

    for i in range(model.model.nbQ()):
        update_functions.append(lambda event, idx=i: update_model(event, idx))
        dof_slider_label = ttk.Label(root, text=f"q{i} - {model.dof_names[i]}: ", anchor="w")
        dof_slider_label.grid(row=i, column=0, padx=10, pady=5, sticky="w")

        dof_sliders.append(ttk.Scale(root, from_=-5, to=5, orient="horizontal", command=update_functions[i]))
        dof_sliders[i].grid(row=i, column=1, padx=20, pady=5)
        dof_slider_values.append(ttk.Label(root, text="0"))
        dof_slider_values[i].grid(row=i, column=2, padx=10, pady=5)

    root.mainloop()


if __name__ == "__main__":
    main()
