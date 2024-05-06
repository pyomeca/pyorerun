import tkinter as tk
from tkinter import ttk

import numpy as np
import rerun as rr

from pyorerun import BiorbdModel, PhaseRerun


def update_model(event):
    global dof_slider
    q = np.zeros((model.model.nbQ() - 6))
    q_root = np.zeros(6)
    q[7] = dof_slider.get()  # Update the selected degree of freedom
    q = np.concatenate((q_root, q))
    for i, rr_model in enumerate(viz.biorbd_models.rerun_models):
        rr_model.to_rerun(q)


def main():
    # building some time components
    global model, viz, dof_slider

    nb_frames = 1
    nb_seconds = 0
    t_span = np.linspace(0, nb_seconds, nb_frames)

    model = BiorbdModel("models/Wu_Shoulder_Model_kinova_scaled_adjusted_2.bioMod")

    # building some generalized coordinates
    q = np.zeros((model.model.nbQ(), nb_frames))

    viz = PhaseRerun(t_span)
    viz.add_animated_model(model, q)
    rr.init(f"hello_{0}", spawn=True)

    # Create the Tkinter window
    root = tk.Tk()
    root.title("Degree of Freedom Slider")

    # Slider for selecting the degree of freedom
    dof_slider_label = ttk.Label(root, text="Select Degree of Freedom:")
    dof_slider_label.grid(row=0, column=0, padx=10, pady=5)
    dof_slider = ttk.Scale(root, from_=0, to=model.model.nbQ() - 7, orient="horizontal", command=update_model)
    dof_slider.grid(row=0, column=1, padx=10, pady=5)

    root.mainloop()


if __name__ == "__main__":
    main()
