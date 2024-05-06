import tkinter as tk
from tkinter import ttk

import numpy as np
import rerun as rr

from pyorerun import BiorbdModel, PhaseRerun


class LiveModelAnimation:
    """
    A class to animate a biorbd model in rerun and update the joint angles in real-time.

    Attributes
    ----------
    counter : int
        A counter to keep track of the number of updates.
    model : BiorbdModel
        The biorbd model to animate.
    biorbd_model : biorbd.Model
        The underlying biorbd model.
    q : np.ndarray
        The current joint angles.
    dof_sliders : list
        The sliders for adjusting the joint angles.
    dof_slider_values : list
        The labels for displaying the current slider values.
    update_functions : list
        The functions for updating the model when a slider is moved.
    with_q_charts : bool
        Whether to plot q values when the joint angles are updated.
    """

    def __init__(self, model_path: str, with_q_charts: bool = False):
        """
        Parameters
        ----------
        model_path: str
            The path to the bioMod file.
        with_q_charts
            If True, q values will be plotted when the joint angles are updated.
        """

        self.counter = 0
        self.model = BiorbdModel(model_path)
        self.biorbd_model = self.model.model
        self.q = np.zeros(self.biorbd_model.nbQ())
        self.dof_sliders = []
        self.dof_slider_values = []
        self.update_functions = []
        self.with_q_charts = with_q_charts

    def update_viewer(self, event, dof_index: int):
        the_dof_idx = dof_index
        the_value = self.dof_sliders[the_dof_idx].get()
        # update the slider value q
        self.dof_slider_values[the_dof_idx].config(text=f"{the_value:.2f}")
        # update joint angles q
        self.q[the_dof_idx] = the_value
        # update counter
        self.counter += 1
        rr.set_time_sequence(timeline="step", sequence=self.counter)
        # Update the model
        self.update_model(self.q)
        # Update the q trajectories
        if self.with_q_charts:
            self.update_trajectories(self.q)

    def update_model(self, q: np.ndarray):
        self.viz.biorbd_models.rerun_models[0].to_rerun(q)

    def update_trajectories(self, q: np.ndarray):
        q_ranges = [q_range for segment in self.biorbd_model.segments() for q_range in segment.QRanges()]
        dof_names = self.model.dof_names
        for joint_idx in range(self.biorbd_model.nbQ()):
            name = f"q{joint_idx} - {dof_names[joint_idx]}"
            rr.log(f"{name}/min", rr.SeriesLine(color=[255, 0, 0], name="min", width=0.5))
            rr.log(f"{name}/max", rr.SeriesLine(color=[255, 0, 0], name="max", width=0.5))
            rr.log(f"{name}/value", rr.SeriesLine(color=[0, 255, 0], name="q", width=0.5))

            q_range = q_ranges[joint_idx]
            self.to_serie_line(name=name, min=q_range.min(), max=q_range.max(), val=q[joint_idx])

    def to_serie_line(self, name: str, min: float, max: float, val: float):
        rr.log(f"{name}/min", rr.Scalar(min))
        rr.log(f"{name}/max", rr.Scalar(max))
        rr.log(f"{name}/value", rr.Scalar(val))

    def rerun(self, name: str = None):
        # Building a fake pyorerun animation
        nb_frames = 1
        nb_seconds = 0
        t_span = np.linspace(0, nb_seconds, nb_frames)
        q = np.zeros((self.model.model.nbQ(), nb_frames))
        self.viz = PhaseRerun(t_span)
        self.viz.add_animated_model(self.model, q)

        # update manually here
        rr.init(application_id=f"{self.model.name}" if name is None else name, spawn=True)
        self.create_window_with_sliders()

    def create_window_with_sliders(self):
        root = tk.Tk()
        root.title("Degree of Freedom q Sliders")

        for i in range(self.model.model.nbQ()):
            self.update_functions.append(lambda event, idx=i: self.update_viewer(event, idx))
            dof_slider_label = ttk.Label(root, text=f"q{i} - {self.model.dof_names[i]}: ", anchor="w")
            dof_slider_label.grid(row=i, column=0, padx=10, pady=5, sticky="w")

            self.dof_sliders.append(
                ttk.Scale(root, from_=-5, to=5, orient="horizontal", command=self.update_functions[i])
            )
            self.dof_sliders[i].grid(row=i, column=1, padx=30, pady=5)
            self.dof_slider_values.append(ttk.Label(root, text="0"))
            self.dof_slider_values[i].grid(row=i, column=2, padx=10, pady=5)

        root.mainloop()
