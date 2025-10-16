import tkinter as tk
from tkinter import ttk

import numpy as np
import rerun as rr

from .model_components.model_updapter import ModelUpdater
from .model_components.model_display_options import DisplayModelOptions


class LiveModelAnimation:
    """
    A class to animate a biorbd model in rerun and update the joint angles in real-time.

    Attributes
    ----------
    counter : int
        A counter to keep track of the number of updates.
    model : BiorbdModel
        The biorbd model to animate.
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

    def __init__(self, model_updater: ModelUpdater, with_q_charts: bool = False):
        """
        Parameters
        ----------
        model_updater : ModelUpdater
            An instance of ModelUpdater to handle model updates.
        with_q_charts
            If True, q values will be plotted when the joint angles are updated.
        """

        self.counter = 0
        self.model_updater = model_updater
        self.model = self.model_updater.model

        self.q = np.zeros(self.model.nb_q)
        self.dof_sliders = []
        self.dof_slider_values = []
        self.update_functions = []
        self.with_q_charts = with_q_charts
        self.options = DisplayModelOptions()

    @classmethod
    def from_model(cls, model, with_q_charts: bool = False):
        model_updater = ModelUpdater("live_model", model)
        return cls(model_updater, with_q_charts)

    @classmethod
    def from_file(cls, model_path: str, with_q_charts: bool = False):
        return cls(ModelUpdater.from_file(model_path), with_q_charts)

    def update_viewer(self, event, dof_index: int):
        the_dof_idx, the_value = self.fetch_and_update_slider_value(event, dof_index)
        self.update_rerun_components(the_dof_idx, the_value)

    def fetch_and_update_slider_value(self, event, dof_index: int) -> tuple[int, float]:
        the_dof_idx = dof_index
        the_value = self.dof_sliders[the_dof_idx].get()
        # update the slider value q
        self.dof_slider_values[the_dof_idx].config(text=f"{the_value:.2f}")
        return the_dof_idx, the_value

    def update_rerun_components(self, the_dof_idx: int, the_value: float):
        self.q[the_dof_idx] = the_value
        # update counter
        self.counter += 1
        rr.set_time(timeline="step", sequence=self.counter)
        # Update the model
        self.update_model(self.q)
        # Update the q trajectories
        if self.with_q_charts:
            self.update_trajectories(self.q)

    def update_model(self, q: np.ndarray):
        self.model_updater.to_rerun(q)

    def update_trajectories(self, q: np.ndarray):
        q_ranges = self.model.q_ranges
        dof_names = self.model.dof_names
        for joint_idx in range(self.model.nb_q):
            name = f"q{joint_idx} - {dof_names[joint_idx]}"
            rr.log(f"{name}/min", rr.SeriesLines(colors=[255, 0, 0], names="min", widths=0.5))
            rr.log(f"{name}/max", rr.SeriesLines(colors=[255, 0, 0], names="max", widths=0.5))
            rr.log(f"{name}/value", rr.SeriesLines(colors=[0, 255, 0], names="q", widths=0.5))

            q_range = q_ranges[joint_idx]
            self.to_serie_line(name=name, min=q_range[0], max=q_range[-1], val=q[joint_idx])

    def to_serie_line(self, name: str, min: float, max: float, val: float):
        rr.log(f"{name}/min", rr.Scalars(min))
        rr.log(f"{name}/max", rr.Scalars(max))
        rr.log(f"{name}/value", rr.Scalars(val))

    def rerun(self, name: str = None):
        # update manually here
        rr.init(application_id=f"{self.model.name}" if name is None else name, spawn=True)
        self.update_rerun_components(the_dof_idx=0, the_value=0.0)
        self.create_window_with_sliders()

    def create_window_with_sliders(self):
        root = tk.Tk()
        root.title("Degree of Freedom q Sliders")

        for i in range(self.model.nb_q):
            self.update_functions.append(lambda event, idx=i: self.update_viewer(event, idx))
            dof_slider_label = ttk.Label(root, text=f"q{i} - {self.model.dof_names[i]}: ", anchor="w")
            dof_slider_label.grid(row=i, column=0, padx=10, pady=5, sticky="w")

            self.dof_sliders.append(
                ttk.Scale(root, from_=-5, to=5, orient="horizontal", command=self.update_functions[i], length=200)
            )
            self.dof_sliders[i].grid(row=i, column=1, padx=30, pady=5)
            self.dof_slider_values.append(ttk.Label(root, text="0"))
            self.dof_slider_values[i].grid(row=i, column=2, padx=10, pady=5)

        root.mainloop()
