import numpy as np
import rerun as rr
from pyomeca import Markers as PyoMarkers

from .abstract.q import QProperties
from .biorbd_components.model_interface import BiorbdModel
from .biorbd_phase import BiorbdRerunPhase
from .timeless import Gravity, Floor, ForcePlate
from .timeless_components import TimelessRerunPhase
from .xp_components import MarkersXp, TimeSeriesQ, ForceVector, Video
from .xp_phase import XpRerunPhase


class PhaseRerun:
    """
    A class to animate a biorbd model in rerun.

    Attributes
    ----------
    phase : int
        The phase number of the animation.
    name : str
        The name of the animation.
    t_span : np.ndarray
        The time span of the animation, such as the time instant of each frame.
    biorbd_models : BiorbdRerunPhase
        The biorbd models to animate.
    xp_data : XpRerunPhase
        The experimental data to display.
    timeless_components : list
        The components to display at the begin of the phase but stay until the end of this phase.
        This not a true timeless in the sens of rerun, as if a new phase is created, the timeless components will be cleared.
    """

    def __init__(self, t_span: np.ndarray, phase: int = 0, window: str = None):
        """
        Parameters
        ----------
        t_span: np.ndarray
            The time span of the animation, such as the time instant of each frame.
        phase: int
            The phase number of the animation, zero by default.
        """
        self.phase = phase
        self.name = f"animation_phase_{self.phase}"
        if window:
            self.name = f"{window}/{self.name}"

        # same t_span for the phase
        self.t_span = t_span

        self.biorbd_models = BiorbdRerunPhase(name=self.name, phase=phase)
        self.xp_data = XpRerunPhase(name=self.name, phase=phase)
        self.timeless_components = TimelessRerunPhase(name=self.name, phase=phase)

    def add_animated_model(
        self,
        biomod: BiorbdModel,
        q: np.ndarray,
        tracked_markers: PyoMarkers = None,
        show_tracked_marker_labels: bool = True,
        display_q: bool = False,
    ) -> None:
        """
        Add an animated model to the phase.

        Parameters
        ----------
        biomod: BiorbdModel
            The biorbd model to display.
        q: np.ndarray
            The generalized coordinates of the model.
        tracked_markers: PyoMarkers
            The markers to display, and sets a link between the model markers and the tracked markers.
        show_tracked_marker_labels: bool
            Whether to display the tracked markers labels.
        display_q: bool
            Whether to display the generalized coordinates q in charts.
        """
        shape_is_not_consistent = q.shape[1] != self.t_span.shape[0]
        if shape_is_not_consistent:
            raise ValueError(
                f"The shapes of q and tspan are inconsistent. "
                f"They must have the same length."
                f"Current shapes are q: {q.shape[1]} and tspan: {self.t_span.shape}."
            )

        if tracked_markers is None:
            self.biorbd_models.add_animated_model(biomod, q)
        else:
            self.biorbd_models.add_animated_model(biomod, q, tracked_markers.to_numpy()[:3, :, :])
            self.__add_tracked_markers(biomod, tracked_markers, show_tracked_marker_labels)

        if display_q:
            self.add_q(
                f"{biomod.name}_q",
                q,
                ranges=biomod.q_ranges,
                dof_names=biomod.dof_names,
            )
        if biomod.options.show_gravity:
            self.timeless_components.add_component(
                Gravity(name=f"{self.name}/{self.biorbd_models.nb_models}_{biomod.name}", vector=biomod.gravity)
            )

    def __add_tracked_markers(
        self, biomod: BiorbdModel, tracked_markers: PyoMarkers, show_tracked_marker_labels: bool
    ) -> None:
        """Add the tracked markers to the phase."""
        shape_of_markers_is_not_consistent = biomod.nb_markers != tracked_markers.shape[1]
        names_are_ordered_differently = biomod.marker_names != tuple(tracked_markers.channel.data.tolist())
        if shape_of_markers_is_not_consistent or names_are_ordered_differently:
            raise ValueError(
                f"The markers of the model and the tracked markers are inconsistent. "
                f"They must have the same names and shape.\n"
                f"Current markers are {biomod.marker_names} and\n tracked markers: {tracked_markers.channel.data.tolist()}."
            )

        self.add_xp_markers(f"{biomod.name}_tracked_markers", tracked_markers, show_tracked_marker_labels)

    def add_xp_markers(self, name, markers: PyoMarkers, show_tracked_marker_labels: bool = True) -> None:
        """
        Add an animated model to the phase.

        Parameters
        ----------
        name: str
            The name of the markers set.
        markers: PyoMarkers
            The experimental data to display.
        show_tracked_marker_labels: bool
            Whether to display the tracked markers labels.
        """
        if markers.shape[2] != self.t_span.shape[0]:
            raise ValueError(
                f"The shapes of markers and tspan are inconsistent. "
                f"They must have the same length."
                f"Current shapes are markers: {markers.shape} and tspan: {self.t_span.shape}."
            )
        self.xp_data.add_data(
            MarkersXp(name=f"{self.name}/{name}", markers=markers, show_labels=show_tracked_marker_labels)
        )

    def add_q(
        self,
        name: str,
        q: np.ndarray,
        dof_names: tuple[str, ...],
        ranges: tuple[tuple[float, float], ...] = None,
    ) -> None:
        """
        Add the generalized coordinates to be displqyed.

        Parameters
        ----------
        name: str
            The name of the q set.
        q: np.ndarray
            The generalized coordinates to display of shape (nb_q, nb_frames).
        dof_names: tuple[str, ...]
            The names of the degrees of freedom.
        ranges: tuple[tuple[float, float], ...]
            The ranges of the q values, min and max.
        """
        if q.shape[1] != self.t_span.shape[0]:
            raise ValueError(
                f"The shapes of q and tspan are inconsistent. "
                f"They must have the same length."
                f"Current shapes are q: {q.shape[1]} and tspan: {self.t_span.shape}."
            )

        self.xp_data.add_data(
            TimeSeriesQ(name=f"{self.name}/{name}", q=q, properties=QProperties(joint_names=dof_names, ranges=ranges))
        )

    def add_floor(self, square_width: float = None, height_offset: float = None, subsquares: int = None) -> None:
        """Add a floor to the phase."""
        self.timeless_components.add_component(
            Floor(name=f"{self.name}", square_width=square_width, height_offset=height_offset, subsquares=subsquares)
        )

    def add_force_plate(self, num: int, corners: np.ndarray) -> None:
        """Add a force plate to the phase."""
        self.timeless_components.add_component(ForcePlate(name=f"{self.name}", num=num, corners=corners))

    def add_force_data(self, num: int, force_origin: np.ndarray, force_vector: np.ndarray) -> None:
        """Add a force data to the phase."""
        if force_origin.shape[1] != self.t_span.shape[0] or force_vector.shape[1] != self.t_span.shape[0]:
            raise ValueError(
                f"The shapes of force_origin/force_vector and tspan are inconsistent. "
                f"They must have the same length."
                f"Got force_origin: {force_origin.shape[1]}, force_vector: {force_vector.shape[1]} and tspan: {self.t_span.shape}."
            )

        self.xp_data.add_data(
            ForceVector(name=f"{self.name}", num=num, vector_origins=force_origin, vector_magnitudes=force_vector)
        )

    def add_video(self, name, video_array: np.ndarray) -> None:
        """Add a video to the phase."""
        if video_array.shape[0] != self.t_span.shape[0]:
            raise ValueError("The video array and tspan are inconsistent. They must have the same length.")

        self.xp_data.add_data(Video(name=f"{self.name}/{name}", video_array=video_array))

    def rerun_by_frame(
        self, name: str = "animation_phase", init: bool = True, clear_last_node: bool = False, notebook: bool = False
    ) -> None:
        if init:
            rr.init(f"{name}_{self.phase}", spawn=True if not notebook else False)

        frame = 0
        rr.set_time_seconds("stable_time", self.t_span[frame])
        self.timeless_components.to_rerun()
        self.biorbd_models.to_rerun(frame)
        self.xp_data.to_rerun(frame)

        for frame, t in enumerate(self.t_span[1:]):
            rr.set_time_seconds("stable_time", t)
            self.biorbd_models.to_rerun(frame + 1)
            self.xp_data.to_rerun(frame + 1)

        if clear_last_node:
            for component in [
                *self.biorbd_models.component_names,
                *self.xp_data.component_names,
                *self.timeless_components.component_names,
            ]:
                rr.log(component, rr.Clear(recursive=False))

    def rerun(
        self, name: str = "animation_phase", init: bool = True, clear_last_node: bool = False, notebook: bool = False
    ) -> None:
        if init:
            rr.init(f"{name}_{self.phase}", spawn=True if not notebook else False)

        frame = 0
        rr.set_time_seconds("stable_time", self.t_span[frame])
        self.timeless_components.to_rerun()
        self.biorbd_models.initialize()
        self.xp_data.initialize()

        times = [rr.TimeSecondsColumn("stable_time", self.t_span)]

        for name, chunk in self.xp_data.to_chunk().items():
            rr.send_columns(
                name,
                times=times,
                components=chunk,
            )

        for name, chunk in self.biorbd_models.to_chunk().items():
            rr.send_columns(
                name,
                times=times,
                components=chunk,
            )

        if clear_last_node:
            rr.set_time_seconds("stable_time", self.t_span[-1])
            for component in [
                *self.biorbd_models.component_names,
                *self.xp_data.component_names,
                *self.timeless_components.component_names,
            ]:
                rr.log(component, rr.Clear(recursive=False))
