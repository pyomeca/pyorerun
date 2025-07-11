import numpy as np

from ..model_interfaces import AbstractModel
from ..pyomarkers import PyoMarkers


def check_and_adjust_markers(model: AbstractModel, tracked_markers: PyoMarkers) -> PyoMarkers:
    """
    Check if the markers of the model and the tracked markers are consistent.
    Plus, if the names are ordered differently, reorder the tracked markers accordingly.
    """

    shape_of_markers_is_not_consistent = model.nb_markers != tracked_markers.shape[1]
    tracked_marker_names = tuple(tracked_markers.marker_names)
    if shape_of_markers_is_not_consistent:
        raise ValueError(
            f"The markers of the model and the tracked markers are inconsistent. "
            f"They must have the same names and shape.\n"
            f"Current markers are {model.marker_names} and\n tracked markers: {tracked_marker_names}."
        )

    tracked_marker_names_are_not_all_in_model_markers = any(
        marker not in tracked_marker_names for marker in model.marker_names
    )

    if tracked_marker_names_are_not_all_in_model_markers:
        raise ValueError(
            f"The markers of the model and the tracked markers are inconsistent. "
            f"Tracked markers {tracked_marker_names} must contain all the markers in the model {model.marker_names}."
        )

    names_are_ordered_differently = model.marker_names != tracked_marker_names
    if names_are_ordered_differently:
        # Replace the markers in the right order based on the names provided in PyoMarkers
        reordered_markers = np.zeros_like(tracked_markers.to_numpy())
        for i_marker, marker in enumerate(model.marker_names):
            marker_index = tracked_marker_names.index(marker)
            reordered_markers[:, i_marker, :] = tracked_markers.to_numpy()[:, marker_index, :]
        tracked_markers = PyoMarkers(reordered_markers, marker_names=list(model.marker_names))

    return tracked_markers
