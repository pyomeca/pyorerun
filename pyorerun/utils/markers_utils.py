import numpy as np

from ..model_interfaces import AbstractModel
from ..pyomarkers import PyoMarkers


def sort_markers_to_a_predefined_order(
    tracked_markers: PyoMarkers,
    tracked_marker_names: tuple[str, ...] | list[str],
    model_marker_names: tuple[str, ...] | list[str],
) -> PyoMarkers:
    """
    This function reorders the tracked markers to match the order in which they are declared in the model.

    Parameters
    ----------
    tracked_markers : PyoMarkers
        The tracked markers to reorder.
    tracked_marker_names : tuple[str] | list[str]
        The names of the tracked markers.
    model_marker_names : tuple[str] | list[str]
        The names of the markers as declared in the model.
    """
    current_tracked_markers = tracked_markers.to_numpy()
    reordered_markers = np.zeros_like(current_tracked_markers)
    for i_marker, model_marker_name in enumerate(model_marker_names):
        name_index = tracked_marker_names.index(model_marker_name)
        reordered_markers[:, i_marker, :] = current_tracked_markers[:, name_index, :]

    reordered_tracked_markers = PyoMarkers(reordered_markers, marker_names=list(model_marker_names))
    return reordered_tracked_markers


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
        tracked_markers = sort_markers_to_a_predefined_order(tracked_markers, tracked_marker_names, model.marker_names)

    return tracked_markers
