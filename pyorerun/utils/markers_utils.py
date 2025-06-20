import numpy as np
from pyomeca import Markers as PyoMarkers

from ..model_interfaces import AbstractModel


def check_and_adjust_markers(model: AbstractModel, tracked_markers: PyoMarkers) -> PyoMarkers:
    """
    Check if the markers of the model and the tracked markers are consistent.
    Plus, if the names are ordered differently, reorder the tracked markers accordingly.
    """

    shape_of_markers_is_not_consistent = model.nb_markers != tracked_markers.shape[1]
    if shape_of_markers_is_not_consistent:
        raise ValueError(
            f"The markers of the model and the tracked markers are inconsistent. "
            f"They must have the same names and shape.\n"
            f"Current markers are {model.marker_names} and\n tracked markers: {tracked_markers.channel.data.tolist()}."
        )

    tracked_marker_names = tuple(tracked_markers.channel.data.tolist())
    names_are_ordered_differently = model.marker_names != tracked_marker_names
    if names_are_ordered_differently:
        # Replace the markers in the right order based on the names provided in PyoMarkers
        reordered_markers = np.zeros_like(tracked_markers.to_numpy())
        for marker in model.marker_names:
            if marker not in tracked_marker_names:
                raise RuntimeError("The marker names in the model and the tracked markers do not match.")
            marker_index = tracked_marker_names.index(marker)
            reordered_markers[:, marker_index, :] = tracked_markers.to_numpy()[:, model.marker_names.index(marker), :]
        tracked_markers = PyoMarkers(reordered_markers, channels=list(model.marker_names))

    return tracked_markers
