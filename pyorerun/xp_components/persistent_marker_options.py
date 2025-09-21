import numpy as np

from ..abstract.abstract_class import PersistentComponent


class PersistentMarkerOptions(PersistentComponent):
    def __init__(self, marker_names: list[str], nb_frames: int | None = None) -> None:
        """
        Initialization of a marker trajectory

        Parameters
        ----------
        marker_names: str
            The name of the markers to display a trajectory for.
        nb_frames: int | None
            The number of frames to display the trajectory for. If None, all previous frames will be displayed.
            Example: nb_frames=20 means that the position of the marker for the last 20 frames will be displayed at each current frame.
        """
        self.marker_names = marker_names
        self.nb_frames = nb_frames

    def marker_to_keep(self, model_markers: np.ndarray, model_markers_names: list[str]) -> tuple[np.ndarray, list[str]]:
        """
        Keep only the markers to compute a marker trajectory for.

        Parameters
        ----------
        model_markers: np.ndarray
            All model marker positions (3, N_markers, N_frames)
        model_markers_names: list[str]
            All model markers names (N_markers)
        """
        trial_nb_frames = model_markers.shape[2]
        markers_to_keep = np.zeros((3, len(self.marker_names), trial_nb_frames))
        markers_to_keep_names = []  # To keep track of the reordering of the marker names
        marker_to_keep_idx = 0
        for i_marker, marker_name in enumerate(model_markers_names):
            if marker_name in self.marker_names:
                markers_to_keep_names += [marker_name]
                markers_to_keep[:, marker_to_keep_idx, :] = model_markers[:, i_marker, :]
                marker_to_keep_idx += 1
        return markers_to_keep, markers_to_keep_names
