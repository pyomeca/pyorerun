class PersistentMarkerOptions:
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

    def frames_to_keep(self, frame_idx: int) -> list[int]:
        """
        Give the list of frames to keep for a given current frame index.

        Examples
        --------
        - If nb_frames=5 and frame_idx=10, it will return [6, 7, 8, 9, 10]
        - If nb_frames=5 and frame_idx=3, it will return [0, 1, 2, 3]

        Parameters
        ----------
        frame_idx : int
            The current frame index.
        """
        n = self.nb_frames
        start = max(0, frame_idx - n + 1)
        return list(range(start, frame_idx + 1))

    def all_frames_to_keep(self, total_frames: int) -> list[list[int]]:
        """
        Give the list of frames to keep for all frames from 0 to total_frames-1.

        Parameters
        ----------
        total_frames : int
            The total number of frames in the simulation.
        """
        return [self.frames_to_keep(frame_idx) for frame_idx in range(total_frames)]
