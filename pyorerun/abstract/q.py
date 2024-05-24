import rerun as rr


class QProperties:
    """
    A class used to represent the properties of generalized coordinates q.


    Attributes
    ----------
    joint_names : list[str]
        a list of names for the markers
    width : float
        the width of the markers
    max_color : np.ndarray
        the color of the max value of the markers, in RGB format from 0 to 255, e.g. [255, 0, 0] for red
    min_color : np.ndarray
        the color of the min value of the markers, in RGB format from 0 to 255, e.g. [255, 0, 0] for red
    value_color : np.ndarray
        the color of the value of the markers, in RGB format from 0 to 255, e.g. [0, 255, 0] for green

    Methods
    -------
    nb_q():
        Returns the number of generalized coordinates q.

    """

    def __init__(
        self,
        joint_names: list[str, ...] | tuple[str, ...],
        ranges: tuple[tuple[float, float], ...] = None,
        width: float = None,
    ):
        """
        Constructs all the necessary attributes for the MarkerProperties object.

        Parameters
        ----------
            joint_names : list[str, ...] | tuple[str, ...]
                a list of names for the joints
            ranges : tuple[tuple[float, float], ...]
                the ranges of the q values, min and max
            width : float
                the width of the markers
        """
        self.joint_names = joint_names
        self.displayed_joint_names = [self.displayed_joint_name(joint_idx) for joint_idx in range(self.nb_q)]
        self.ranges = ranges
        self.width = 0.5 if width is None else width
        self.max_color = [255, 0, 0]
        self.min_color = [255, 0, 0]
        self.value_color = [0, 255, 0]

    @property
    def nb_q(self) -> int:
        """
        Returns the number of generalized coordinates q.

        Returns
        -------
        int
            The number of q.
        """
        return len(self.joint_names)

    def displayed_joint_name(self, joint_idx) -> str:
        return f"q{joint_idx} - {self.joint_names[joint_idx]}"

    def set_time_series(self, base_name: str):
        # Note: base_name is not used yet because of tree like structure of rerun, it was not a good idea to use it.
        for joint_idx in range(self.nb_q):
            joint_name = self.displayed_joint_names[joint_idx]
            rr.log(f"{joint_name}/min", rr.SeriesLine(color=self.min_color, name="min", width=self.width))
            rr.log(f"{joint_name}/max", rr.SeriesLine(color=self.max_color, name="max", width=self.width))
            rr.log(f"{joint_name}/value", rr.SeriesLine(color=self.value_color, name="q", width=self.width))
