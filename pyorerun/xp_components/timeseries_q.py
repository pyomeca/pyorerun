import numpy as np
import rerun as rr

from ..abstract.abstract_class import ExperimentalData
from ..abstract.q import QProperties


class TimeSeriesQ(ExperimentalData):
    def __init__(self, name, q: np.ndarray, properties: QProperties):
        self.name = name
        self.q = q
        self.properties = properties
        self.properties.set_time_series(base_name="dontknow")

    @property
    def nb_q(self):
        return self.q.shape[0]

    @property
    def q_names(self):
        return self.properties.joint_names

    @property
    def nb_frames(self):
        return self.q.shape[1]

    @property
    def nb_components(self):
        return 1

    def to_rerun(self, frame: int) -> None:
        if self.properties.ranges is None:
            for joint_idx in range(self.nb_q):
                name = f"{self.properties.displayed_joint_names[joint_idx]}"
                rr.log(
                    f"{name}/value",
                    rr.SeriesLine(color=self.properties.value_color, name="q", width=self.properties.width),
                )
                rr.log(f"{name}/value", rr.Scalar(self.q[joint_idx, frame]))
        else:
            for joint_idx in range(self.nb_q):
                name = f"{self.properties.displayed_joint_names[joint_idx]}"
                qmin, qmax = self.properties.ranges[joint_idx]
                #  todo: this log calls should be done once for all somewhere in the properties of Q
                rr.log(
                    f"{name}/min",
                    rr.SeriesLine(color=self.properties.min_color, name="min", width=self.properties.width),
                )
                rr.log(
                    f"{name}/max",
                    rr.SeriesLine(color=self.properties.max_color, name="max", width=self.properties.width),
                )
                rr.log(
                    f"{name}/value",
                    rr.SeriesLine(color=self.properties.value_color, name="q", width=self.properties.width),
                )
                self.to_serie_line(name=name, min=qmin, max=qmax, val=self.q[joint_idx, frame])

    @staticmethod
    def to_serie_line(name: str, min: float, max: float, val: float):
        rr.log(f"{name}/min", rr.Scalar(min))
        rr.log(f"{name}/max", rr.Scalar(max))
        rr.log(f"{name}/value", rr.Scalar(val))
