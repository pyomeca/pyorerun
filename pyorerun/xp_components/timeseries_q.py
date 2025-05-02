import numpy as np
import rerun as rr
try:
    import opensim as osim
except ImportError:
    pass
from ..abstract.abstract_class import ExperimentalData
from ..abstract.q import QProperties


class OsimTimeSeries:
    """
    Convert a .mot file to a time series of generalized coordinates q.
    The .mot file is a file format used in OpenSim to store motion capture data.
    """
    def __init__(self, mot_file : str, osim_model : any = None):
        """
        Parameters
        ----------
        mot_file : str
            The path to the .mot file.
        """
        self.mot_file = mot_file
        self.osim_model = None
        self.initialize_file()
        self.set_opensim_model(osim_model)
    
    @property
    def is_degree(self):
        """
        Returns True if the .mot file is in degrees, False otherwise.
        """
        return self.motion_data.getTableMetaDataAsString('inDegrees') == 'yes'
    
    @property
    def times(self):
        """
        Returns the time series of the .mot file.
        """
        return np.array(self.motion_data.getIndependentColumn())
    
    @property
    def coordinate_names(self):
        """
        Returns the names of the coordinates in the .mot file.
        """
        return self.motion_data.getColumnLabels()
    
    def initialize_file(self):
        self.motion_data = osim.TimeSeriesTable(self.mot_file)

    @property
    def q(self):
        """
        Returns the time series of the generalized coordinates q.
        """
        return self.motion_data.getMatrix().to_numpy().T
    
    @property
    def q_in_degree(self):
        """
        Convert the .mot file to radians.
        """
        if self.is_degree:
            return self.q
        if self.osim_model is None:
            raise ValueError("The original .mot file is not in degrees. Please set the osim_model before calling this method.")
        q_tmp = self.q.copy()
        for q_idx, q in enumerate(q_tmp):
            q_tmp[q_idx] = np.rad2deg(q) if self.motion_types[q_idx] == 1 else q
        return self.q    
    
    @property
    def q_in_radian(self):
        """
        Convert the .mot file to radians.
        """
        if not self.is_degree:
            return self.q
        if self.osim_model is None:
            raise ValueError("The original .mot file is in degrees. Please set the osim_model before calling this method.")
        q_tmp = self.q.copy()
        for q_idx, q in enumerate(q_tmp):
            q_tmp[q_idx] = np.deg2rad(q) if self.motion_types[q_idx] != 2 else q
        return q_tmp
    
    def set_opensim_model(self, osim_model: any):
        """
        Set the OpenSim model to be used for the conversion.
        """
        self.osim_model = osim_model if not isinstance(osim_model, str) else osim.Model(osim_model)
        coordinates_ordered = [self.osim_model.getCoordinateSet().get(coordinate) for coordinate in self.coordinate_names]
        self.motion_types = [coordinate.getMotionType() for coordinate in coordinates_ordered]

        

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

    def initialize(self):
        pass

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

    def to_chunk(self):
        pass