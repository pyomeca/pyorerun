import numpy as np

from pyorerun import OsimTimeSeries
from pyorerun.abstract.abstract_class import ExperimentalData
from pyorerun.xp_phase import XpRerunPhase

mot_file = "../examples/osim/ik.mot"
osim_ts = OsimTimeSeries(mot_file)


class MockExperimentalData(ExperimentalData):
    def __init__(self, name):
        self.name = name
        self.initialized = False
        self.rerun_frame = None
        self.chunk_data = {name: [1, 2, 3]}

    def initialize(self):
        self.initialized = True

    def to_rerun(self, frame):
        self.rerun_frame = frame

    def to_chunk(self):
        return self.chunk_data

    def nb_components(self):
        return 1

    def nb_frames(self):
        return 3


def test_xp_phase_init():
    phase = XpRerunPhase("test_phase", 1)
    assert phase.name == "test_phase"
    assert phase.phase == 1
    assert phase.xp_data == []


def test_add_data():
    phase = XpRerunPhase("test_phase", 1)
    data = MockExperimentalData("test_data")
    phase.add_data(data)
    assert len(phase.xp_data) == 1
    assert phase.xp_data[0] == data


def test_initialize():
    phase = XpRerunPhase("test_phase", 1)
    data1 = MockExperimentalData("data1")
    data2 = MockExperimentalData("data2")
    phase.add_data(data1)
    phase.add_data(data2)

    phase.initialize()
    assert data1.initialized
    assert data2.initialized


def test_to_rerun():
    phase = XpRerunPhase("test_phase", 1)
    data1 = MockExperimentalData("data1")
    data2 = MockExperimentalData("data2")
    phase.add_data(data1)
    phase.add_data(data2)

    phase.to_rerun(5)
    assert data1.rerun_frame == 5
    assert data2.rerun_frame == 5


def test_to_chunk():
    phase = XpRerunPhase("test_phase", 1)
    data1 = MockExperimentalData("data1")
    data2 = MockExperimentalData("data2")
    phase.add_data(data1)
    phase.add_data(data2)

    chunk = phase.to_chunk()
    assert chunk == {"data1": [1, 2, 3], "data2": [1, 2, 3]}


def test_component_names():
    phase = XpRerunPhase("test_phase", 1)
    data1 = MockExperimentalData("data1")
    data2 = MockExperimentalData("data2")
    phase.add_data(data1)
    phase.add_data(data2)

    assert phase.component_names == ["data1", "data2"]


def test_osim_time_series():
    assert isinstance(osim_ts, OsimTimeSeries)
    assert osim_ts.coordinate_names == (
        "pelvis_tilt",
        "pelvis_list",
        "pelvis_rotation",
        "pelvis_tx",
        "pelvis_ty",
        "pelvis_tz",
        "hip_flexion_r",
        "hip_adduction_r",
        "hip_rotation_r",
        "knee_angle_r",
        "ankle_angle_r",
        "subtalar_angle_r",
        "mtp_angle_r",
        "hip_flexion_l",
        "hip_adduction_l",
        "hip_rotation_l",
        "knee_angle_l",
        "ankle_angle_l",
        "subtalar_angle_l",
        "mtp_angle_l",
        "lumbar_extension",
        "lumbar_bending",
        "lumbar_rotation",
        "arm_flex_r",
        "arm_add_r",
        "arm_rot_r",
        "elbow_flex_r",
        "pro_sup_r",
        "wrist_flex_r",
        "wrist_dev_r",
        "arm_flex_l",
        "arm_add_l",
        "arm_rot_l",
        "elbow_flex_l",
        "pro_sup_l",
        "wrist_flex_l",
        "wrist_dev_l",
    )
    assert osim_ts.q[0, 50] == np.float64(3.16086172)
    assert osim_ts.is_degree is True
