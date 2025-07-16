from pathlib import Path

import pytest

from pyorerun import OsimTimeSeries


@pytest.mark.skipif("opensim" not in globals(), reason="OpenSim is not installed.")
def test_osim_time_series():

    MOT_FILE = "../examples/osim/ik.mot"
    OSIM_TIME_SERIES = OsimTimeSeries(str(Path(__file__).parent) + "/" + MOT_FILE)

    assert isinstance(OSIM_TIME_SERIES, OsimTimeSeries)
    assert OSIM_TIME_SERIES.coordinate_names == (
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
    assert OSIM_TIME_SERIES.q[0, 50] == np.float64(3.16086172)
    assert OSIM_TIME_SERIES.is_degree is True
