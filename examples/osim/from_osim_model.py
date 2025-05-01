import opensim
import numpy as np

from pyorerun import OsimModel, PhaseRerun, OsimTimeSeries


def main():
    # building some time components
    nb_frames = 100
    nb_seconds = 1
    t_span = np.linspace(0, nb_seconds, nb_frames)

    biorbd_model = opensim.Model(r"D:\Documents\Programmation\osim_to_biomod\example\Models\Model_Pose2Sim.osim")
    prr_model = OsimModel.from_osim_object(biorbd_model)

    # building some generalized coordinates
    q = np.ones((biorbd_model.getNumCoordinates(), nb_frames))
    mot_file = r"F:\CIME_LOC\tmp_videos\20250422_162024_output_jsons\ik_mot.mot"
    mot_time_series = OsimTimeSeries(mot_file, biorbd_model)
    prr_model.set_xp_coordinate_names(mot_time_series.coordinate_names)


    viz = PhaseRerun(mot_time_series.times[:50])
    viz.add_animated_model(prr_model, mot_time_series.q_in_radian[:, :50])
    viz.rerun("msk_model")


if __name__ == "__main__":
    main()
