import opensim
import numpy as np

from pyorerun import OsimModel, PhaseRerun, OsimTimeSeries, DisplayModelOptions


def main():
    # building some time components
    nb_frames = 100
    nb_seconds = 1
    t_span = np.linspace(0, nb_seconds, nb_frames)

    osim_model = opensim.Model(r"D:\Documents\Programmation\osim_to_biomod\example\Models\Model_Pose2Sim.osim")
    display_options = DisplayModelOptions()
    display_options.mesh_path = "Geometry_cleaned"
    prr_model = OsimModel.from_osim_object(osim_model, options=display_options)

    # building some generalized coordinates
    q = np.zeros((prr_model.nb_q, nb_frames))
    q[10, :] = np.linspace(0, 0.2, nb_frames)
    # q[1, :] = np.linspace(0, 2, nb_frames)
    mot_file = r"F:\CIME_LOC\tmp_videos\20250422_162024_output_jsons\ik_mot.mot"
    mot_file = r"ik.mot"
    mot_time_series = OsimTimeSeries(mot_file, None)
    viz = PhaseRerun(mot_time_series.times)
    viz.add_animated_model(prr_model, mot_time_series.q )#_in_radian[:, :], display_q=False)
    # viz = PhaseRerun(t_span=t_span)
    # viz.add_animated_model(prr_model, q)
    viz.rerun("msk_model")


if __name__ == "__main__":
    main()
