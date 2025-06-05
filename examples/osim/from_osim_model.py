import os
import opensim
import numpy as np

from pyorerun import OsimModel, PhaseRerun, OsimTimeSeries, DisplayModelOptions


def main():
    osim_model = opensim.Model(r"models\Rajagopal2015.osim")
    display_options = DisplayModelOptions()
    display_options.mesh_path = "D:\Documents\Programmation\pyorerun\examples\biorbd\models\Geometry_cleaned"
    prr_model = OsimModel.from_osim_object(osim_model, options=display_options)

    mot_file = r"ik.mot"
    mot_time_series = OsimTimeSeries(mot_file, osim_model)
    viz = PhaseRerun(mot_time_series.times)
    viz.add_animated_model(prr_model, mot_time_series.q_in_radian)
    # viz = PhaseRerun(t_span=t_span)
    # viz.add_animated_model(prr_model, q)
    viz.rerun("msk_model")


if __name__ == "__main__":
    main()
