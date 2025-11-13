from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr

from .phase_rerun import PhaseRerun
from .pyomarkers import PyoMarkers
from .multi_frame_rate_phase_rerun import MultiFrameRatePhaseRerun


def rrtrc(
    trc_filename: str,
    marker_trajectories: bool = False,
    show_floor: bool = True,
    notebook: bool = False,
) -> None:
    """
    Display a c3d file in rerun.

    Parameters
    ----------
    trc_filename: str
        The path to the trc file.
    marker_trajectories: bool
        If True, show the marker trajectories.
    show_floor: bool
        If True, show the floor.
    notebook: bool
        If True, display the animation in the notebook.
    """

    # Load a c3d file
    pyomarkers = PyoMarkers.from_trc(trc_filename)
    units = pyomarkers.units
    pyomarkers = adjust_position_unit_to_meters(pyomarkers, pyomarkers.units)
    pyomarkers.show_labels = False

    t_span = pyomarkers.time
    filename = Path(trc_filename).name

    phase_reruns = []
    phase_rerun = PhaseRerun(t_span)
    phase_reruns.append(phase_rerun)
    phase_rerun.add_xp_markers(filename, pyomarkers)

    if show_floor:
        square_width = max_xy_coordinate_span_by_markers(pyomarkers)
        lowest_corner = 0
        phase_rerun.add_floor(square_width, height_offset=lowest_corner - 0.0005)

    multi_phase_rerun = MultiFrameRatePhaseRerun(phase_reruns)
    multi_phase_rerun.rerun(filename, notebook=notebook)

    if marker_trajectories:
        # # todo: find a better way to display curves but hacky way ok for now
        marker_names = phase_rerun.xp_data.xp_data[0].marker_names
        for m in marker_names:
            for j, axis in enumerate(["X", "Y", "Z"]):
                rr.send_columns(
                    f"markers_graphs/{m}/{axis}",
                    indexes=[rr.TimeColumn("stable_time", duration=t_span)],
                    columns=[
                        *rr.Scalars.columns(
                            scalars=phase_rerun.xp_data.xp_data[0].markers_numpy[j, marker_names.index(m), :]
                        )
                    ],
                )


def max_xy_coordinate_span_by_markers(pyomarkers: PyoMarkers) -> float:
    """Return the max span of the x and y coordinates of the markers."""
    min_pyomarkers = np.nanmin(np.nanmin(pyomarkers.to_numpy(), axis=2), axis=1)
    max_pyomarkers = np.nanmax(np.nanmax(pyomarkers.to_numpy(), axis=2), axis=1)
    x_absolute_max = np.nanmax(np.abs([min_pyomarkers[0], max_pyomarkers[0]]))
    y_absolute_max = np.nanmax(np.abs([min_pyomarkers[1], max_pyomarkers[1]]))

    return np.max([x_absolute_max, y_absolute_max])


def adjust_pyomarkers_unit_to_meters(pyomarkers: PyoMarkers, unit: str) -> PyoMarkers:
    """Adjust the positions to meters for displaying purposes."""
    pyomarkers = adjust_position_unit_to_meters(pyomarkers, unit)
    pyomarkers.attrs["units"] = "m"
    return pyomarkers


def adjust_position_unit_to_meters(array: Any, unit: str) -> Any:
    conversion_factors = {"mm": 1000, "cm": 100, "m": 1}
    for u, factor in conversion_factors.items():
        if u in unit:
            array /= factor
            break
    else:
        raise ValueError("The unit of the c3d file is not in meters, mm or cm.")
    return array
