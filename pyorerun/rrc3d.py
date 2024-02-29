from pathlib import Path

import ezc3d
import numpy as np
from pyomeca import Markers as PyoMarkers

from .phase_rerun import PhaseRerun

# from .rr_utils import display_markers

COLOR = np.array([255, 255, 255])


def rrc3d(c3d_file: str) -> None:
    """
    Display a c3d file in rerun.

    Parameters
    ----------
    cd3_file: str
        The c3d file to display.
    """

    # Load a c3d file
    pyomarkers = PyoMarkers.from_c3d(c3d_file)
    pyomarkers = adjust_position_unit_to_meters(pyomarkers, pyomarkers.units)
    t_span = pyomarkers.time.to_numpy()
    filename = Path(c3d_file).name

    phase_rerun = PhaseRerun(t_span)
    phase_rerun.add_xp_markers(filename, pyomarkers)
    phase_rerun.rerun(filename)

    #     for m in labels:
    #         for j, axis in enumerate(["X", "Y", "Z"]):
    #             rr.log(
    #                 f"markers_graphs/{m}/{axis}",
    #                 rr.Scalar(
    #                     positions_f[labels.index(m), j],
    #                 ),
    #             )


def c3d_file_format(cd3_file) -> ezc3d.c3d:
    """Return the c3d file in the format of ezc3d.c3d if it is a string path."""
    if isinstance(cd3_file, str):
        return ezc3d.c3d(cd3_file)

    return cd3_file


def adjust_position_unit_to_meters(pyomarkers: PyoMarkers, unit: str) -> PyoMarkers:
    """Adjust the positions to meters for displaying purposes."""
    conversion_factors = {"mm": 1000, "cm": 100, "m": 1}
    for u, factor in conversion_factors.items():
        if u in unit:
            pyomarkers /= factor
            break
    else:
        raise ValueError("The unit of the c3d file is not in meters, mm or cm.")

    pyomarkers.attrs["units"] = "m"
    return pyomarkers
