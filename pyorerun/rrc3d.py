# import ezc3d
# import numpy as np
# import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!
#
# from .markerset import MarkerSet
# from .rr_utils import display_markers
#
# COLOR = np.array([255, 255, 255])
#
#
# def rrc3d(cd3_file: ezc3d.c3d | str) -> None:
#     """
#     Display a c3d file in rerun.
#
#     Parameters
#     ----------
#     cd3_file: ezc3d.c3d | str
#         The c3d file to display.
#     """
#
#     # Load a c3d file
#     c3d_file = c3d_file_format(cd3_file)
#     positions = c3d_file["data"]["points"]
#
#     filename = cd3_file if isinstance(cd3_file, str) else "c3d file"
#
#     nb_frames = positions.shape[2]
#
#     frequency = c3d_file["header"]["points"]["frame_rate"]
#     first_frame = c3d_file["header"]["points"]["first_frame"]
#     unit = c3d_file["parameters"]["POINT"]["UNITS"]["value"]
#
#     labels = c3d_file["parameters"]["POINT"]["LABELS"]["value"]
#     labels = [label.encode("utf-8") for label in labels]
#
#     marker_set = MarkerSet(np.transpose(positions, axes=(2, 1, 0)), labels)
#     marker_set.set_color(COLOR)
#     marker_set.set_size(0.01)
#
#     positions = adjust_position_unit_to_meters(positions, unit)
#     t_span = calculate_time_span(first_frame, frequency, nb_frames)
#
#     rr.init(filename.split(".")[0], spawn=True)
#
#     for i in range(nb_frames):
#         rr.set_time_seconds("stable_time", t_span[i])
#         # put first frame in shape (n_mark, 3)
#         positions_f = positions[:3, :, i].T
#
#         display_markers("C3D", "markers", positions_f, marker_set.to_rerun(True))
#
#         for m in labels:
#             for j, axis in enumerate(["X", "Y", "Z"]):
#                 rr.log(
#                     f"markers_graphs/{m}/{axis}",
#                     rr.Scalar(
#                         positions_f[labels.index(m), j],
#                     ),
#                 )
#
#
# def c3d_file_format(cd3_file) -> ezc3d.c3d:
#     """Return the c3d file in the format of ezc3d.c3d if it is a string path."""
#     if isinstance(cd3_file, str):
#         return ezc3d.c3d(cd3_file)
#
#     return cd3_file
#
#
# def adjust_position_unit_to_meters(positions: np.ndarray, unit: str) -> np.ndarray:
#     """Adjust the positions to meters for displaying purposes."""
#     conversion_factors = {"mm": 1000, "cm": 100, "m": 1}
#     for u, factor in conversion_factors.items():
#         if u in unit:
#             positions /= factor
#             break
#     else:
#         raise ValueError("The unit of the c3d file is not in meters, mm or cm.")
#
#     return positions
#
#
# def calculate_time_span(first_frame: int, frequency: float, nb_frames: int) -> np.ndarray:
#     """
#     Calculate the time span based on the first frame, frequency, and number of frames.
#
#     Parameters
#     ----------
#     first_frame : int
#         The first frame of the data.
#     frequency : float
#         The frequency of the data.
#     nb_frames : int
#         The number of frames in the data.
#
#     Returns
#     -------
#     np.ndarray
#         The calculated time span.
#     """
#     initial_time = first_frame / frequency
#     t_span = np.linspace(initial_time, nb_frames / frequency, nb_frames)
#     return t_span
