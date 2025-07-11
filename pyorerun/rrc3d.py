from pathlib import Path
from typing import Any

import ezc3d
import imageio
import numpy as np
import rerun as rr

from .multi_frame_rate_phase_rerun import MultiFrameRatePhaseRerun
from .phase_rerun import PhaseRerun
from .pyomarkers import PyoMarkers


def rrc3d(
    c3d_file: str,
    show_floor: bool = True,
    show_force_plates: bool = True,
    show_forces: bool = True,
    show_events: bool = True,
    show_marker_labels: bool = True,
    down_sampled_forces: bool = False,
    video: str | tuple[str, ...] = None,
    video_crop_mode: str = "from_c3d",
    marker_trajectories: bool = False,
    notebook: bool = False,
) -> None:
    """
    Display a c3d file in rerun.

    Parameters
    ----------
    cd3_file: str
        The c3d file to display.
    show_floor: bool
        If True, show the floor.
    show_force_plates: bool
        If True, show the force plates.
    show_forces: bool
        If True, show the forces.
    show_events: bool
        If True, show the events, as log entries.
    show_marker_labels: bool
        If True, show the marker labels.
    down_sampled_forces: bool
        If True, down sample the force data to align with the marker data.
        If False, the force data will be displayed at their original frame rate, It may get slower when loading the data.
    video: str or tuple
        If str, the path to the video to display.
        If tuple, the first element is the path to the video and the second element is the path to the time data of the video.
    video_crop_mode: str
        The mode to crop the video. If 'from_c3d', the video will be cropped to the same time span as the c3d file.
    marker_trajectories: bool
        If True, show the marker trajectories.
    notebook: bool
        If True, display the animation in the notebook.
    """

    # Load a c3d file
    pyomarkers = PyoMarkers.from_c3d(c3d_file, show_labels=show_marker_labels)
    units = pyomarkers.units
    pyomarkers = adjust_position_unit_to_meters(pyomarkers, pyomarkers.units)
    t_span = pyomarkers.time
    filename = Path(c3d_file).name

    phase_reruns = []
    phase_rerun = PhaseRerun(t_span)
    phase_reruns.append(phase_rerun)
    phase_rerun.add_xp_markers(filename, pyomarkers)

    lowest_corner = 0.0
    if show_force_plates:
        force_plates_corners = get_force_plates(c3d_file, units=units)
        lowest_corner = get_lowest_corner(c3d_file, units=units)

        for i, corners in enumerate(force_plates_corners):
            phase_rerun.add_force_plate(f"force_plate_{i}", corners["corners"])

    if show_forces:
        force_data = get_force_vector(c3d_file)
        if len(force_data) == 0:
            raise RuntimeError("No force data found in the c3d file. Set show_forces to False.")
        if down_sampled_forces:
            for i, force in enumerate(force_data):
                force["center_of_pressure"], force["force"] = down_sample_force(force, t_span, units)
                phase_rerun.add_force_data(
                    num=i,
                    force_origin=force["center_of_pressure"],
                    force_vector=force["force"],
                )
        else:
            phase_rerun_plateform = PhaseRerun(force_data[0]["time"])  # assuming the same time for all force data
            for i, force in enumerate(force_data):
                phase_rerun_plateform.add_force_data(
                    num=i,
                    force_origin=adjust_position_unit_to_meters(force["center_of_pressure"], unit=units),
                    force_vector=force["force"],
                )

            phase_reruns.append(phase_rerun_plateform)

    if show_floor:
        square_width = max_xy_coordinate_span_by_markers(pyomarkers)
        phase_rerun.add_floor(square_width, height_offset=lowest_corner - 0.0005)

    if video is not None:
        for i, vid in enumerate(video if isinstance(video, tuple) else [video]):
            vid_name = Path(vid).name
            vid, time = load_a_video(vid)
            vid, time = crop_video(
                vid,
                time,
                video_crop_mode,
                first_frame_in_sec=pyomarkers.first_frame / pyomarkers.rate,
                last_frame_in_sec=pyomarkers.last_frame / pyomarkers.rate,
            )

            phase_reruns.append(PhaseRerun(time))
            phase_reruns[-1].add_video(vid_name, np.array(vid, dtype=np.uint8))

    multi_phase_rerun = MultiFrameRatePhaseRerun(phase_reruns)
    multi_phase_rerun.rerun(filename, notebook=notebook)

    if show_events:
        try:
            set_event_as_log(c3d_file)
        except:
            raise NotImplementedError(
                "The events feature is still experimental and may not work properly. " "Set show_events=False."
            )

    if marker_trajectories:
        # # todo: find a better way to display curves but hacky way ok for now
        markers_names = phase_rerun.xp_data.xp_data[0].markers_names
        for m in markers_names:
            for j, axis in enumerate(["X", "Y", "Z"]):
                rr.send_columns(
                    f"markers_graphs/{m}/{axis}",
                    times=[rr.TimeSecondsColumn("stable_time", t_span)],
                    components=[
                        rr.components.ScalarBatch(
                            phase_rerun.xp_data.xp_data[0].markers_numpy[j, markers_names.index(m), :]
                        )
                    ],
                )


def set_event_as_log(c3d_file: str) -> None:
    c3d_file = c3d_file_format(c3d_file)
    times = c3d_file["parameters"]["EVENT"]["TIMES"]["value"][1, :]
    labels = c3d_file["parameters"]["EVENT"]["LABELS"]["value"]
    descriptions = c3d_file["parameters"]["EVENT"]["DESCRIPTIONS"]["value"]
    context = c3d_file["parameters"]["EVENT"]["CONTEXTS"]["value"]

    for i, (time, label, description, context) in enumerate(zip(times, labels, descriptions, context)):
        rr.set_time_seconds("stable_time", time)
        rr.log(
            f"events",
            rr.TextLog(
                f"{label}_{context} - {description}",
            ),
        )


def max_xy_coordinate_span_by_markers(pyomarkers: PyoMarkers) -> float:
    """Return the max span of the x and y coordinates of the markers."""
    min_pyomarkers = np.nanmin(np.nanmin(pyomarkers.to_numpy(), axis=2), axis=1)
    max_pyomarkers = np.nanmax(np.nanmax(pyomarkers.to_numpy(), axis=2), axis=1)
    x_absolute_max = np.nanmax(np.abs([min_pyomarkers[0], max_pyomarkers[0]]))
    y_absolute_max = np.nanmax(np.abs([min_pyomarkers[1], max_pyomarkers[1]]))

    return np.max([x_absolute_max, y_absolute_max])


def c3d_file_format(cd3_file) -> ezc3d.c3d:
    """Return the c3d file in the format of ezc3d.c3d if it is a string path."""
    if isinstance(cd3_file, str):
        return ezc3d.c3d(cd3_file, extract_forceplat_data=True)

    return cd3_file


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


def get_force_plates(c3d_file, units) -> list[dict[str, np.ndarray]]:
    c3d_file = c3d_file_format(c3d_file)
    force_plates = []
    nb_force_plates = c3d_file["parameters"]["FORCE_PLATFORM"]["USED"]["value"][0]
    for i in range(nb_force_plates):
        force_plates.append(
            {
                "corners": adjust_position_unit_to_meters(
                    c3d_file["parameters"]["FORCE_PLATFORM"]["CORNERS"]["value"][:, :, i],
                    unit=units,
                ),
            }
        )

    return force_plates


def get_lowest_corner(c3d_file, units) -> float:
    c3d_file = c3d_file_format(c3d_file)
    corners = c3d_file["parameters"]["FORCE_PLATFORM"]["CORNERS"]["value"][2, :, :]

    if corners.shape[1] == 0:
        return 0

    return np.min(
        adjust_position_unit_to_meters(
            c3d_file["parameters"]["FORCE_PLATFORM"]["CORNERS"]["value"][2, :, :],
            unit=units,
        )
    )


def get_force_vector(c3d_file) -> list[dict[str, np.ndarray]]:
    c3d_file = c3d_file_format(c3d_file)
    plateforms = c3d_file["data"]["platform"]

    frame_rate = c3d_file["header"]["analogs"]["frame_rate"]
    first_frame = c3d_file["header"]["analogs"]["first_frame"]
    last_frame = c3d_file["header"]["analogs"]["last_frame"]
    analog_time = (
        np.linspace(0, last_frame - first_frame, last_frame - first_frame + 1)
        / frame_rate
        # + 1 / frame_rate * first_frame
    )

    plateforms_dict = []
    for plateform in plateforms:
        plateform_dict = {key: plateform[key] for key in plateform.keys()}
        plateform_dict["time"] = analog_time
        plateforms_dict.append(plateform_dict)

    return plateforms_dict


def down_sample_force(plateform, t_span, units) -> tuple[np.ndarray, np.ndarray]:
    ratio = plateform["force"].shape[1] / t_span.shape[0]
    ratio_is_a_integer = ratio.is_integer()

    if not ratio_is_a_integer:
        raise NotImplementedError(
            "Set down_sampled_forces to False."
            "The ratio between the force data and the marker data is not an integer."
            "Interpolation is not implemented yet."
        )

    down_sampled_slice = slice(0, plateform["center_of_pressure"].shape[1], int(ratio))
    down_sampled_center_of_pressure = adjust_position_unit_to_meters(
        plateform["center_of_pressure"][:, down_sampled_slice], unit=units
    )

    return down_sampled_center_of_pressure, plateform["force"][:, down_sampled_slice]


def load_a_video(video_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a video from a path."""
    # Load the video
    video_reader = imageio.get_reader(video_path, "ffmpeg")

    # Extract frames and convert to numpy ndarray
    frames = []
    for frame in video_reader:
        frames.append(frame)

    frame_rate = video_reader.get_meta_data()["fps"]
    time = np.linspace(0, len(frames) / frame_rate, len(frames))

    return np.array(frames, dtype=np.uint16), time


def crop_video(
    video: np.ndarray, time, video_crop_mode: str, first_frame_in_sec: float, last_frame_in_sec: float
) -> tuple[np.ndarray, np.ndarray]:

    if video_crop_mode == "from_c3d":
        closest_time = lambda t: np.argmin(np.abs(t - time))
        first_frame = closest_time(first_frame_in_sec)
        last_frame = closest_time(last_frame_in_sec)
        time = time[first_frame : last_frame + 1] - first_frame_in_sec
        video = video[first_frame : last_frame + 1, :, :, :]

        return video, time

    elif video_crop_mode is not None:
        raise NotImplementedError(
            f"video_crop_mode={video_crop_mode} is not implemented yet."
            "Please use video_crop_mode='from_c3d' or None if already cropped."
        )

    return video, time
