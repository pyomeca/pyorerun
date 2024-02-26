import ezc3d
import numpy as np
import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!


def rrc3d(cd3_file: ezc3d.c3d | str) -> None:
    """
    Display a c3d file in rerun.

    Parameters
    ----------
    cd3_file: ezc3d.c3d | str
        The c3d file to display.
    """

    # Load a c3d file
    c3d_file = c3d_file_format(cd3_file)
    positions = c3d_file["data"]["points"]

    filename = cd3_file if isinstance(cd3_file, str) else "c3d file"

    nb_markers = positions.shape[1]
    nb_frames = positions.shape[2]

    frequency = c3d_file["header"]["points"]["frame_rate"]
    first_frame = c3d_file["header"]["points"]["first_frame"]
    initial_time = first_frame / frequency
    unit = c3d_file["parameters"]["POINT"]["UNITS"]["value"]

    labels = c3d_file["parameters"]["POINT"]["LABELS"]["value"]
    labels = [label.encode("utf-8") for label in labels]

    COLORS = np.ones((nb_markers, 3))

    if unit == "mm":
        positions /= 1000

    time = initial_time
    t_span = np.linspace(initial_time, nb_frames / frequency, nb_frames)

    rr.init(filename, spawn=True)

    for i in range(nb_frames):
        rr.set_time_seconds("stable_time", t_span[i])
        # put first frame in shape (n_mark, 3)
        positions_f = positions[:3, :, i].T

        rr.log(
            "my_markers",
            rr.Points3D(positions_f, colors=COLORS, radii=10, labels=labels),
        )

        for m in labels:
            for i, axis in enumerate(["X", "Y", "Z"]):
                rr.log(
                    f"markers_graphs/{m}/{axis}",
                    rr.Scalar(
                        positions_f[labels.index(m), i],
                    ),
                )


def c3d_file_format(cd3_file) -> ezc3d.c3d:
    if isinstance(cd3_file, str):
        return ezc3d.c3d(cd3_file)

    return cd3_file
