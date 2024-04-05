from dataclasses import dataclass


@dataclass
class DisplayModelOptions:
    """
    Options to display the model
    """

    markers_color: tuple[int, int, int] = (0, 0, 255)
    markers_radius: float = 0.01

    ligaments_color: tuple[int, int, int] = (255, 255, 0)
    ligaments_radius: float = 0.01

    muscles_color: tuple[int, int, int] = (255, 0, 0)
    muscles_radius: float = 0.004

    # NOTE : mesh_opacity doesn't exist in rerun yet
    # segment_frame_size: float = 0.1 not implemented yet
    mesh_color: tuple[int, int, int] = (255, 255, 255)

    background_color: tuple[int, int, int] = (15, 15, 15)

    # Flags to display the different components
    show_markers: bool = True
    show_marker_labels: bool = True
    show_mesh: bool = True
    show_global_ref_frame: bool = True
    show_local_ref_frame: bool = True
