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

    # NOTE : mesh_opacity doesnt exist in rerun yet
    # segment_frame_size: float = 0.1 not implemented yet
    mesh_color: tuple[int, int, int] = (255, 255, 255)
    show_gravity: bool = False
