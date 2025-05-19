from dataclasses import dataclass


@dataclass
class DisplayModelOptions:
    """
    Options to display the model
    """

    markers_color: tuple[int, int, int] = (0, 0, 255)
    markers_radius: float = 0.01
    show_marker_labels: bool = True

    centers_of_mass_color: tuple[int, int, int] = (0, 0, 0)
    centers_of_mass_radius: float = 0.01
    show_center_of_mass_labels: bool = True

    ligaments_color: tuple[int, int, int] = (255, 255, 0)
    ligaments_radius: float = 0.01
    show_ligament_labels: bool = True

    muscles_color: tuple[int, int, int] = (255, 0, 0)
    muscles_radius: float = 0.004
    show_muscle_labels: bool = True

    # NOTE : mesh_opacity doesnt exist in rerun yet
    # segment_frame_size: float = 0.1 not implemented yet
    mesh_color: tuple[int, int, int] = (255, 255, 255)
    transparent_mesh: bool = False
    show_gravity: bool = False

    soft_contacts_color: tuple[int, int, int] = (0, 255, 255)
    rigid_contacts_color: tuple[int, int, int] = (255, 0, 255)
    show_contact_labels: bool = True
