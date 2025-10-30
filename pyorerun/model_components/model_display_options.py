from dataclasses import dataclass
from ..xp_components import PersistentMarkerOptions


@dataclass
class DisplayModelOptions:
    """
    Options to display the model
    """

    _markers_color: tuple[int, int, int] = (0, 0, 255)
    _markers_radius: float = 0.01
    _show_marker_labels: bool = False

    _centers_of_mass_color: tuple[int, int, int] = (0, 0, 0)
    _centers_of_mass_radius: float = 0.01
    _show_center_of_mass_labels: bool = False

    _ligaments_color: tuple[int, int, int] = (255, 255, 0)
    _ligaments_radius: float = 0.01
    _show_ligament_labels: bool = False

    _muscles_color: tuple[int, int, int] = (255, 0, 0)
    _muscles_radius: float = 0.004
    _show_muscle_labels: bool = False

    # NOTE : mesh_opacity doesnt exist in rerun yet
    # segment_frame_size: float = 0.1 not implemented yet
    _mesh_color: tuple[int, int, int] = (255, 255, 255)
    _transparent_mesh: bool = False
    _show_gravity: bool = False

    _soft_contacts_color: tuple[int, int, int] = (0, 255, 255)
    _rigid_contacts_color: tuple[int, int, int] = (255, 0, 255)
    _show_contact_labels: bool = False

    _mesh_path: str = ""

    _persistent_markers: PersistentMarkerOptions = None

    @property
    def markers_color(self) -> tuple[int, int, int]:
        return self._markers_color

    @markers_color.setter
    def markers_color(self, value: tuple[int, int, int]):
        if not isinstance(value, tuple) or len(value) != 3:
            raise ValueError("markers_color must be a tuple of three integers.")
        self.markers_color = value

    @property
    def markers_radius(self) -> float:
        return self._markers_radius

    @markers_radius.setter
    def markers_radius(self, value: float):
        if not isinstance(value, (int, float)):
            raise ValueError("markers_radius must be a number.")
        self._markers_radius = value

    @property
    def show_marker_labels(self) -> bool:
        return self._show_marker_labels

    @show_marker_labels.setter
    def show_marker_labels(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("show_marker_labels must be a boolean.")
        self._show_marker_labels = value

    @property
    def centers_of_mass_color(self) -> tuple[int, int, int]:
        return self._centers_of_mass_color

    @centers_of_mass_color.setter
    def centers_of_mass_color(self, value: tuple[int, int, int]):
        if not isinstance(value, tuple) or len(value) != 3:
            raise ValueError("centers_of_mass_color must be a tuple of three integers.")
        self._centers_of_mass_color = value

    @property
    def centers_of_mass_radius(self) -> float:
        return self._centers_of_mass_radius

    @centers_of_mass_radius.setter
    def centers_of_mass_radius(self, value: float):
        if not isinstance(value, (int, float)):
            raise ValueError("centers_of_mass_radius must be a number.")
        self._centers_of_mass_radius = value

    @property
    def show_center_of_mass_labels(self) -> bool:
        return self._show_center_of_mass_labels

    @show_center_of_mass_labels.setter
    def show_center_of_mass_labels(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("show_center_of_mass_labels must be a boolean.")
        self._show_center_of_mass_labels = value

    @property
    def ligaments_color(self) -> tuple[int, int, int]:
        return self._ligaments_color

    @ligaments_color.setter
    def ligaments_color(self, value: tuple[int, int, int]):
        if not isinstance(value, tuple) or len(value) != 3:
            raise ValueError("ligaments_color must be a tuple of three integers.")
        self._ligaments_color = value

    @property
    def ligaments_radius(self) -> float:
        return self._ligaments_radius

    @ligaments_radius.setter
    def ligaments_radius(self, value: float):
        if not isinstance(value, (int, float)):
            raise ValueError("ligaments_radius must be a number.")
        self._ligaments_radius = value

    @property
    def show_ligament_labels(self) -> bool:
        return self._show_ligament_labels

    @show_ligament_labels.setter
    def show_ligament_labels(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("show_ligament_labels must be a boolean.")
        self._show_ligament_labels = value

    @property
    def muscles_color(self) -> tuple[int, int, int]:
        return self._muscles_color

    @muscles_color.setter
    def muscles_color(self, value: tuple[int, int, int]):
        if not isinstance(value, tuple) or len(value) != 3:
            raise ValueError("muscles_color must be a tuple of three integers.")
        self._muscles_color = value

    @property
    def muscles_radius(self) -> float:
        return self._muscles_radius

    @muscles_radius.setter
    def muscles_radius(self, value: float):
        if not isinstance(value, (int, float)):
            raise ValueError("muscles_radius must be a number.")
        self._muscles_radius = value

    @property
    def show_muscle_labels(self) -> bool:
        return self._show_muscle_labels

    @show_muscle_labels.setter
    def show_muscle_labels(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("show_muscle_labels must be a boolean.")
        self._show_muscle_labels = value

    @property
    def mesh_color(self) -> tuple[int, int, int]:
        return self._mesh_color

    @mesh_color.setter
    def mesh_color(self, value: tuple[int, int, int]):
        if not isinstance(value, tuple) or len(value) != 3:
            raise ValueError("mesh_color must be a tuple of three integers.")
        self._mesh_color = value

    @property
    def transparent_mesh(self) -> bool:
        return self._transparent_mesh

    @transparent_mesh.setter
    def transparent_mesh(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("transparent_mesh must be a boolean.")
        self._transparent_mesh = value

    @property
    def show_gravity(self) -> bool:
        return self._show_gravity

    @show_gravity.setter
    def show_gravity(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("show_gravity must be a boolean.")
        self._show_gravity = value

    @property
    def soft_contacts_color(self) -> tuple[int, int, int]:
        return self._soft_contacts_color

    @soft_contacts_color.setter
    def soft_contacts_color(self, value: tuple[int, int, int]):
        if not isinstance(value, tuple) or len(value) != 3:
            raise ValueError("soft_contacts_color must be a tuple of three integers.")
        self._soft_contacts_color = value

    @property
    def rigid_contacts_color(self) -> tuple[int, int, int]:
        return self._rigid_contacts_color

    @rigid_contacts_color.setter
    def rigid_contacts_color(self, value: tuple[int, int, int]):
        if not isinstance(value, tuple) or len(value) != 3:
            raise ValueError("rigid_contacts_color must be a tuple of three integers.")
        self._rigid_contacts_color = value

    @property
    def show_contact_labels(self) -> bool:
        return self._show_contact_labels

    @show_contact_labels.setter
    def show_contact_labels(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("show_contact_labels must be a boolean.")
        self._show_contact_labels = value

    @property
    def mesh_path(self) -> str:
        return self._mesh_path

    @mesh_path.setter
    def mesh_path(self, value: str):
        if not isinstance(value, str):
            raise ValueError("mesh_path must be a string.")
        self._mesh_path = value

    @property
    def persistent_markers(self) -> PersistentMarkerOptions:
        return self._persistent_markers

    @persistent_markers.setter
    def persistent_markers(self, value: PersistentMarkerOptions):
        if not isinstance(value, PersistentMarkerOptions):
            raise ValueError("persistent_markers must be a PersistentMarkerOptions object.")
        self._persistent_markers = value

    def set_all_labels(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("Value must be a boolean.")
        self._show_marker_labels = value
        self._show_center_of_mass_labels = value
        self._show_ligament_labels = value
        self._show_muscle_labels = value
        self._show_contact_labels = value
