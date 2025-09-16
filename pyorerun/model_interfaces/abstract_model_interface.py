from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np

from ..model_components.model_display_options import DisplayModelOptions


class AbstractSegment(ABC):
    """
    An abstract interface for a model segment (or body).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the segment."""
        pass

    @property
    @abstractmethod
    def id(self) -> int:
        """The index of the segment."""
        pass

    @property
    @abstractmethod
    def has_mesh(self) -> bool:
        """Whether the segment has a visual mesh."""
        pass

    @property
    @abstractmethod
    def has_meshlines(self) -> bool:
        """Whether the segment has mesh lines (for biorbd compatibility)."""
        pass

    @property
    @abstractmethod
    def mesh_path(self) -> list[str]:
        """The file path(s) to the segment's mesh(es)."""
        pass

    @property
    @abstractmethod
    def mesh_scale_factor(self) -> list[np.ndarray]:
        """The scale factor(s) for the segment's mesh(es)."""
        pass

    @property
    @abstractmethod
    def mass(self) -> float:
        """The mass of the segment."""
        pass


class AbstractModelNoMesh(ABC):
    """
    An abstract interface for a biomechanical model.
    This class defines the common methods and properties for accessing model data
    without requiring a visual mesh.
    """

    def __init__(self, path: str, options: DisplayModelOptions = None):
        self.path = path
        self.options = options if options is not None else DisplayModelOptions()

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the model."""
        pass

    @property
    @abstractmethod
    def nb_markers(self) -> int:
        """The number of markers in the model."""
        pass

    @property
    @abstractmethod
    def marker_names(self) -> Tuple[str, ...]:
        """The names of the markers."""
        pass

    @abstractmethod
    def markers(self, q: np.ndarray) -> np.ndarray:
        """Get the global positions of markers for a given joint configuration q."""
        pass

    @abstractmethod
    def persistent_markers(self, q: np.ndarray, frame_range: range) -> np.ndarray:
        """Get the global positions of markers for a given time series of joint configuration q."""
        pass

    @property
    @abstractmethod
    def nb_segments(self) -> int:
        """The total number of segments in the model."""
        pass

    @property
    @abstractmethod
    def segment_names(self) -> Tuple[str, ...]:
        """The names of all segments."""
        pass

    @property
    @abstractmethod
    def segments(self) -> Tuple[AbstractSegment, ...]:
        """A tuple of all segment objects."""
        pass

    @property
    @abstractmethod
    def segments_with_mass(self) -> Tuple[AbstractSegment, ...]:
        """A tuple of segments with a mass greater than a minimal threshold."""
        pass

    @property
    @abstractmethod
    def segment_names_with_mass(self) -> Tuple[str, ...]:
        """The names of segments with a mass greater than a minimal threshold."""
        pass

    @abstractmethod
    def segment_homogeneous_matrices_in_global(self, q: np.ndarray, segment_index: int) -> np.ndarray:
        """Get the 4x4 homogeneous transformation matrix of a segment in the global frame."""
        pass

    @abstractmethod
    def centers_of_mass(self, q: np.ndarray) -> np.ndarray:
        """Get the global positions of the centers of mass for a given joint configuration q."""
        pass

    @property
    @abstractmethod
    def nb_ligaments(self) -> int:
        """The number of ligaments in the model."""
        pass

    @property
    @abstractmethod
    def ligament_names(self) -> Tuple[str, ...]:
        """The names of the ligaments."""
        pass

    @abstractmethod
    def ligament_strips(self, q: np.ndarray) -> List[List[np.ndarray]]:
        """Get the global positions of the ligament paths for a given joint configuration q."""
        pass

    @property
    @abstractmethod
    def nb_muscles(self) -> int:
        """The number of muscles in the model."""
        pass

    @property
    @abstractmethod
    def muscle_names(self) -> Tuple[str, ...]:
        """The names of the muscles."""
        pass

    @abstractmethod
    def muscle_strips(self, q: np.ndarray) -> List[List[np.ndarray]]:
        """Get the global positions of the muscle paths for a given joint configuration q."""
        pass

    @property
    @abstractmethod
    def nb_q(self) -> int:
        """The number of generalized coordinates (Q)."""
        pass

    @property
    @abstractmethod
    def dof_names(self) -> Tuple[str, ...]:
        """The names of the degrees of freedom."""
        pass

    @property
    @abstractmethod
    def q_ranges(self) -> Tuple[Tuple[float, float], ...]:
        """The ranges (min, max) for each generalized coordinate."""
        pass

    @property
    @abstractmethod
    def gravity(self) -> np.ndarray:
        """The gravity vector."""
        pass

    @property
    @abstractmethod
    def has_soft_contacts(self) -> bool:
        """Whether the model has soft contacts."""
        pass

    @property
    @abstractmethod
    def has_rigid_contacts(self) -> bool:
        """Whether the model has rigid contacts."""
        pass

    @abstractmethod
    def soft_contacts(self, q: np.ndarray) -> np.ndarray:
        """Get the positions of soft contacts."""
        pass

    @abstractmethod
    def rigid_contacts(self, q: np.ndarray) -> np.ndarray:
        """Get the positions of rigid contacts."""
        pass

    @property
    @abstractmethod
    def soft_contacts_names(self) -> tuple[str, ...]:
        """The names of the soft contacts."""
        pass

    @property
    @abstractmethod
    def rigid_contacts_names(self) -> tuple[str, ...]:
        """The names of the rigid contacts."""
        pass

    @property
    @abstractmethod
    def soft_contact_radii(self) -> tuple[float, ...]:
        """The radii of the soft contacts."""
        pass


class AbstractModel(AbstractModelNoMesh):
    """
    An abstract interface for a biomechanical model that has visual meshes.
    It extends the base model interface with methods to handle mesh data.
    """

    @property
    @abstractmethod
    def has_mesh(self) -> bool:
        """Whether any segment in the model has a visual mesh."""
        pass

    @property
    @abstractmethod
    def has_meshlines(self) -> bool:
        """Whether any segment in the model has mesh lines (biorbd)."""
        pass

    @property
    @abstractmethod
    def meshlines(self) -> list[np.ndarray]:
        """Get the vertices for mesh lines."""
        pass

    @abstractmethod
    def mesh_homogenous_matrices_in_global(self, q: np.ndarray, segment_index: int, **kwargs) -> np.ndarray:
        """Get the 4x4 homogeneous transformation matrix of a mesh in the global frame."""
        pass
