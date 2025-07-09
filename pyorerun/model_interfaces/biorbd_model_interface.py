from functools import cached_property

import biorbd
import numpy as np
from biorbd import GeneralizedCoordinates

# Import the abstract classes
from .abstract_model_interface import AbstractModel, AbstractModelNoMesh, AbstractSegment

MINIMAL_SEGMENT_MASS = 1e-08


class BiorbdSegment(AbstractSegment):  # Inherits from AbstractSegment
    """
    An interface to simplify the access to a segment of a biorbd model
    """

    def __init__(self, segment, index):
        self.segment = segment
        self._index: int = index

    @cached_property
    def name(self) -> str:
        return self.segment.name().to_string()

    @cached_property
    def id(self) -> int:
        return self._index

    @cached_property
    def has_mesh(self) -> bool:
        has_mesh = self.segment.characteristics().mesh().hasMesh()
        if has_mesh:
            return not self.mesh_path[0].endswith("/")  # Avoid empty mesh path
        return has_mesh

    @cached_property
    def has_meshlines(self) -> bool:
        has_mesh = self.segment.characteristics().mesh().hasMesh()
        if has_mesh:
            return self.mesh_path[0].endswith("/")  # Avoid empty mesh path
        return has_mesh

    @cached_property
    def mesh_path(self) -> list[str]:
        return [self.segment.characteristics().mesh().path().absolutePath().to_string()]

    @cached_property
    def mesh_scale_factor(self) -> list[np.ndarray]:
        """
        return: numpy array (3,) of the scale factor of the mesh
        """
        return [self.segment.characteristics().mesh().getScale().to_array()]

    @cached_property
    def mass(self) -> float:
        return self.segment.characteristics().mass()


class BiorbdModelNoMesh(AbstractModelNoMesh):  # Inherits from AbstractModelNoMesh
    """
    A class to handle a biorbd model and its transformations
    """

    def __init__(self, path: str, options=None):
        super().__init__(path, options)
        self.model = biorbd.Model(path)

    @classmethod
    def from_biorbd_object(cls, model: biorbd.Model, options=None):
        return cls(model.path().absolutePath().to_string(), options)

    @cached_property
    def name(self):
        return self.path.split("/")[-1].split(".")[0]

    @cached_property
    def marker_names(self) -> tuple[str, ...]:
        return tuple([s.to_string() for s in self.model.markerNames()])

    @cached_property
    def nb_markers(self) -> int:
        return self.model.nbMarkers()

    @cached_property
    def segment_names(self) -> tuple[str, ...]:
        return tuple([s.name().to_string() for s in self.model.segments()])

    @cached_property
    def nb_segments(self) -> int:
        return self.model.nbSegment()

    @cached_property
    def segments(self) -> tuple[BiorbdSegment, ...]:
        return tuple(BiorbdSegment(s, i) for i, s in enumerate(self.model.segments()))

    @cached_property
    def segments_with_mass(self) -> tuple[BiorbdSegment, ...]:
        return tuple([s for s in self.segments if s.mass > MINIMAL_SEGMENT_MASS])

    @cached_property
    def segment_names_with_mass(self) -> tuple[str, ...]:
        return tuple([s.name for s in self.segments_with_mass])

    def segment_homogeneous_matrices_in_global(self, q: np.ndarray, segment_index: int) -> np.ndarray:
        """
        Returns a biorbd object containing the roto-translation matrix of the segment in the global reference frame.
        This is useful if you want to interact with biorbd directly later on.
        """
        if np.sum(np.isnan(q)) != 0:
            # If q contains NaN, return an identity matrix as biorbd will throw an error otherwise
            rt_matrix = np.identity(4)
        else:
            rt_matrix = self.model.globalJCS(GeneralizedCoordinates(q), segment_index).to_array()
        return rt_matrix

    def markers(self, q: np.ndarray) -> np.ndarray:
        """
        Returns a [N_markers x 3] array containing the position of each marker in the global reference frame
        """
        return np.array([self.model.markers(GeneralizedCoordinates(q))[i].to_array() for i in range(self.nb_markers)])

    def centers_of_mass(self, q: np.ndarray) -> np.ndarray:
        """
        Returns the position of the centers of mass in the global reference frame
        """
        all_com = np.array([com.to_array() for com in self.model.CoMbySegment(GeneralizedCoordinates(q))])
        all_com_with_mass = np.zeros((len(self.segments_with_mass), 3))
        for i, segment in enumerate(self.segments_with_mass):
            all_com_with_mass[i, :] = all_com[segment.id, :]

        return all_com_with_mass

    @cached_property
    def nb_ligaments(self) -> int:
        """
        Returns the number of ligaments
        """
        return self.model.nbLigaments()

    @cached_property
    def ligament_names(self) -> tuple[str, ...]:
        """
        Returns the names of the ligaments
        """
        return tuple([s.to_string() for s in self.model.ligamentNames()])

    def ligament_strips(self, q: np.ndarray) -> list[list[np.ndarray]]:
        """
        Returns the position of the ligaments in the global reference frame
        """
        ligaments = []
        self.model.updateLigaments(q, True)
        for ligament_idx in range(self.nb_ligaments):
            ligament = self.model.ligament(ligament_idx)
            ligament_strip = []
            for pts in ligament.position().pointsInGlobal():
                ligament_strip.append(pts.to_array().tolist())
            ligaments.append(ligament_strip)
        return ligaments

    @cached_property
    def nb_muscles(self) -> int:
        """
        Returns the number of ligaments
        """
        return self.model.nbMuscles()

    @cached_property
    def muscle_names(self) -> tuple[str, ...]:
        """
        Returns the names of the ligaments
        """
        return tuple([s.to_string() for s in self.model.muscleNames()])

    def muscle_strips(self, q: np.ndarray) -> list[list[np.ndarray]]:
        """
        Returns the position of the ligaments in the global reference frame
        """
        muscles = []
        self.model.updateMuscles(q, True)
        for idx in range(self.nb_muscles):
            muscle = self.model.muscle(idx)
            muscle_strip = []
            for pts in muscle.position().pointsInGlobal():
                muscle_strip.append(pts.to_array().tolist())
            muscles.append(muscle_strip)
        return muscles

    @cached_property
    def nb_q(self) -> int:
        return self.model.nbQ()

    @cached_property
    def dof_names(self) -> tuple[str, ...]:
        return tuple(s.to_string() for s in self.model.nameDof())

    @cached_property
    def q_ranges(self) -> tuple[tuple[float, float], ...]:
        q_ranges = [q_range for segment in self.model.segments() for q_range in segment.QRanges()]
        return tuple((q_range.min(), q_range.max()) for q_range in q_ranges)

    @cached_property
    def gravity(self) -> np.ndarray:
        return self.model.getGravity().to_array()

    @cached_property
    def has_mesh(self) -> bool:
        return any([s.has_mesh for s in self.segments])

    @cached_property
    def has_meshlines(self) -> bool:
        return any([s.has_meshlines for s in self.segments])

    @cached_property
    def has_soft_contacts(self) -> bool:
        return self.model.nbSoftContacts() > 0

    @property
    def has_rigid_contacts(self) -> bool:
        return self.model.nbRigidContacts() > 0

    def soft_contacts(self, q: np.ndarray) -> np.ndarray:
        """
        Returns the position of the soft contacts spheres in the global reference frame
        """
        soft_contacts = self.model.softContacts(q, True)
        return np.array([soft_contacts[i].to_array() for i in range(self.model.nbSoftContacts())])

    def rigid_contacts(self, q: np.ndarray) -> np.ndarray:
        """
        Returns the position of the rigid contacts in the global reference frame
        """
        rigid_contacts = self.model.rigidContacts(q, True)
        return np.array([rigid_contacts[i].to_array() for i in range(self.model.nbRigidContacts())])

    @cached_property
    def soft_contacts_names(self) -> tuple[str, ...]:
        """
        Returns the names of the soft contacts
        """
        return tuple([s.to_string() for s in self.model.softContactNames()])

    @cached_property
    def rigid_contacts_names(self) -> tuple[str, ...]:
        """
        Returns the names of the soft contacts
        """
        return tuple([f"contact_{s}" for s in range(self.model.nbRigidContacts())])

    @cached_property
    def soft_contact_radii(self) -> tuple[float, ...]:
        """
        Returns the radii of the soft contacts
        """
        radii = []
        for i in range(self.model.nbSoftContacts()):
            sc = self.model.softContact(i)
            scs = biorbd.SoftContactSphere(sc)
            radii.append(scs.radius())

        return tuple(radii)


class BiorbdModel(BiorbdModelNoMesh, AbstractModel):  # Inherits from BiorbdModelNoMesh and AbstractModel
    """
    This class extends the BiorbdModelNoMesh class and overrides the segments property.
    It filters the segments to only include those that have a mesh.
    """

    def __init__(self, path, options=None):
        super().__init__(path, options)

    @property
    def segments(self) -> tuple[BiorbdSegment, ...]:
        return tuple([s for s in super().segments if s.has_mesh or s.has_meshlines])

    @cached_property
    def meshlines(self) -> list[np.ndarray]:

        meshes = []
        for segment in self.segments:
            segment_mesh = segment.segment.characteristics().mesh()
            meshes += [np.array([segment_mesh.point(i).to_array() for i in range(segment_mesh.nbVertex())])]

        return meshes

    def mesh_homogenous_matrices_in_global(self, q: np.ndarray, segment_index: int, **kwargs) -> np.ndarray:
        """
        Returns a list of homogeneous matrices of the mesh in the global reference frame
        """
        if np.sum(np.isnan(q)) != 0:
            # If q contains NaN, return an identity matrix as biorbd will throw an error otherwise
            return np.identity(4)
        else:
            mesh_rt = (
                super(BiorbdModel, self)
                .segments[segment_index]
                .segment.characteristics()
                .mesh()
                .getRotation()
                .to_array()
            )
            # mesh_rt = self.segments[segment_index].segment.characteristics().mesh().getRotation().to_array()
            segment_rt = self.segment_homogeneous_matrices_in_global(q, segment_index=segment_index)
            return segment_rt @ mesh_rt
