from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from biobuddy import NamedList, SegmentReal, BiomechanicalModelReal
    
from functools import cached_property

import numpy as np

# Import the abstract classes
from .abstract_model_interface import AbstractModel, AbstractModelNoMesh, AbstractSegment

MINIMAL_SEGMENT_MASS = 1e-08


class BiobuddySegment(AbstractSegment):  # Inherits from AbstractSegment
    """
    An interface to simplify the access to a segment of a biobuddy model
    """

    def __init__(self, segment, index):
        self.segment = segment
        self._index: int = index

    @cached_property
    def name(self) -> str:
        return self.segment.name

    @cached_property
    def id(self) -> int:
        return self._index

    @cached_property
    def has_mesh(self) -> bool:
        has_mesh = self.segment.mesh_file is not None
        if has_mesh:
            return not self.segment.mesh_file.mesh_file_name.endswith("/")  # Avoid empty mesh path
        return has_mesh

    @cached_property
    def has_meshlines(self) -> bool:
        """
        * Not implemented in biobuddy yet, so returns False for now *
        """
        return False

    @cached_property
    def mesh_path(self) -> list[str]:
        return [self.segment.mesh_file.mesh_file_name]

    @cached_property
    def mesh_scale_factor(self) -> list[np.ndarray]:
        """
        return: numpy array (3,) of the scale factor of the mesh
        """
        return [self.segment.mesh_file.mesh_scale.reshape(-1)[:3]]

    @cached_property
    def mass(self) -> float:
        return self.segment.mass


class BiobuddyModelNoMesh(AbstractModelNoMesh):  # Inherits from AbstractModelNoMesh
    """
    A class to handle a biorbd model and its transformations
    """

    def __init__(self, path: str = None, options=None):
        """
        A biobuddy.BiomechanicalModelReal cannot be created from a path directly, so we need to use the
        BiobuddyModelNoMesh.from_biobuddy_object() to set it.
        """
        if path is not None:
            raise NotImplementedError("Loading a model from a path is not implemented yet for BioBuddy.")
        super().__init__(path, options)
        self.model = None

    @classmethod
    def from_biobuddy_object(cls, model: BiomechanicalModelReal, options=None):
        new_object = cls(None, None)
        new_object.model = model
        if options is not None:
            new_object.options = options
        return new_object

    @cached_property
    def name(self):
        return "BioBuddy model"

    @cached_property
    def marker_names(self) -> tuple[str, ...]:
        return self.model.marker_names

    @cached_property
    def nb_markers(self) -> int:
        return self.model.nb_markers

    @cached_property
    def segment_names(self) -> tuple[str, ...]:
        return self.model.segment_names

    @cached_property
    def nb_segments(self) -> int:
        return self.model.nb_segments

    @cached_property
    def segments(self) -> NamedList[SegmentReal]:
        return self.model.segments

    @cached_property
    def segments_with_mass(self) -> tuple[SegmentReal]:
        segments_with_mass_list = []
        for s in self.segments:
            inertia_parameters = s.segment.inertia_parameters
            if inertia_parameters is not None:
                mass = inertia_parameters.mass
                if mass > MINIMAL_SEGMENT_MASS:
                    segments_with_mass_list.append(s)
        return tuple(segments_with_mass_list)

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
            segment_name = self.model.segment_names[segment_index]
            rt_matrix = self.model.forward_kinematics(q)[segment_name][0].rt_matrix
        return rt_matrix

    def markers(self, q: np.ndarray) -> np.ndarray:
        """
        Returns a [N_markers x 3] array containing the position of each marker in the global reference frame
        """
        return self.model.markers_in_global(q)[:3, :, 0].T

    def centers_of_mass(self, q: np.ndarray) -> np.ndarray:
        """
        Returns a [N_segment_with_mass x 3] array containing the position of the centers of mass in the global reference frame
        """
        segments_com = np.empty((0, 3))
        for s in self.segments_with_mass:
            com = self.model.segment_com_in_global(s.segment.name, q)[:3, 0]
            segments_com = np.vstack((segments_com, com))
        return segments_com

    @cached_property
    def nb_ligaments(self) -> int:
        """
        Returns the number of ligaments
        * Not implemented in biobuddy yet, so returns 0 for now *
        """
        return 0

    @cached_property
    def ligament_names(self) -> tuple[str, ...]:
        """
        Returns the names of the ligaments
        * Not implemented in biobuddy yet, so returns 0 for now *
        """
        return ()

    def ligament_strips(self, q: np.ndarray) -> list[list[np.ndarray]]:
        """
        Returns the position of the ligaments in the global reference frame
        * Not implemented in biobuddy yet, so returns 0 for now *
        """
        return []

    @cached_property
    def nb_muscles(self) -> int:
        """
        Returns the number of muscles
        """
        return self.model.nb_muscles

    @cached_property
    def muscle_names(self) -> tuple[str, ...]:
        """
        Returns the names of the muscles
        """
        return self.model.muscle_names

    def muscle_strips(self, q: np.ndarray) -> list[list[np.ndarray]]:
        """
        Returns the position of the muscles in the global reference frame
        """
        muscles = []
        for muscle_group in self.model.muscle_groups:
            for muscle in muscle_group.muscles:
                muscle_strip = []
                muscle_strip += [self.model.muscle_origin_in_global(muscle.name, q)[:3, 0].tolist()]
                if muscle.nb_via_points > 0:
                    muscle_strip += self.model.via_points_in_global(muscle.name, q)[:3, 0].T.tolist()
                muscle_strip += [self.model.muscle_insertion_in_global(muscle.name, q)[:3, 0].tolist()]
                muscles += [muscle_strip]
        return muscles

    @cached_property
    def nb_q(self) -> int:
        return self.model.nb_q

    @cached_property
    def dof_names(self) -> tuple[str, ...]:
        return tuple(s.dof_names for s in self.model.segments)

    @cached_property
    def q_ranges(self) -> tuple[tuple[float, float], ...]:
        q_ranges = [q_range for segment in self.model.segments for q_range in segment.q_ranges]
        return tuple((q_range.min_bound(), q_range.max_bound()) for q_range in q_ranges)

    @cached_property
    def gravity(self) -> np.ndarray:
        return self.model.gravity

    @cached_property
    def has_mesh(self) -> bool:
        return any([s.has_mesh for s in self.segments])

    @cached_property
    def has_meshlines(self) -> bool:
        return any([s.has_meshlines for s in self.segments])

    @cached_property
    def has_soft_contacts(self) -> bool:
        """
        * Not implemented in biobuddy yet, so returns False for now *
        """
        return False

    @property
    def has_rigid_contacts(self) -> bool:
        return self.model.nb_contacts > 0

    def soft_contacts(self, q: np.ndarray) -> np.ndarray:
        """
        Returns the position of the soft contacts spheres in the global reference frame
        * Not implemented in biobuddy yet, so returns an empty array for now *
        """
        return np.array([])

    def rigid_contacts(self, q: np.ndarray) -> np.ndarray:
        """
        Returns the position of the rigid contacts in the global reference frame
        """
        return self.model.contacts_in_global(q)[:3, :, 0].T

    @cached_property
    def soft_contacts_names(self) -> tuple[str, ...]:
        """
        Returns the names of the soft contacts
        * Not implemented in biobuddy yet, so returns an empty tuple for now *
        """
        return ()

    @cached_property
    def rigid_contacts_names(self) -> tuple[str, ...]:
        """
        Returns the names of the soft contacts
        """
        return self.model.contact_names

    @cached_property
    def soft_contact_radii(self) -> tuple[float, ...]:
        """
        Returns the radii of the soft contacts
        * Not implemented in biobuddy yet, so returns an empty tuple for now *
        """
        return ()


class BiobuddyModel(BiobuddyModelNoMesh, AbstractModel):  # Inherits from BiobuddyModelNoMesh and AbstractModel
    """
    This class extends the BiobuddyModelNoMesh class and overrides the segments property.
    It filters the segments to only include those that have a mesh.
    """

    def __init__(self, path, options=None):
        super().__init__(path, options)

    @property
    def segments(self) -> tuple[BiobuddySegment, ...]:
        segments_with_mesh = []
        for i, s in enumerate(self.model.segments):
            segment = BiobuddySegment(s, i)
            if segment.has_mesh or segment.has_meshlines:
                segments_with_mesh.append(segment)
        return tuple(segments_with_mesh)

    @cached_property
    def meshlines(self) -> list[np.ndarray]:
        raise NotImplementedError("Meshlines were not implemented for BioBuddy models.")
        # # TODO
        # meshes = []
        # for segment in self.segments:
        #     segment_mesh = segment.segment.mesh
        #     meshes += [np.array([segment_mesh.point(i).to_array() for i in range(segment_mesh.nbVertex())])]
        #
        # return meshes

    def mesh_homogenous_matrices_in_global(self, q: np.ndarray, segment_index: int, **kwargs) -> np.ndarray:
        """
        Returns a list of homogeneous matrices of the mesh in the global reference frame
        """
        if np.sum(np.isnan(q)) != 0:
            # If q contains NaN, return an identity matrix as biorbd will throw an error otherwise
            return np.identity(4)
        else:
            mesh_rt = super(BiobuddyModel, self).segments[segment_index].mesh_file.mesh_rt.rt_matrix

            # mesh_rt = self.segments[segment_index].segment.characteristics().mesh().getRotation().to_array()
            segment_rt = self.segment_homogeneous_matrices_in_global(q, segment_index=segment_index)
            return segment_rt @ mesh_rt
