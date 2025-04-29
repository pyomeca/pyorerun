from functools import cached_property

import opensim as osim
from typing import List, Union

import numpy as np
# from biorbd import GeneralizedCoordinates

from .model_display_options import DisplayModelOptions

MINIMAL_SEGMENT_MASS = 1e-08


class OsimSegment:
    """
    An interface to simplify the access to a segment of a biorbd model
    """

    def __init__(self, segment, index):
        self.segment = segment
        self._index: int = index

    @cached_property
    def name(self) -> str:
        return self.segment.getName()

    @cached_property
    def id(self) -> int:
        return self._index

    @cached_property
    def has_mesh(self) -> bool:
        return len(self.mesh_path) > 0

    @cached_property
    def mesh_path(self) -> Union[str, List[str]]:
        mesh_files = []
        count = 0
        while True:
            try:
                mesh_files.append(self.segment.get_attached_geometry(count).getPropertyByName("mesh_file").toString())
            except ValueError:
                break
        return mesh_files

    @cached_property
    def mass(self) -> float:
        return self.segment.get_mass()


class BiorbdModelNoMesh:
    """
    A class to handle a biorbd model and its transformations
    """

    def __init__(self, path: str, options=None):
        self.path = path
        self.model = osim.Model(path)
        self.state = self.model.initSystem()
        self.options: DisplayModelOptions = options if options is not None else DisplayModelOptions()
        self.previous_q = None

    @classmethod
    def from_biorbd_object(cls, model: osim.Model, options=None):
        return cls(model.getInputFileName(), options)

    @cached_property
    def name(self):
        return self.model.getName()

    @cached_property
    def marker_names(self) -> tuple[str, ...]:
        return tuple([s.getName() for s in self.model.getMarkerSet()])

    @cached_property
    def nb_markers(self) -> int:
        return self.model.getNumMarkers()

    @cached_property
    def segment_names(self) -> tuple[str, ...]:
        return tuple([s.getName() for s in self.model.getBodySet()])

    @cached_property
    def nb_segments(self) -> int:
        return self.model.getNumBodies()

    @cached_property
    def segments(self) -> tuple[OsimSegment, ...]:
        return tuple(OsimSegment(s, i) for i, s in enumerate(self.model.getBodySet()))

    @cached_property
    def segments_with_mass(self) -> tuple[OsimSegment, ...]:
        return tuple([s for s in self.segments if s.mass > MINIMAL_SEGMENT_MASS])

    @cached_property
    def segment_names_with_mass(self) -> tuple[str, ...]:
        return tuple([s.name for s in self.segments_with_mass])

    def segment_homogeneous_matrices_in_global(self, q: np.ndarray, segment_index: int) -> np.ndarray:
        """
        Returns a biorbd object containing the roto-translation matrix of the segment in the global reference frame.
        This is useful if you want to interact with biorbd directly later on.
        """
        self.update_kinematics(q)
        transform = self.model.getBodySet().get(segment_index).getTransformInGround(self.state)
        T = transform.T().to_numpy()
        R = transform.R()
        R = np.array([[R.get(0, 0), R.get(0, 1), R.get(0, 2)],
                      [R.get(1, 0), R.get(1, 1), R.get(1, 2)],
                      [R.get(2, 0), R.get(2, 1), R.get(2, 2)]])
        return np.block([[R, T.reshape(3, 1)], [np.zeros(3), 1]])

    def markers(self, q: np.ndarray) -> np.ndarray:
        """
        Returns a [N_markers x 3] array containing the position of each marker in the global reference frame
        """
        self.update_kinematics(q)
        return np.array([mark.getLocationInGround(self.state).to_numpy() for mark in self.model.getMarkerSet()])

    def centers_of_mass(self, q: np.ndarray) -> np.ndarray:
        """
        Returns the position of the centers of mass in the global reference frame
        """
        self.update_kinematics(q)
        all_com_with_mass = np.zeros((len(self.segments_with_mass), 3))
        for i in range(self.model.getNumBodies()):
            if self.model.getBodySet().get(i).getMass() > MINIMAL_SEGMENT_MASS:
                rt_matrix = self.segment_homogeneous_matrices_in_global(q, 0)
                com_local = self.model.getBodySet().get(0).getMassCenter().to_numpy()
                all_com_with_mass[i, :] = np.dot(rt_matrix, np.append(com_local, [1]))[:3]
        return all_com_with_mass

    @cached_property
    def nb_ligaments(self) -> int:
        """
        Returns the number of ligaments
        """
        return 0

    @cached_property
    def ligament_names(self) -> tuple[str, ...]:
        """
        Returns the names of the ligaments
        """
        # return tuple([s.toString() for s in self.model.ligamentNames()])
        raise NotImplementedError("Ligament names are not implemented yet")

    def ligament_strips(self, q: np.ndarray) -> list[list[np.ndarray]]:
        """
        Returns the position of the ligaments in the global reference frame
        """
        # ligaments = []
        # self.model.updateLigaments(q, True)
        # for ligament_idx in range(self.nb_ligaments):
        #     ligament = self.model.ligament(ligament_idx)
        #     ligament_strip = []
        #     for pts in ligament.position().pointsInGlobal():
        #         ligament_strip.append(pts.to_array().tolist())
        #     ligaments.append(ligament_strip)
        # return ligaments
        raise NotImplementedError("Ligament strips are not implemented yet")

    @cached_property
    def nb_muscles(self) -> int:
        """
        Returns the number of ligaments
        """
        return self.model.getForceSet().getMuscles().getSize()

    @cached_property
    def muscle_names(self) -> tuple[str, ...]:
        """
        Returns the names of the ligaments
        """
        return tuple([s.getName() for s in self.model.get_ForceSet().getMuscles()])

    def muscle_strips(self, q: np.ndarray) -> list[list[np.ndarray]]:
        """
        Returns the position of the ligaments in the global reference frame
        """
        # muscles = []
        # self.model.updateMuscles(q, True)
        # for idx in range(self.nb_muscles):
        #     muscle = self.model.muscle(idx)
        #     muscle_strip = []
        #     for pts in muscle.position().pointsInGlobal():
        #         muscle_strip.append(pts.to_array().tolist())
        #     muscles.append(muscle_strip)
        # return muscles
        raise NotImplementedError("Muscle strips are not implemented yet")

    @cached_property
    def nb_q(self) -> int:
        return self.model.getNumCoordinates()

    @cached_property
    def dof_names(self) -> tuple[str, ...]:
        return tuple(s.toString() for s in self.model.getCoordinateSet())

    @cached_property
    @cached_property
    def q_ranges(self) -> tuple[tuple[float, float], ...]:
        return tuple((c.getRangeMin(), c.getRangeMax()) for c in self.model.getCoordinateSet())

    @cached_property
    def gravity(self) -> np.ndarray:
        return self.model.getGravity().to_numpy()

    @cached_property
    def has_mesh(self) -> bool:
        return any([s.has_mesh for s in self.segments])

    @cached_property
    def has_meshlines(self) -> bool:
        return False

    @cached_property
    def has_soft_contacts(self) -> bool:
        return False

    @property
    def has_rigid_contacts(self) -> bool:
        return False

    def soft_contacts(self, q: np.ndarray) -> None:
        """
        Returns the position of the soft contacts spheres in the global reference frame
        """
        return None

    def rigid_contacts(self, q: np.ndarray) -> None:
        """
        Returns the position of the rigid contacts in the global reference frame
        """
        return None

    @cached_property
    def soft_contacts_names(self) -> None:
        """
        Returns the names of the soft contacts
        """
        return None

    @cached_property
    def rigid_contacts_names(self) -> None:
        """
        Returns the names of the soft contacts
        """
        return None

    @cached_property
    def soft_contact_radii(self) -> None:
        """
        Returns the radii of the soft contacts
        """
        return None

    def update_kinematics(self, q: np.ndarray) -> None:
        """
        Updates the kinematics of the model
        """
        if self.previous_q is not None and np.allclose(q, self.previous_q):
            return
        self.previous_q = q.copy()
        [coordinate.setValue(self.state, q[i]) for i, coordinate in enumerate(self.model.getCoordinateSet())]


class BiorbdModel(BiorbdModelNoMesh):
    """
    This class extends the BiorbdModelNoMesh class and overrides the segments property.
    It filters the segments to only include those that have a mesh.
    """

    def __init__(self, path, options=None):
        super().__init__(path, options)

    @property
    def segments(self) -> tuple[OsimSegment, ...]:
        return tuple([s for s in super().segments if s.has_mesh or s.has_meshlines])

    @cached_property
    def meshlines(self) -> list[np.ndarray]:

        meshes = []
        for segment in self.segments:
            segment_mesh = segment.segment.characteristics().mesh()
            meshes += [np.array([segment_mesh.point(i).to_array() for i in range(segment_mesh.nbVertex())])]

        return meshes

    def mesh_homogenous_matrices_in_global(self, q: np.ndarray, segment_index: int) -> np.ndarray:
        """
        Returns a list of homogeneous matrices of the mesh in the global reference frame
        """
        mesh_rt = (
            super(BiorbdModel, self).segments[segment_index].segment.characteristics().mesh().getRotation().to_array()
        )
        # mesh_rt = self.segments[segment_index].segment.characteristics().mesh().getRotation().to_array()
        segment_rt = self.segment_homogeneous_matrices_in_global(q, segment_index=segment_index)
        return segment_rt @ mesh_rt
