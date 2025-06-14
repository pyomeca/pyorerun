import os
from functools import cached_property
from xml.dom import minidom

import numpy as np
import opensim as osim
from scipy.spatial.transform import Rotation as R

# Import the abstract classes
from .abstract_model_interface import AbstractModel, AbstractModelNoMesh, AbstractSegment

MINIMAL_SEGMENT_MASS = 0.001  # Need to be this value as minimum mass of opensim segment is 0.001


class OsimSegment(AbstractSegment):  # Inherits from AbstractSegment
    """
    An interface to simplify the access to a segment of an Opensim model
    """

    def __init__(self, segment, index, model_path=None, mesh_path=None):
        self.segment = segment
        self._index: int = index
        self.model_path = model_path
        self.mesh_directory = self.check_for_mesh_path(mesh_path)

        # Private attributes
        self.__body_from_xml = None
        self.__mesh_path = None
        self.__mesh_factor = None
        self.__mesh_rt = None

    def check_for_mesh_path(self, mesh_path: str) -> str:
        """
        Check if the mesh path is valid and return it
        """
        if os.path.exists(mesh_path):
            return mesh_path
        elif os.path.exists(os.path.join(os.path.dirname(self.model_path), mesh_path)):
            return os.path.join(os.path.dirname(self.model_path), mesh_path)
        else:
            raise FileNotFoundError(
                f"Mesh path {mesh_path} does not exist." "The mesh can be set in the DisplayModelOptions class."
            )

    @cached_property
    def body_from_xml(self) -> any:
        """
        Returns the body from the xml file of the model
        """
        if self.__body_from_xml is None:
            self.model_parsed = minidom.parse(self.model_path)
            bodySet = self.model_parsed.getElementsByTagName("BodySet")[0]
            self.__body_from_xml = [
                body for body in bodySet.getElementsByTagName("Body") if body.getAttribute("name") == self.name
            ][0]
        return self.__body_from_xml

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
    def has_meshlines(self) -> bool:
        return False

    @cached_property
    def mass(self) -> float:
        return self.segment.getMass()

    @cached_property
    def mesh_path(self) -> list[str]:
        """
        Returns the mesh file of the segment
        Get the mesh file from the xml file of the model because there is no way to know how many meshes are attached to a segment in opensim.
        Therefore it raise an error in a getter and it is not possible to avoid it (even whith try/except).
        """
        if self.__mesh_path is None:
            self.__mesh_path = []
            meshes = self.body_from_xml.getElementsByTagName("Mesh")
            if len(meshes) != 0:
                self.__mesh_path = [mesh.getElementsByTagName("mesh_file")[0].firstChild.nodeValue for mesh in meshes]
                self.__mesh_path = [os.path.join(self.mesh_directory, mesh) for mesh in self.__mesh_path]
        return self.__mesh_path

    @cached_property
    def mesh_scale_factor(self) -> list[np.ndarray]:
        """
        Returns the mesh file of the segment
        Get the mesh file from the xml file of the model because there is no way to know how many meshes are attached to a segment in opensim.
        Therefore it raise an error in a getter and it is not possible to avoid it (even whith try/except).
        """
        if self.__mesh_factor is None:
            self.__mesh_factor = []
            meshes = self.body_from_xml.getElementsByTagName("Mesh")
            if len(meshes) != 0:
                self.__mesh_factor = [
                    mesh.getElementsByTagName("scale_factors")[0].firstChild.nodeValue for mesh in meshes
                ]
                self.__mesh_factor = [np.array(scale.split(" ")).astype(float) for scale in self.__mesh_factor]
        return self.__mesh_factor

    @cached_property
    def mesh_rt(self) -> list[np.ndarray]:
        """
        Returns the mesh rotation and translation matrix of the segment
        """
        if self.__mesh_rt is None:
            self.__mesh_rt = []
            meshes = self.body_from_xml.getElementsByTagName("Mesh")
            physical_offset = self.body_from_xml.getElementsByTagName("PhysicalOffsetFrame")
            path_meshes = [mesh.getElementsByTagName("mesh_file")[0].firstChild.nodeValue for mesh in meshes]
            path_physical_offset = [
                offset.getElementsByTagName("mesh_file")[0].firstChild.nodeValue for offset in physical_offset
            ]
            mesh_with_offset = [mesh for mesh in path_meshes if mesh in path_physical_offset]
            count = 0
            for i in range(len(path_meshes)):
                if not path_meshes[i] in mesh_with_offset:
                    self.__mesh_rt.extend(np.eye(4))
                elif len(physical_offset) != 0:
                    rt_matrix = np.eye(4)
                    translations = physical_offset[count].getElementsByTagName("translation")[0].firstChild.nodeValue
                    translations = np.array(translations.split(" ")).astype(float)
                    orientations = physical_offset[count].getElementsByTagName("orientation")[0].firstChild.nodeValue
                    orientations = np.array(orientations.split(" ")).astype(float)
                    rotation_matrix = R.from_euler("xyz", orientations).as_matrix()
                    rt_matrix[:3, :3] = rotation_matrix
                    rt_matrix[:3, 3] = translations
                    self.__mesh_rt.extend([rt_matrix])
                    count += 1
        return self.__mesh_rt


class OsimModelNoMesh(AbstractModelNoMesh):  # Inherits from AbstractModelNoMesh
    """
    A class to handle an Opensim model and its transformations
    """

    def __init__(self, path: str, options=None, loaded_model=None):
        super().__init__(path, options)
        if loaded_model:
            self.path = loaded_model.getInputFileName()
        self.model = loaded_model if loaded_model is not None else osim.Model(self.path)
        self.state = self.model.initSystem()
        self.coordinate_set = self.model.getCoordinateSet()
        self.previous_q = None
        self.xp_coordinate_names = None
        self.state_variables = self.model.getStateVariableValues(self.state).to_numpy()

        # Private attributes
        self.__segments = None
        self.__segment_names_with_mass = None
        self.__segments_with_mass = None

    @classmethod
    def from_osim_object(cls, model: osim.Model, options=None):
        return cls(path="", loaded_model=model, options=options)

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
        if self.__segments is None:
            self.__segments = tuple(
                [OsimSegment(s, i, self.path, self.options.mesh_path) for i, s in enumerate(self.model.getBodySet())]
            )
        return self.__segments

    @cached_property
    def segments_with_mass(self) -> tuple[OsimSegment, ...]:
        if self.__segments_with_mass is None or self.__segment_names_with_mass is None:
            self.__segment_names_with_mass = tuple([s.name for s in self.segments if s.mass > MINIMAL_SEGMENT_MASS])
            self.__segments_with_mass = tuple([s for s in self.segments if s.mass > MINIMAL_SEGMENT_MASS])
        return self.__segments_with_mass

    @cached_property
    def segment_names_with_mass(self) -> tuple[str, ...]:
        if self.__segment_names_with_mass is None:
            _ = self.segments_with_mass
        return self.__segment_names_with_mass

    def segment_homogeneous_matrices_in_global(self, q: np.ndarray, segment_index: int) -> np.ndarray:
        self._update_kinematics(q)
        transform = self.model.getBodySet().get(segment_index).getTransformInGround(self.state)
        T = transform.T().to_numpy()
        R_ = transform.R()
        R = np.array(
            [
                [R_.get(0, 0), R_.get(0, 1), R_.get(0, 2)],
                [R_.get(1, 0), R_.get(1, 1), R_.get(1, 2)],
                [R_.get(2, 0), R_.get(2, 1), R_.get(2, 2)],
            ]
        )
        return np.block([[R, T.reshape(3, 1)], [np.zeros(3), 1]])

    def markers(self, q: np.ndarray) -> np.ndarray:
        """
        Returns a [N_markers x 3] array containing the position of each marker in the global reference frame
        """
        self._update_kinematics(q)
        return np.array([mark.getLocationInGround(self.state).to_numpy() for mark in self.model.getMarkerSet()])

    def centers_of_mass(self, q: np.ndarray) -> np.ndarray:
        """
        Returns the position of the centers of mass in the global reference frame
        """
        self._update_kinematics(q)
        all_com_with_mass = np.zeros((len(self.segments_with_mass), 3))
        count = 0
        for i in range(self.model.getNumBodies()):
            if self.model.getBodySet().get(i).getMass() > MINIMAL_SEGMENT_MASS:
                rt_matrix = self.segment_homogeneous_matrices_in_global(q, i)
                com_local = self.model.getBodySet().get(i).getMassCenter().to_numpy()
                all_com_with_mass[count, :] = np.dot(rt_matrix, np.append(com_local, [1]))[:3]
                count += 1
        return all_com_with_mass

    @cached_property
    def nb_ligaments(self) -> int:
        return 0

    @cached_property
    def ligament_names(self) -> tuple[str, ...]:
        raise NotImplementedError("Ligaments are not implemented in the OpenSim interface.")

    def ligament_strips(self, q: np.ndarray) -> list[list[np.ndarray]]:
        raise NotImplementedError("Ligaments are not implemented in the OpenSim interface.")

    @cached_property
    def nb_muscles(self) -> int:
        return self.model.getForceSet().getMuscles().getSize()

    @cached_property
    def muscle_names(self) -> tuple[str, ...]:
        return tuple([s.getName() for s in self.model.get_ForceSet().getMuscles()])

    def muscle_strips(self, q: np.ndarray) -> list[list[np.ndarray]]:
        """
        Returns the position of the ligaments in the global reference frame
        """
        self._update_kinematics(q)
        muscles = []
        for idx in range(self.nb_muscles):
            muscle = self.model.getMuscles().get(idx)
            muscle_strip = []
            path_point = muscle.getGeometryPath().getPathPointSet()
            for p in range(path_point.getSize()):
                muscle_strip.append(list(path_point.get(p).getLocationInGround(self.state).to_numpy()))
            muscles.append(muscle_strip)
        return muscles

    @cached_property
    def nb_q(self) -> int:
        return self.model.getNumCoordinates()

    @cached_property
    def dof_names(self) -> tuple[str, ...]:
        return tuple(s.toString() for s in self.coordinate_set)

    @cached_property
    def q_ranges(self) -> tuple[tuple[float, float], ...]:
        return tuple((c.getRangeMin(), c.getRangeMax()) for c in self.coordinate_set)

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
        return None

    def rigid_contacts(self, q: np.ndarray) -> None:
        return None

    @cached_property
    def soft_contacts_names(self) -> None:
        return None

    @cached_property
    def rigid_contacts_names(self) -> None:
        return None

    @cached_property
    def soft_contact_radii(self) -> None:
        return None

    def _update_kinematics(self, q: np.ndarray) -> None:
        """
        Updates the kinematics of the model
        """
        if self.previous_q is not None and np.allclose(q, self.previous_q):
            return
        self.previous_q = q.copy()
        map_q = np.array([[q_val, 0] for q_val in self.previous_q]).flatten()
        self.state_variables[: self.nb_q * 2] = map_q
        self.model.setStateVariableValues(self.state, osim.Vector(self.state_variables))
        self.model.realizePosition(self.state)


class OsimModel(OsimModelNoMesh, AbstractModel):  # Inherits from OsimModelNoMesh and AbstractModel
    """
    This class extends the OsimModelNoMesh class and overrides the segments property.
    It filters the segments to only include those that have a mesh.
    """

    def __init__(self, path, options=None, loaded_model=None):
        super().__init__(path, options, loaded_model)

    @property
    def segments(self) -> tuple[OsimSegment, ...]:
        return tuple([s for s in super().segments if s.has_mesh])

    @cached_property
    def meshlines(self) -> list[np.ndarray]:
        return []

    def mesh_homogenous_matrices_in_global(
        self, q: np.ndarray, segment_index: int, mesh_index: int = None
    ) -> np.ndarray:
        """
        Returns a list of homogeneous matrices of the mesh in the global reference frame
        """
        segment_rt = self.segment_homogeneous_matrices_in_global(q, segment_index=segment_index)
        # In this implementation, mesh_index from kwargs is ignored.
        # mesh_index = kwargs.get("mesh_index")
        return segment_rt @ np.eye(4)  # self.segments[segment_index].mesh_rt[mesh_index]
