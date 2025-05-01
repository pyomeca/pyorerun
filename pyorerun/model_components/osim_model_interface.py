from functools import cached_property
from xml.dom import minidom

import opensim as osim
from typing import List, Union

import numpy as np

from .model_display_options import DisplayModelOptions

MINIMAL_SEGMENT_MASS = 1e-08


class OsimSegment:
    """
    An interface to simplify the access to a segment of a biorbd model
    """

    def __init__(self, segment, index, model_path=None):
        self.segment = segment
        self._index: int = index
        self.mesh_path = None
        self.model_path = model_path

    @cached_property
    def name(self) -> str:
        return self.segment.getName()

    @cached_property
    def id(self) -> int:
        return self._index

    @cached_property
    def has_mesh(self) -> bool:
        return len(self.get_mesh_path()) > 0
    
    @cached_property
    def has_meshlines(self) -> bool:
        return False

    @cached_property
    def mass(self) -> float:
        return self.segment.get_mass()
    
    def get_mesh_path(self) -> list[str]:
        """
        Returns the mesh file of the segment
        Get the mesh file from the xml file of the model because there is no way to know how many meshes are attached to a segment in opensim.
        Therefore it raise an error in a getter and it is not possible to avoid it (even whith try/except).
        """
        if self.mesh_path is None:
            xmldoc = minidom.parse(self.model_path)
            bodySet = xmldoc.getElementsByTagName('BodySet')[0]
            body = [body for body in bodySet.getElementsByTagName('Body') if body.getAttribute('name') == self.name][0]
            try : 
                meshes=body.getElementsByTagName('Mesh')  
                self.mesh_path = [mesh.getElementsByTagName('mesh_file')[0].firstChild.nodeValue for mesh in meshes]
                self.mesh_path = r"D:\Documents\Programmation\pyorerun\examples\biorbd\models\Geometry_cleaned" + f"\{self.mesh_path[0]}"
            except IndexError:
                self.mesh_path = []
        return self.mesh_path


class OsimModelNoMesh:
    """
    A class to handle a biorbd model and its transformations
    """

    def __init__(self, path: str, options=None, loaded_model=None):
        self.segment_names_with_mass_prop = None
        self.segments = None
        self.segments_with_mass_prop = None
        self.path = path if loaded_model is None else loaded_model.getInputFileName()
        self.model = loaded_model if loaded_model is not None else osim.Model(path)
        self.state = self.model.initSystem()
        self.options: DisplayModelOptions = options if options is not None else DisplayModelOptions()
        self.previous_q = None
        self.xp_coordinate_names = None

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

    # @cached_property
    def get_segments(self) -> tuple[OsimSegment, ...]:
        if self.segments is None:
            self.segments = []
            for i, s in enumerate(self.model.getBodySet()):
                self.segments.append(OsimSegment(s, i, self.path))
            # self.segments_prop = tuple(OsimSegment(s, i) for i, s in enumerate(self.model.getBodySet()))
        return self.segments

    @cached_property
    def segments_with_mass(self) -> tuple[OsimSegment, ...]:
        if self.segments_with_mass_prop is None or self.segment_names_with_mass_prop is None:
            segments_names_with_mass = []
            segments_with_mass = []
            if self.segments is None:
                self.segments = self.get_segments()
            for idx, s in enumerate(self.segments):
                if s.mass > MINIMAL_SEGMENT_MASS:
                    segments_names_with_mass.append(s.name)
                    segments_with_mass.append(s)
            self.segment_names_with_mass_prop = tuple(segments_names_with_mass)
            self.segments_with_mass_prop = tuple(segments_with_mass)
        return self.segments_with_mass_prop
        # return tuple([s for s in self.segments if s.mass > MINIMAL_SEGMENT_MASS])

    @cached_property
    def segment_names_with_mass(self) -> tuple[str, ...]:
        if self.segment_names_with_mass_prop is None:
            _ = self.segments_with_mass
        return self.segment_names_with_mass_prop

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
    def q_ranges(self) -> tuple[tuple[float, float], ...]:
        return tuple((c.getRangeMin(), c.getRangeMax()) for c in self.model.getCoordinateSet())

    @cached_property
    def gravity(self) -> np.ndarray:
        return self.model.getGravity().to_numpy()

    @cached_property
    def has_mesh(self) -> bool:
        if self.segments is None:
            self.segments = self.get_segments()
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
    
    def set_xp_coordinate_names(self, names: list[str]) -> None:
        """
        Set the names of the coordinates
        """
        self.xp_coordinate_names = names

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
        # if len(self.dof_names) != len(q) and self.xp_coordinate_names is None:
        #     raise ValueError("The number of coordinates in the model and the number of coordinates in the q array do not match."
        #                      "Make sure to set the experimental coordinate names in the model.")
        xp_coordinate_names = self.xp_coordinate_names if self.xp_coordinate_names is not None else self.dof_names
        
        coordinates = [self.model.getCoordinateSet().get(coord) for coord in self.dof_names]

        [coordinate.setValue(self.state, q[i], enforceContraints=False) for i, coordinate in enumerate(coordinates)]
        self.model.realizePosition(self.state)

class OsimModel(OsimModelNoMesh):
    """
    This class extends the BiorbdModelNoMesh class and overrides the segments property.
    It filters the segments to only include those that have a mesh.
    """

    def __init__(self, path, options=None, loaded_model=None):
        super().__init__(path, options, loaded_model)

    # @property
    # def get_segments(self) -> tuple[OsimSegment, ...]:
    #     if self.segments is None:
    #         self.segments = []
    #         for i, s in enumerate(self.model.getBodySet()):
    #             s_object = OsimSegment(s, i)
    #             if s_object.has_mesh:
    #                 self.segments.append(s_object)
    #     return self.segments
        # return tuple([s for s in super().segments if s.has_mesh])

    # @cached_property
    # def meshlines(self) -> list[np.ndarray]:
    #     return []

    def mesh_homogenous_matrices_in_global(self, q: np.ndarray, segment_index: int) -> np.ndarray:
        """
        Returns a list of homogeneous matrices of the mesh in the global reference frame
        """
        mesh_rt = np.eye(4)
        segment_rt = self.segment_homogeneous_matrices_in_global(q, segment_index=segment_index)
        return segment_rt @ mesh_rt
