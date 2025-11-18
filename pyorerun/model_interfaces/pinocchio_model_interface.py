import os
from functools import cached_property
from pathlib import Path

import numpy as np
import pinocchio as pin

# Import the abstract classes
from .abstract_model_interface import AbstractModel, AbstractModelNoMesh, AbstractSegment
from ..model_components.model_display_options import DisplayModelOptions

MINIMAL_SEGMENT_MASS = 1e-08


class PinocchioSegment(AbstractSegment):
    """
    An interface to simplify the access to a segment (body/frame) of a Pinocchio model
    """

    def __init__(self, model, frame_id: int, mesh_dir: Path = None):
        self.model = model
        self.frame = model.frames[frame_id]
        self._index = frame_id
        self._mesh_dir = mesh_dir

        # Cache for visual geometries associated with this frame
        self._visual_geometries = None

    @cached_property
    def name(self) -> str:
        return self.frame.name

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
    def mesh_path(self) -> list[str]:
        """
        Returns the mesh file paths associated with this frame/body.
        Only returns actual mesh files, not primitive shapes (spheres, boxes, etc.).
        """
        mesh_paths = []
        if self._visual_geometries is not None:
            for geom in self._visual_geometries:
                # meshPath is an attribute of GeometryObject, not geometry
                # Check if it has a meshPath and it's not empty (primitives have empty or just shape name)
                if hasattr(geom, "meshPath") and geom.meshPath:
                    mesh_path_str = str(geom.meshPath)
                    # Filter out primitive shapes (they don't have file extensions)
                    if "." in mesh_path_str and os.path.splitext(mesh_path_str)[1].lower() in [
                        ".dae",
                        ".stl",
                        ".obj",
                        ".mesh",
                        ".ply",
                    ]:
                        if self._mesh_dir and not os.path.isabs(mesh_path_str):
                            mesh_path_str = str(self._mesh_dir / mesh_path_str)
                        mesh_paths.append(mesh_path_str)
        return mesh_paths

    @cached_property
    def mesh_scale_factor(self) -> list[np.ndarray]:
        """
        Returns the scale factor for each mesh.
        """
        scales = []
        if self._visual_geometries is not None:
            for geom in self._visual_geometries:
                # meshScale is an attribute of GeometryObject, not geometry
                if hasattr(geom, "meshScale"):
                    scales.append(np.array(geom.meshScale))
                else:
                    scales.append(np.ones(3))
        return scales if scales else [np.ones(3)]

    @cached_property
    def mass(self) -> float:
        """
        Returns the mass of the body associated with this frame.
        """
        # Get the parent joint id
        parent_joint_id = self.frame.parentJoint
        if parent_joint_id < len(self.model.inertias):
            return float(self.model.inertias[parent_joint_id].mass)
        return 0.0

    def set_visual_geometries(self, geometries: list):
        """
        Set the visual geometries associated with this frame.
        """
        self._visual_geometries = geometries


class PinocchioModelNoMesh(AbstractModelNoMesh):
    """
    A class to handle a Pinocchio model and its transformations
    """

    def __init__(self, path: str, options=None):
        super().__init__(path, options)

        # Determine the mesh directory (where mesh files are located relative to URDF)
        path_obj = Path(path)
        if path_obj.suffix == ".urdf":
            # For URDF files, mesh paths in the file are relative to the URDF's parent's parent
            # e.g., if URDF is at examples/pinocchio/urdf/model.urdf
            # and meshes are at examples/pinocchio/meshes/...
            # then mesh_dir should be examples/pinocchio
            self._mesh_dir = path_obj.parent.parent
        else:
            self._mesh_dir = path_obj.parent

        # Load the model
        if path.endswith(".urdf"):
            self.model = pin.buildModelFromUrdf(path)
            # Try to load visual model if available
            try:
                _, _, self.visual_model = pin.buildModelsFromUrdf(path, str(self._mesh_dir))
            except Exception as e:
                print(f"Warning: Could not load visual model: {e}")
                self.visual_model = None
        else:
            raise ValueError(f"Unsupported file format for Pinocchio: {path}. Expected .urdf")

        self.data = self.model.createData()

    @classmethod
    def from_pinocchio_object(cls, model: pin.Model, path: str = None, options=None):
        """
        Create from an existing Pinocchio model object.
        """
        instance = cls.__new__(cls)
        instance.model = model
        instance.data = model.createData()
        instance.path = path if path else "pinocchio_model"
        instance.options = options if options is not None else DisplayModelOptions()
        instance._mesh_dir = Path(path).parent if path else Path.cwd()
        instance.visual_model = None
        return instance

    @cached_property
    def name(self) -> str:
        return self.model.name if hasattr(self.model, "name") else Path(self.path).stem

    @cached_property
    def marker_names(self) -> tuple[str, ...]:
        """
        In Pinocchio, markers can be represented as frames or operational frames.
        This returns all frame names that could represent markers.
        """
        # Get all frames that are not joint frames
        marker_frames = []
        for frame in self.model.frames:
            if frame.type == pin.FrameType.OP_FRAME:
                marker_frames.append(frame.name)
        return tuple(marker_frames)

    @cached_property
    def nb_markers(self) -> int:
        return len(self.marker_names)

    @cached_property
    def segment_names(self) -> tuple[str, ...]:
        """
        Returns the names of all bodies/frames in the model.
        """
        return tuple(frame.name for frame in self.model.frames if frame.type == pin.FrameType.BODY)

    @cached_property
    def nb_segments(self) -> int:
        return len(self.segment_names)

    @cached_property
    def segments(self) -> tuple[PinocchioSegment, ...]:
        """
        Returns all segments (bodies) in the model.
        """
        segments = []
        for i, frame in enumerate(self.model.frames):
            if frame.type == pin.FrameType.BODY:
                segment = PinocchioSegment(self.model, i, self._mesh_dir)

                # Associate visual geometries if available
                if self.visual_model is not None:
                    visual_geoms = []
                    for geom in self.visual_model.geometryObjects:
                        if geom.parentFrame == i:
                            visual_geoms.append(geom)
                    segment.set_visual_geometries(visual_geoms)

                segments.append(segment)
        return tuple(segments)

    @cached_property
    def segments_with_mass(self) -> tuple[PinocchioSegment, ...]:
        return tuple([s for s in self.segments if s.mass > MINIMAL_SEGMENT_MASS])

    @cached_property
    def segment_names_with_mass(self) -> tuple[str, ...]:
        return tuple([s.name for s in self.segments_with_mass])

    def segment_homogeneous_matrices_in_global(self, q: np.ndarray, segment_index: int) -> np.ndarray:
        """
        Returns the roto-translation matrix of the segment in the global reference frame.
        """
        if np.sum(np.isnan(q)) != 0:
            return np.identity(4)

        # Update the kinematics
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # Get the transformation matrix for the frame
        frame = self.model.frames[segment_index]
        oMf = self.data.oMf[segment_index]

        return oMf.homogeneous

    def markers(self, q: np.ndarray) -> np.ndarray:
        """
        Returns a [N_markers x 3] array containing the position of each marker in the global reference frame.
        """
        # Update kinematics
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        markers = []
        for frame_name in self.marker_names:
            frame_id = self.model.getFrameId(frame_name)
            position = self.data.oMf[frame_id].translation
            markers.append(position)

        return np.array(markers)

    def centers_of_mass(self, q: np.ndarray) -> np.ndarray:
        """
        Returns the position of the centers of mass in the global reference frame.
        """
        # Update kinematics
        pin.forwardKinematics(self.model, self.data, q)

        all_com = []
        for segment in self.segments_with_mass:
            # Get the parent joint
            frame = self.model.frames[segment.id]
            parent_joint = frame.parentJoint

            if parent_joint < len(self.data.oMi):
                # Get joint placement
                oMi = self.data.oMi[parent_joint]
                # Get local CoM position
                inertia = self.model.inertias[parent_joint]
                local_com = inertia.lever
                # Transform to global frame
                global_com = oMi.act(local_com)
                all_com.append(global_com)

        return np.array(all_com) if all_com else np.zeros((0, 3))

    @cached_property
    def nb_q(self) -> int:
        return self.model.nq

    @cached_property
    def dof_names(self) -> tuple[str, ...]:
        """
        Returns the names of all degrees of freedom.
        """
        return tuple(self.model.names[1:])  # Skip universe/world

    @cached_property
    def q_ranges(self) -> tuple[tuple[float, float], ...]:
        """
        Returns the joint limits for all DoFs.
        """
        q_ranges = []
        for i in range(self.model.nq):
            lower = self.model.lowerPositionLimit[i]
            upper = self.model.upperPositionLimit[i]
            q_ranges.append((float(lower), float(upper)))
        return tuple(q_ranges)

    @cached_property
    def nb_contacts(self) -> int:
        """
        Returns the number of contact points (if defined as frames).
        """
        contact_frames = [f for f in self.model.frames if "contact" in f.name.lower()]
        return len(contact_frames)

    @cached_property
    def contact_names(self) -> tuple[str, ...]:
        """
        Returns the names of contact frames.
        """
        return tuple(f.name for f in self.model.frames if "contact" in f.name.lower())

    def contact_positions(self, q: np.ndarray) -> np.ndarray:
        """
        Returns the positions of contact points in the global frame.
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        contacts = []
        for contact_name in self.contact_names:
            frame_id = self.model.getFrameId(contact_name)
            position = self.data.oMf[frame_id].translation
            contacts.append(position)

        return np.array(contacts) if contacts else np.zeros((0, 3))

    @cached_property
    def nb_ligaments(self) -> int:
        """
        Pinocchio doesn't have native ligament support.
        """
        return 0

    @cached_property
    def ligament_names(self) -> tuple[str, ...]:
        """
        Pinocchio doesn't have native ligament support.
        """
        return tuple()

    def ligament_strips(self, q: np.ndarray) -> list[list[np.ndarray]]:
        """
        Pinocchio doesn't have native ligament support.
        """
        return []

    @cached_property
    def nb_muscles(self) -> int:
        """
        Pinocchio doesn't have native muscle support.
        """
        return 0

    @cached_property
    def muscle_names(self) -> tuple[str, ...]:
        """
        Pinocchio doesn't have native muscle support.
        """
        return tuple()

    def muscle_strips(self, q: np.ndarray) -> list[list[np.ndarray]]:
        """
        Pinocchio doesn't have native muscle support.
        """
        return []

    @cached_property
    def gravity(self) -> np.ndarray:
        """
        Returns the gravity vector.
        """
        return self.model.gravity.linear

    @cached_property
    def has_soft_contacts(self) -> bool:
        """
        Pinocchio uses contact frames - consider all as soft contacts.
        """
        return self.nb_contacts > 0

    @cached_property
    def has_rigid_contacts(self) -> bool:
        """
        Pinocchio doesn't distinguish rigid contacts separately.
        """
        return False

    def soft_contacts(self, q: np.ndarray) -> np.ndarray:
        """
        Returns the positions of soft contacts.
        """
        return self.contact_positions(q)

    def rigid_contacts(self, q: np.ndarray) -> np.ndarray:
        """
        No rigid contacts in Pinocchio.
        """
        return np.zeros((0, 3))

    @cached_property
    def soft_contacts_names(self) -> tuple[str, ...]:
        """
        Returns the names of soft contacts.
        """
        return self.contact_names

    @cached_property
    def rigid_contacts_names(self) -> tuple[str, ...]:
        """
        No rigid contacts in Pinocchio.
        """
        return tuple()

    @cached_property
    def soft_contact_radii(self) -> tuple[float, ...]:
        """
        Default radius for all contacts.
        """
        return tuple(0.01 for _ in range(self.nb_contacts))


class PinocchioModel(PinocchioModelNoMesh, AbstractModel):
    """
    A Pinocchio model with mesh support, inheriting from both PinocchioModelNoMesh and AbstractModel.
    """

    def __init__(self, path: str, options=None):
        # Parent class already loads the model and visual_model
        super().__init__(path, options)
        # No need to reload - visual_model is already set by parent class

    @classmethod
    def from_pinocchio_object(
        cls,
        model: pin.Model,
        visual_model: pin.GeometryModel = None,
        collision_model: pin.GeometryModel = None,
        path: str = None,
        options=None,
    ):
        """
        Create from existing Pinocchio objects.
        """
        instance = cls.__new__(cls)
        instance.model = model
        instance.data = model.createData()
        instance.visual_model = visual_model
        instance.collision_model = collision_model
        instance.path = path if path else "pinocchio_model"
        instance.options = options if options is not None else DisplayModelOptions()
        instance._mesh_dir = Path(path).parent if path else Path.cwd()
        return instance

    @cached_property
    def has_mesh(self) -> bool:
        """
        Whether any segment in the model has a visual mesh.
        """
        return any(s.has_mesh for s in super().segments)

    @property
    def segments(self) -> tuple[PinocchioSegment, ...]:
        """
        Returns only segments that have meshes (filtering out segments without visual geometry).
        """
        return tuple([s for s in super().segments if s.has_mesh])

    @cached_property
    def has_meshlines(self) -> bool:
        """
        Pinocchio doesn't support mesh lines.
        """
        return False

    @cached_property
    def meshlines(self) -> list[np.ndarray]:
        """
        Pinocchio doesn't support mesh lines.
        """
        return []

    def mesh_homogenous_matrices_in_global(self, q: np.ndarray, segment_index: int, **kwargs) -> np.ndarray:
        """
        Get the 4x4 homogeneous transformation matrix of a mesh in the global frame.
        """
        return self.segment_homogeneous_matrices_in_global(q, segment_index)
