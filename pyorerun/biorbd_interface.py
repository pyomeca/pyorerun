import biorbd
import numpy as np
from biorbd import GeneralizedCoordinates

from .mesh_utils import load_biorbd_meshes


class BiorbdModel:
    """
    A class to handle a biorbd model and its transformations
    """

    def __init__(self, path):
        self.path = path
        self.model = biorbd.Model(path)
        self.meshes = load_biorbd_meshes(self.model)

    def segment_homogeneous_matrices_in_global(self, q: np.ndarray, segment_index: int) -> np.ndarray:
        """
        Returns a biorbd object containing the roto-translation matrix of the segment in the global reference frame.
        This is useful if you want to interact with biorbd directly later on.
        """
        rt_matrix = self.model.globalJCS(GeneralizedCoordinates(q), segment_index)
        return rt_matrix.to_array()

    def all_segment_homogeneous_matrices_in_global(self, q: np.ndarray) -> np.ndarray:
        """
        Returns a Nsegx4x4 array containing the roto-translation matrix of each segment that has a mesh
        in the global reference frame.
        """
        return np.array(
            [
                self.segment_homogeneous_matrices_in_global(q, i)
                for i in range(self.model.nbSegment())
                if self.model.segment(i).characteristics().mesh().hasMesh()
            ]
        )

    def all_frame_homogeneous_matrices(self, q: np.ndarray) -> np.ndarray:
        """
        Returns a NframesxNsegx4x4 array containing the roto-translation matrix of each segment in the global reference frame
        """
        return np.array([self.all_segment_homogeneous_matrices_in_global(q[:, i]) for i in range(q.shape[1])])

    def markers(self, q: np.ndarray) -> np.ndarray:
        """
        Returns a Nmarkersx3 array containing the position of each marker in the global reference frame
        """
        return np.array(
            [self.model.markers(GeneralizedCoordinates(q))[i].to_array() for i in range(self.model.nbMarkers())]
        )

    def all_frame_markers(self, q: np.ndarray) -> np.ndarray:
        """
        Returns a NframesxNmarkersx3 array containing the position of each marker in the global reference frame
        """
        return np.array([self.markers(q[:, i]) for i in range(q.shape[1])])

    @property
    def marker_names(self) -> tuple[str, ...]:
        return tuple([s.to_string() for s in self.model.markerNames()])

    @property
    def nb_markers(self) -> int:
        return self.model.nbMarkers()
