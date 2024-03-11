import biorbd
import numpy as np
from biorbd import GeneralizedCoordinates, segment_index


class BiorbdSegment:
    """
    An interface to simplify the access to a segment of a biorbd model
    """

    def __init__(self, segment, index):
        self.segment = segment
        self._index: int = index

    @property
    def name(self) -> str:
        return self.segment.name().to_string()

    @property
    def id(self) -> int:
        return self._index

    @property
    def has_mesh(self) -> bool:
        return self.segment.characteristics().mesh().hasMesh()


class BiorbdModel:
    """
    A class to handle a biorbd model and its transformations
    """

    def __init__(self, path):
        self.path = path
        self.model = biorbd.Model(path)

    @property
    def name(self):
        return self.path.split("/")[-1].split(".")[0]

    @property
    def marker_names(self) -> tuple[str, ...]:
        return tuple([s.to_string() for s in self.model.markerNames()])

    @property
    def nb_markers(self) -> int:
        return self.model.nbMarkers()

    @property
    def segment_names(self) -> tuple[str, ...]:
        return tuple([s.name().to_string() for s in self.model.segments()])

    @property
    def nb_segments(self) -> int:
        return self.model.nbSegment()

    @property
    def segments(self) -> tuple[BiorbdSegment, ...]:
        return tuple(BiorbdSegment(s, i) for i, s in enumerate(self.model.segments()))

    @property
    def segments_with_mesh(self) -> tuple[BiorbdSegment, ...]:
        return tuple([s for s in self.segments if s.has_mesh])

    @property
    def mesh_paths(self) -> list[str]:
        return [
            s.characteristics().mesh().path().absolutePath().to_string()
            for s in self.model.segments()
            if s.characteristics().mesh().hasMesh()
        ]

    def segment_homogeneous_matrices_in_global(self, q: np.ndarray, segment_index: int) -> np.ndarray:
        """
        Returns a biorbd object containing the roto-translation matrix of the segment in the global reference frame.
        This is useful if you want to interact with biorbd directly later on.
        """
        rt_matrix = self.model.globalJCS(GeneralizedCoordinates(q), segment_index)
        return rt_matrix.to_array()

    def markers(self, q: np.ndarray) -> np.ndarray:
        """
        Returns a Nmarkersx3 array containing the position of each marker in the global reference frame
        """
        return np.array(
            [self.model.markers(GeneralizedCoordinates(q))[i].to_array() for i in range(self.model.nbMarkers())]
        )

    def center_of_mass(self, q: np.ndarray) -> np.ndarray:
        """
        Returns the position of the center of mass in the global reference frame
        """
        return self.model.CoM(GeneralizedCoordinates(q)).to_array()

    @property
    def nb_ligaments(self) -> int:
        """
        Returns the number of ligaments
        """
        return self.model.nbLigaments()

    @property
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

    @property
    def nb_muscles(self) -> int:
        """
        Returns the number of ligaments
        """
        return self.model.nbMuscles()

    @property
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
