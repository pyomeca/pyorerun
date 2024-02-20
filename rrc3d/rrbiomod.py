import biorbd
import numpy as np
import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!
import trimesh
from biorbd import GeneralizedCoordinates

MY_STL = "my_stl"


class TransformableMesh:
    """
    A class to handle a trimesh object and its transformations
    and always 'apply_transform' from its initial position
    """

    def __init__(self, mesh: trimesh.Trimesh):
        self.__mesh = mesh
        self.transformed_mesh = mesh.copy()

    def apply_transform(self, transform) -> trimesh.Trimesh:
        """Apply a transform to the mesh from its initial position"""
        self.transformed_mesh = self.__mesh.copy()
        self.transformed_mesh.apply_transform(transform)

        return self.transformed_mesh

    @property
    def mesh(self):
        return self.__mesh


def load_biorbd_meshes(biomod: biorbd.Model) -> list[TransformableMesh]:
    """
    Load all the meshes from a biorbd model
    """
    meshes = []
    for i in range(biomod.nbSegment()):
        stl_file_path = (
            biomod.segment(i).characteristics().mesh().path().absolutePath().to_string()
        )
        mesh = trimesh.load(stl_file_path, file_type="stl")
        meshes.append(TransformableMesh(mesh))

    return meshes


class BiorbdModel:
    def __init__(self, path):
        self.path = path
        self.model = biorbd.Model(path)
        self.meshes = load_biorbd_meshes(self.model)

    def segment_homogeneous_matrices_in_global(
        self, q: np.ndarray, segment_index: int
    ) -> np.ndarray:
        """
        Returns a biorbd object containing the roto-translation matrix of the segment in the global reference frame.
        This is useful if you want to interact with biorbd directly later on.
        """
        rt_matrix = self.model.globalJCS(GeneralizedCoordinates(q), segment_index)
        return rt_matrix.to_array()

    def all_segment_homogeneous_matrices_in_global(self, q: np.ndarray) -> np.ndarray:
        """
        Returns a Nsegx4x4 array containing the roto-translation matrix of each segment in the global reference frame.
        """
        return np.array(
            [
                self.segment_homogeneous_matrices_in_global(q, i)
                for i in range(self.model.nbSegment())
            ]
        )

    def all_frame_homogeneous_matrices(self, q: np.ndarray) -> np.ndarray:
        """
        Returns a NframesxNsegx4x4 array containing the roto-translation matrix of each segment in the global reference frame
        """
        return np.array(
            [
                self.all_segment_homogeneous_matrices_in_global(q[:, i])
                for i in range(q.shape[1])
            ]
        )

    def markers(self, q: np.ndarray) -> np.ndarray:
        """
        Returns a Nmarkersx3 array containing the position of each marker in the global reference frame
        """
        return np.array(
            [
                self.model.markers(GeneralizedCoordinates(q))[i].to_array()
                for i in range(self.model.nbMarkers())
            ]
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


class RerunBiorbd:
    def __init__(self, biomod: BiorbdModel) -> None:
        self.model = biomod
        self.homogenous_matrices = None
        self.model_markers = None
        self.tspan = None
        self.model_marker_color = np.ones((self.model.nb_markers, 3))

    def set_marker_color(self, color: np.ndarray) -> None:
        self.model_marker_color = np.tile(color, (self.model.nb_markers, 1))

    def set_q(self, q: np.ndarray) -> None:
        self.homogenous_matrices = self.model.all_frame_homogeneous_matrices(q)
        self.model_markers = self.model.all_frame_markers(q)

    def set_tspan(self, tspan: np.ndarray) -> None:
        self.tspan = tspan

    def rerun(self, name: str = "animation_id") -> None:

        rr.init(name, spawn=True)

        for i, t in enumerate(self.tspan):
            rr.set_time_seconds("stable_time", t)

            display_frame(rr)

            for j, mesh in enumerate(self.model.meshes):
                transformed_trimesh = self.model.meshes[j].apply_transform(
                    self.homogenous_matrices[i, j, :, :]
                )

                rr.log(
                    MY_STL + f"/{j}",
                    rr.Mesh3D(
                        vertex_positions=transformed_trimesh.vertices,
                        vertex_normals=transformed_trimesh.vertex_normals,
                        indices=transformed_trimesh.faces,
                    ),
                )

            # put first frame in shape (n_mark, 3)
            # positions_f = self.model_markers[i, :, :3]
            #
            # rr.log(
            #     MY_STL + "/my_markers",
            #     rr.Points3D(
            #         positions_f,
            #         # colors=self.model_marker_color,
            #         radii=100,
            #         # labels=self.model.marker_names,
            #     ),
            # )


def display_frame(rr):
    """Display the world reference frame"""
    rr.log(
        MY_STL + "/X",
        rr.Arrows3D(
            origins=np.zeros(3),
            vectors=np.array([1, 0, 0]),
            colors=np.array([0, 0, 1]),
        ),
    )
    rr.log(
        MY_STL + "/Y",
        rr.Arrows3D(
            origins=np.zeros(3),
            vectors=np.array([0, 1, 0]),
            colors=np.array([0, 1, 0]),
        ),
    )
    rr.log(
        MY_STL + "/Z",
        rr.Arrows3D(
            origins=np.zeros(3),
            vectors=np.array([0, 0, 1]),
            colors=np.array([1, 0, 0]),
        ),
    )
    return rr


def rr_biorbd(biomod: str, q: np.ndarray, tspan: np.ndarray) -> None:
    """
    Display a biorbd model in rerun.

    Parameters
    ----------
    biomod: str
        The biomod file to display.
    q: np.ndarray
        The generalized coordinates of the model.
    tspan: np.ndarray
        The time span of the animation, such as the time instant of each frame.
    """
    model = BiorbdModel(biomod)

    rerun_biorbd = RerunBiorbd(model)
    rerun_biorbd.set_q(q)
    rerun_biorbd.set_tspan(tspan)
    rerun_biorbd.rerun("animation")
