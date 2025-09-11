import os
import numpy as np
import rerun as rr
from trimesh import Trimesh, load

from ..abstract.abstract_class import Component
from ..utils.vtp_parser import read_vtp_file


class TransformableMeshUpdater(Component):
    """
    A class to handle a trimesh object and its transformations
    and always 'apply_transform' from its initial position
    """

    def __init__(self, name: str, mesh: Trimesh, transform_callable: callable):
        filename = (
            mesh.metadata["file_name"] if "file_name" in mesh.metadata else mesh.metadata["header"].replace(" ", "")
        )
        self.__name = name + "/" + filename.split(os.sep)[-1]
        self.__mesh = mesh

        self.transformed_mesh = mesh.copy()
        self.__color = np.array([0, 0, 0])
        self.__transparency = False
        self.transform_callable = transform_callable
        self.__rerun_mesh = None

    def set_transparency(self, transparency: bool) -> None:
        self.__transparency = transparency

    def set_color(self, color: tuple[int, int, int]) -> None:
        self.__color = np.array(color)
        self._set_rerun_mesh3d()

    def _set_rerun_mesh3d(self):
        transformed_trimesh = self.apply_transform(np.eye(4))
        if self.__transparency:

            # Create a list of line strips from the faces the fourth vertex is the first one to close the loop.
            # Each triangle is a substrip
            strips = [
                [self.__mesh.vertices[element] for element in [face[0], face[1], face[2], face[0]]]
                for face in self.__mesh.faces
            ]

            self.__rerun_mesh = rr.LineStrips3D(
                strips=strips,
                colors=[self.__color for _ in range(len(strips))],
                radii=[0.0002 for _ in range(len(strips))],
            )
        else:
            self.__rerun_mesh = rr.Mesh3D(
                vertex_positions=self.__mesh.vertices,
                vertex_normals=transformed_trimesh.vertex_normals,
                vertex_colors=np.tile(self.__color, (self.__mesh.vertices.shape[0], 1)),
                triangle_indices=self.__mesh.faces,
            )

    @classmethod
    def from_file(
        cls, name, file_path: str, transform_callable, scale_factor: list[float] = (1, 1, 1)
    ) -> "TransformableMeshUpdater":
        if file_path.endswith(".stl") or file_path.endswith(".STL"):
            mesh = load(file_path, file_type="stl")
            mesh.apply_scale(scale_factor)
            mesh.metadata["file_name"] = file_path
            return cls(name, mesh, transform_callable)
        elif file_path.endswith(".vtp"):
            output = read_vtp_file(file_path)
            is_not_a_trimesh = output["polygons"].shape[1] > 3
            if is_not_a_trimesh:
                raise ValueError(
                    f"The file {file_path} is not a triangular-only mesh. It has polygons with more than 3 vertices."
                )
            mesh = Trimesh(
                vertices=output["nodes"],
                faces=output["polygons"],
                vertex_normals=output["normals"],
                metadata={"file_name": file_path.split("/")[-1].split(".")[0]},
            )
            mesh.apply_scale(scale_factor)
            return cls(name, mesh, transform_callable)
        else:
            raise ValueError(f"The file {file_path} is not a valid mesh file. It should be either .stl or .vtp.")

    def apply_transform(self, homogenous_matrix: np.ndarray) -> Trimesh:
        """Apply a transform to the mesh from its initial position"""
        self.transformed_mesh = self.__mesh.copy()
        self.transformed_mesh.apply_transform(homogenous_matrix)
        return self.transformed_mesh

    @property
    def mesh(self):
        return self.__mesh

    @property
    def rerun_mesh(self) -> rr.Mesh3D:
        return self.__rerun_mesh

    @property
    def name(self):
        return self.__name

    @property
    def nb_components(self):
        return 1

    def initialize(self):
        rr.log(
            self.name,
            self.rerun_mesh,
        )

    def to_rerun(self, q: np.ndarray) -> None:
        homogenous_matrices = self.transform_callable(q)
        rr.log(
            self.name,
            rr.Transform3D(
                translation=homogenous_matrices[:3, 3],
                mat3x3=homogenous_matrices[:3, :3],
            ),
        )

    def to_component(self, q: np.ndarray) -> rr.Transform3D:
        homogenous_matrices = self.transform_callable(q)
        return self.to_component_from_homogenous_mat(homogenous_matrices)

    @staticmethod
    def to_component_from_homogenous_mat(mat: np.ndarray) -> rr.Transform3D:
        return rr.Transform3D(
            translation=mat[:3, 3],
            mat3x3=mat[:3, :3],
        )

    @property
    def component_names(self):
        return [self.name]

    def compute_all_transforms(self, q: np.ndarray) -> np.ndarray:
        nb_frames = q.shape[1]
        homogenous_matrices = np.zeros((4, 4, nb_frames))
        for f in range(nb_frames):
            homogenous_matrices[:, :, f] = self.transform_callable(q[:, f])

        return homogenous_matrices

    def to_chunk(self, q: np.ndarray) -> dict[str, list]:
        homogenous_matrices = self.compute_all_transforms(q)

        return {
            self.name: [
                rr.InstancePoses3D.indicator(),
                rr.components.PoseTranslation3DBatch(homogenous_matrices[:3, 3, :].T),
                rr.components.PoseTransformMat3x3Batch(
                    [homogenous_matrices[:3, :3, f] for f in range(homogenous_matrices.shape[2])]
                ),
            ]
        }
