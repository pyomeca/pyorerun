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
        self.__name = name + "/" + mesh.metadata["file_name"]
        self.__mesh = mesh

        self.transformed_mesh = mesh.copy()
        self.__color = np.array([0, 0, 0])
        self.transform_callable = transform_callable

    def set_color(self, color: tuple[int, int, int]) -> None:
        self.__color = np.array(color)
        self._set_rerun_mesh3d()

    def _set_rerun_mesh3d(self):
        transformed_trimesh = self.apply_transform(np.eye(4))
        self.__rerun_mesh = rr.Mesh3D(
            vertex_positions=self.__mesh.vertices,
            vertex_normals=transformed_trimesh.vertex_normals,
            vertex_colors=np.tile(self.__color, (self.__mesh.vertices.shape[0], 1)),
            triangle_indices=self.__mesh.faces,
        )

    @classmethod
    def from_file(cls, name, file_path: str, transform_callable) -> "TransformableMeshUpdater":
        if file_path.endswith(".stl") or file_path.endswith(".STL"):
            mesh = load(file_path, file_type="stl")
            return cls(name, mesh, transform_callable)
        if file_path.endswith(".vtp"):
            output = read_vtp_file(file_path)
            mesh = Trimesh(
                vertices=output["nodes"],
                faces=output["polygons"],
                vertex_normals=output["normals"],
                metadata={"file_name": file_path.split("/")[-1].split(".")[0]},
            )
            return cls(name, mesh, transform_callable)

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

    def to_rerun(self, q: np.ndarray) -> None:
        homogenous_matrices = self.transform_callable(q)
        rr.log(
            self.name,
            rr.Transform3D(
                translation=homogenous_matrices[:3, 3],
                mat3x3=homogenous_matrices[:3, :3],
            ),
        )
        rr.log(
            self.name,
            self.rerun_mesh,
        )

    @property
    def component_names(self):
        return [self.name]
