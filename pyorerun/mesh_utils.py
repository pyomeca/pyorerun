import biorbd
import numpy as np
import trimesh


class TransformableMesh:
    """
    A class to handle a trimesh object and its transformations
    and always 'apply_transform' from its initial position
    """

    def __init__(self, mesh: trimesh.Trimesh):
        self.__name = mesh.metadata["file_name"]
        self.__mesh = mesh
        self.transformed_mesh = mesh.copy()
        self.__color = np.array([0, 0, 0])

    def apply_transform(self, transform) -> trimesh.Trimesh:
        """Apply a transform to the mesh from its initial position"""
        self.transformed_mesh = self.__mesh.copy()
        self.transformed_mesh.apply_transform(transform)

        return self.transformed_mesh

    @property
    def mesh(self):
        return self.__mesh

    @property
    def name(self):
        return self.__name


def load_biorbd_meshes(biomod: biorbd.Model) -> list[TransformableMesh]:
    """
    Load all the meshes from a biorbd model

    todo: add mesh color, scaling and location from the biomod file
    """
    meshes = []

    for segment in biomod.segments():
        if segment.characteristics().mesh().hasMesh():
            stl_file_path = segment.characteristics().mesh().path().absolutePath().to_string()
            mesh = trimesh.load(stl_file_path, file_type="stl")
            meshes.append(TransformableMesh(mesh))

    return meshes
