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
        if segment_has_meshes(biomod, segment):
            stl_file_path = segment.characteristics().mesh().path().absolutePath().to_string()
            mesh = trimesh.load(stl_file_path, file_type="stl")
            meshes.append(TransformableMesh(mesh))

    return meshes


def segment_has_meshes(biomod: biorbd.Model, segment) -> bool:
    """
    Check if the biorbd model has meshes, by checking if the mesh path is different from the biomod path
    if it is the same, it means that the mesh is not present in the segment
    """
    full_path = biomod.path().absolutePath().to_string()
    biomod_path = full_path[: full_path.rfind("/")]
    mesh_path = segment.characteristics().mesh().path().absolutePath().to_string()[:-1]
    return biomod_path != mesh_path
