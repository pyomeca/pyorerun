import biorbd
import trimesh

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



