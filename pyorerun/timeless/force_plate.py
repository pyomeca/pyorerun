import numpy as np
import rerun as rr

from ..abstract.abstract_class import TimelessComponent


class ForcePlate(TimelessComponent):

    def __init__(self, name, num: int, corners: np.ndarray):
        self.entity = f"force_plate{num}"
        self.name = name + f"/{self.entity}"
        self.vertices, self.faces = rectangle_mesh_from_corners(corners)

    @property
    def nb_components(self):
        return 1

    def to_rerun(self) -> None:
        rr.log(
            self.entity,
            rr.Mesh3D(
                vertex_positions=self.vertices,
                vertex_normals=np.tile([0.0, 0.0, 1.0], reps=(self.vertices.shape[0], 1)),
                vertex_colors=np.tile([44, 115, 148], reps=(self.vertices.shape[0], 1)),
                indices=self.faces,
            ),
        )


def rectangle_mesh_from_corners(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a rectangle mesh from the corners.

    Parameters
    ----------
    corners: np.ndarray
        The corners of the rectangle.

    Returns
    -------
    np.ndarray
        The vertices of the floor.
    np.ndarray
        The faces of the floor.
    """

    vertices = corners.T

    triangle1_faces = [0, 1, 2]
    triangle2_faces = [2, 3, 0]

    faces = []
    faces.extend(triangle1_faces)
    faces.extend(triangle2_faces)

    return vertices, np.array(faces)


def rectangle_mesh(length: float, width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a rectangle mesh centered in zero.

    Parameters
    ----------
    length: float
        The length of the rectangle in meters (x-axis).
    width: float
        The width of the rectangle in meters (y-axis).
    height: float
        The height of the rectangle in meters (z-axis).

    Returns
    -------
    np.ndarray
        The vertices of the floor.
    np.ndarray
        The faces of the floor.
    """
    x, y = np.meshgrid(
        np.linspace(-length / 2, length / 2, 2),
        np.linspace(-width / 2, width / 2, 2),
    )

    vertices = []
    faces = []
    base_index = 0

    for i in range(0, 1, 1):
        for j in range(0, 1, 1):
            triangle1_vertices = [
                (x[i, j], y[i, j], height),
                (x[i + 1, j], y[i + 1, j], height),
                (x[i, j + 1], y[i, j + 1], height),
            ]
            triangle2_vertices = [
                (x[i + 1, j + 1], y[i + 1, j + 1], height),
            ]

            vertices.extend(triangle1_vertices)
            vertices.extend(triangle2_vertices)

            triangle1_faces = [(base_index, base_index + 1, base_index + 2)]
            triangle2_faces = [(base_index + 3, base_index + 1, base_index + 2)]

            faces.extend(triangle1_faces)
            faces.extend(triangle2_faces)

            base_index += 4

    return np.array(vertices), np.array(faces)
