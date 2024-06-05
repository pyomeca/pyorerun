import numpy as np
import rerun as rr

from ..abstract.abstract_class import TimelessComponent


class Floor(TimelessComponent):

    def __init__(self, name, square_width: float, height_offset: float, subsquares: int):
        self.name = name + "/floor"
        self.vertices, self.faces = floor_mesh(
            square_width=square_width,
            height_offset=height_offset if height_offset is not None else 0,
            subsquares=subsquares if subsquares is not None else 10,
        )

    @property
    def nb_components(self):
        return 1

    def to_rerun(self) -> None:
        rr.log(
            "floor",
            rr.Mesh3D(
                vertex_positions=self.vertices,
                vertex_normals=np.tile([0.0, 0.0, 1.0], reps=(self.vertices.shape[0], 1)),
                vertex_colors=np.tile([144, 181, 198], reps=(self.vertices.shape[0], 1)),
                triangle_indices=self.faces,
            ),
        )


def floor_mesh(square_width: float, height_offset: float, subsquares: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Display a floor in rerun.

    Parameters
    ----------
    square_width: float
        The width of the floor in meters centered in zero.
    height_offset: float
        The height offset of the floor.
    subsquares: int
        The number of subsquares for each side of the floor. The total number of subsquares is subsquares^2.

    Returns
    -------
    np.ndarray
        The vertices of the floor.
    np.ndarray
        The faces of the floor.
    """
    x, y = np.meshgrid(
        np.linspace(-square_width, square_width, subsquares),
        np.linspace(-square_width, square_width, subsquares),
    )

    vertices = []
    faces = []
    base_index = 0

    for i in range(0, subsquares - 1, 1):
        starting_j = 0 if i % 2 == 0 else 1
        for j in range(starting_j, subsquares - 1, 2):
            triangle1_vertices = [
                (x[i, j], y[i, j], height_offset),
                (x[i + 1, j], y[i + 1, j], height_offset),
                (x[i, j + 1], y[i, j + 1], height_offset),
            ]
            triangle2_vertices = [
                (x[i + 1, j + 1], y[i + 1, j + 1], height_offset),
            ]

            vertices.extend(triangle1_vertices)
            vertices.extend(triangle2_vertices)

            triangle1_faces = [(base_index, base_index + 1, base_index + 2)]
            triangle2_faces = [(base_index + 3, base_index + 1, base_index + 2)]

            faces.extend(triangle1_faces)
            faces.extend(triangle2_faces)

            base_index += 4

    return np.array(vertices), np.array(faces)
