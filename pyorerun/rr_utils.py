from typing import Callable

import numpy as np
import rerun as rr


def display_frame(animation_id) -> None:
    """Display the world reference frame"""
    for axis, color in zip(["X", "Y", "Z"], [[255, 0, 0], [0, 255, 0], [0, 0, 255]]):
        rr.log(
            f"{animation_id}/{axis}",
            rr.Arrows3D(
                origins=np.zeros(3),
                vectors=np.eye(3)[list("XYZ").index(axis)],
                colors=np.array(color),
            ),
            timeless=True,
        )


def display_meshes(animation_id, meshes, homogenous_matrices) -> None:
    """Display the meshes"""
    for j, mesh in enumerate(meshes):
        transformed_trimesh = mesh.apply_transform(homogenous_matrices[j, :, :])

        rr.log(
            animation_id + f"/{mesh.name}_{j}",
            rr.Mesh3D(
                vertex_positions=transformed_trimesh.vertices,
                vertex_normals=transformed_trimesh.vertex_normals,
                indices=transformed_trimesh.faces,
            ),
        )


def display_markers(
    animation_id, name: str, positions: np.ndarray, point3d: Callable[[np.ndarray], rr.Points3D]
) -> None:
    """
    Display the markers

    Parameters
    ----------
    animation_id: str
        The animation id
    name: str
        The name of the markers
    positions: np.ndarray
        The markers positions [n_markers x 3]
    point3d: Callable[[np.ndarray], rr.Points3D]
        The function to create the markers partially filled with functools.partial
    """

    rr.log(
        animation_id + f"/{name}_markers",
        point3d(positions=positions),
    )
