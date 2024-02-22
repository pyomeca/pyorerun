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
        )


def display_meshes(animation_id, meshes, homogenous_matrices) -> None:
    """Display the meshes"""
    for j, mesh in enumerate(meshes):
        transformed_trimesh = meshes[j].apply_transform(homogenous_matrices[j, :, :])

        rr.log(
            animation_id + f"/{j}",
            rr.Mesh3D(
                vertex_positions=transformed_trimesh.vertices,
                vertex_normals=transformed_trimesh.vertex_normals,
                indices=transformed_trimesh.faces,
            ),
        )


def display_markers(animation_id, name, positions, colors, radii, labels=None) -> None:
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
    colors: np.ndarray
        The markers colors [n_markers x 3]
    radii: np.ndarray
        The markers radii [n_markers]

    """
    if labels is not None:
        labels = [label.encode("utf-8") for label in labels]

    rr.log(
        animation_id + f"/{name}_markers",
        rr.Points3D(
            positions=positions,
            colors=colors,
            radii=radii,
            labels=labels,
        ),
    )
