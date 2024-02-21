import numpy as np
import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!

from .biorbd_interface import BiorbdModel
from .rr_utils import display_frame

MY_STL = "my_stl"


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


class RerunBiorbd:
    """
    A class to animate a biorbd model in rerun.
    """

    def __init__(self, biomod: BiorbdModel) -> None:
        self.model = biomod
        self.homogenous_matrices = None
        self.model_markers = None
        self.tspan = None
        self.__model_markers_color = np.array([0, 0, 255])
        self.__model_markers_size = 0.01

    def set_marker_color(self, color: np.ndarray) -> None:
        self.__model_markers_color = color

    def set_marker_size(self, size: float) -> None:
        self.__model_markers_size = size

    def set_q(self, q: np.ndarray) -> None:
        self.homogenous_matrices = self.model.all_frame_homogeneous_matrices(q)
        self.model_markers = self.model.all_frame_markers(q)

    def set_tspan(self, tspan: np.ndarray) -> None:
        self.tspan = tspan

    def rerun(self, name: str = "animation_id") -> None:

        rr.init(name, spawn=True)

        for i, t in enumerate(self.tspan):
            rr.set_time_seconds("stable_time", t)

            display_frame(rr, MY_STL)

            for j, mesh in enumerate(self.model.meshes):
                transformed_trimesh = self.model.meshes[j].apply_transform(self.homogenous_matrices[i, j, :, :])

                rr.log(
                    MY_STL + f"/{j}",
                    rr.Mesh3D(
                        vertex_positions=transformed_trimesh.vertices,
                        vertex_normals=transformed_trimesh.vertex_normals,
                        indices=transformed_trimesh.faces,
                    ),
                )

            # put first frame in shape (n_mark, 3)
            positions_f = self.model_markers[i, :, :3]

            labels = [f"marker:_{i}" for i in range(self.model.nb_markers)]
            labels = [label.encode("utf-8") for label in labels]

            rr.log(
                MY_STL + "/my_markers",
                rr.Points3D(
                    positions_f,
                    colors=np.tile(self.__model_markers_color, (self.model.nb_markers, 1)),
                    radii=np.ones(self.model.nb_markers) * self.__model_markers_size,
                    # NOTE: not sure how to register the labels but I don't want to display then in the viewer.
                    # labels=labels,
                ),
            )
