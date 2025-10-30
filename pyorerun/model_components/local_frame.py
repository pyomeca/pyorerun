import numpy as np
import rerun as rr

from ..abstract.abstract_class import Component


class LocalFrameUpdater(Component):
    def __init__(self, name, transform_callable: callable):
        """

        ----------
        name : str
            The name of the axis
        transform_callable : callable
            The function to transform the axis
        """
        self.name = name
        self.transform_callable = transform_callable
        self.scale = 0.1

    @property
    def nb_components(self):
        return 1

    def initialize(self):
        rr.log(
            self.name,
            rr.Transform3D(
                translation=np.zeros(3),
                mat3x3=np.eye(3),
            ),
        )

    def to_rerun(self, q: np.ndarray) -> None:
        rr.log(
            self.name,
            self.to_component(q),
        )

    def to_component(self, q: np.ndarray) -> rr.Transform3D:
        homogenous_matrices = self.transform_callable(q)
        return rr.Transform3D(
            translation=homogenous_matrices[:3, 3],
            mat3x3=homogenous_matrices[:3, :3],
            axis_length=self.scale,
        )

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
                *rr.Transform3D.columns(
                    translation=homogenous_matrices[:3, 3, :].T.tolist(),
                    mat3x3=[homogenous_matrices[:3, :3, f] for f in range(homogenous_matrices.shape[2])],
                    scale=[[self.scale] * 3] * homogenous_matrices.shape[2],
                )
            ]
        }
