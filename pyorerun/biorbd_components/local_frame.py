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
        axis : int
            The axis to display (0, 1, 2) for (X, Y, Z)
        """
        self.name = name
        self.transform_callable = transform_callable
        self.scale = 0.1  # NOTE: This is a hard-coded value, and not transferred to the x,y,z axis objects

    @property
    def nb_components(self):
        return 1

    def to_rerun(self, q: np.ndarray) -> None:
        homogenous_matrices = self.transform_callable(q)
        rr.log(
            self.name,
            rr.Transform3D(
                # scale=self.scale,
                translation=homogenous_matrices[:3, 3],
                mat3x3=homogenous_matrices[:3, :3],
            ),
        )
