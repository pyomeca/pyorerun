import numpy as np
import rerun as rr

from pyorerun.abstract.abstract_class import Component


class AxisUpdater(Component):
    def __init__(self, name, transform_callable: callable, axis: int):
        """

        ----------
        name : str
            The name of the axis
        transform_callable : callable
            The function to transform the axis in the world reference frame from generalized coordinates q
        axis : int
            The axis to display (0, 1, 2) for (X, Y, Z)
        """
        self.name = name
        self.transform_callable = transform_callable
        self.axis = axis
        self.scale = 0.1

    def nb_components(self):
        return 1

    @property
    def color(self):
        if self.axis == 0:
            return [255, 0, 0]
        if self.axis == 1:
            return [0, 255, 0]
        if self.axis == 2:
            return [0, 0, 255]

    def to_rerun(self, q: np.ndarray) -> None:
        homogenous_matrices = self.transform_callable(q)
        rr.log(
            self.name,
            rr.Arrows3D(
                origins=homogenous_matrices[:3, 3],
                vectors=homogenous_matrices[:3, self.axis] * self.scale,
                colors=np.array(self.color),
            ),
        )
