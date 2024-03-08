import numpy as np

from pyorerun.abstract.abstract_class import Components
from .axisupdater import AxisUpdater


class LocalFrameUpdater(Components):
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
        self.x_axis = AxisUpdater(name + "/X", transform_callable, 0)
        self.y_axis = AxisUpdater(name + "/Y", transform_callable, 1)
        self.z_axis = AxisUpdater(name + "/Z", transform_callable, 2)

    @property
    def components(self):
        return [self.x_axis, self.y_axis, self.z_axis]

    @property
    def nb_components(self):
        return len(self.components)

    def to_rerun(self, q: np.ndarray) -> None:
        for component in self.components:
            component.to_rerun(q)

    @property
    def component_names(self) -> list[str]:
        return [component.name for component in self.components]
