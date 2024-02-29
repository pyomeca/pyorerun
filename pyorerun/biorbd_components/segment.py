import numpy as np

from .abstract_class import Component
from .local_frame import LocalFrame
from .mesh import TransformableMesh


class BiorbdModelSegment(Component):
    def __init__(self, name, transform_callable: callable, mesh: TransformableMesh):
        self.name = name
        self.transform_callable = transform_callable
        self.mesh = mesh
        self.local_frame = LocalFrame(name + "/frame", transform_callable)

    @property
    def nb_components(self):
        nb_components = 0
        for component in self.components:
            nb_components += component.nb_components()

    @property
    def components(self) -> list[Component]:
        return [self.mesh, *self.local_frame.components]

    def to_rerun(self, q: np.ndarray) -> None:
        for component in self.components:
            component.to_rerun(q)
