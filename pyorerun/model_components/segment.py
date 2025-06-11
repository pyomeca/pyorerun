import numpy as np

from pyorerun.abstract.abstract_class import Component
from .local_frame import LocalFrameUpdater
from .mesh import TransformableMeshUpdater


class SegmentUpdater(Component):
    def __init__(self, name, transform_callable: callable, meshes: list[TransformableMeshUpdater]):
        self.name = name
        self.transform_callable = transform_callable
        self.meshes = meshes
        self.local_frame = LocalFrameUpdater(name + "/frame", transform_callable)

    @property
    def nb_components(self):
        nb_components = 0
        for component in self.components:
            nb_components += component.nb_components()

    @property
    def components(self) -> list[Component]:
        return [*self.meshes, self.local_frame]

    def to_rerun(self, q: np.ndarray) -> None:
        for component in self.components:
            component.to_rerun(q)

    @property
    def component_names(self) -> list[str]:
        return [component.name for component in self.components]

    def initialize(self):
        self.local_frame.initialize()
        [mesh.initialize() for mesh in self.meshes]

    def to_chunk(self, q: np.ndarray) -> dict[str, list]:
        return {component.name: component.to_chunk(q) for component in self.components}
