from typing import Any

import numpy as np

from .ligaments import ModelMarkerLinkUpdater
from .model_interface import BiorbdModel
from ..abstract.abstract_class import Components
from ..abstract.linestrip import LineStripProperties


class ModelMarkerLinksUpdater(Components):
    """
    It applies to markers, but it could be extended to other components, such as IMUs,
    contact point in global frame, etc.
    """

    def __init__(self, name, model: BiorbdModel):
        self.name = name
        self.model = model
        self.markers_link = self.create_markers_link_updater()

    def create_markers_link_updater(self):
        return ModelMarkerLinkUpdater(
            self.name,
            properties=LineStripProperties(
                strip_names=self.model.marker_names, color=np.array([248, 131, 121]), radius=0.001  # Coral pink
            ),
            update_callable=self.model.markers,
        )

    @property
    def nb_components(self):
        nb_components = 0
        for component in self.components:
            nb_components += component.nb_components()

    @property
    def components(self) -> list[Any]:
        return [self.markers_link]

    @property
    def component_names(self) -> list[str]:
        return [component.name for component in self.components]

    def to_rerun(self, q: np.ndarray = None, markers: np.ndarray = None) -> None:
        for component in self.components:
            component.to_rerun(q, markers)

    def to_component(self, q: np.ndarray, markers: np.ndarray) -> list:
        return [component.to_component(q, markers) for component in self.components]
