from functools import partial
from typing import Any

import numpy as np

from .mesh import TransformableMeshUpdater
from .model_interface import BiorbdModel
from .model_markers import MarkersUpdater
from .segment import SegmentUpdater
from ..abstract.abstract_class import Components
from ..abstract.empty_updater import EmptyUpdater
from ..abstract.linestrip import LineStripProperties
from ..abstract.markers import MarkerProperties
from ..biorbd_components.ligaments import LigamentsUpdater, MusclesUpdater


class ModelUpdater(Components):
    def __init__(self, name, model: BiorbdModel):
        self.name = name
        self.model = model
        self.markers = self.create_markers_updater()
        self.ligaments = self.create_ligaments_updater()
        self.segments = self.create_segments_updater()
        self.muscles = self.create_muscles_updater()

    def create_markers_updater(self):
        if self.model.nb_markers == 0:
            return EmptyUpdater()
        return MarkersUpdater(
            self.name,
            marker_properties=MarkerProperties(
                markers_names=self.model.marker_names, color=np.array([0, 0, 255]), radius=0.01
            ),
            callable_markers=self.model.markers,
        )

    def create_ligaments_updater(self):
        if self.model.nb_ligaments == 0:
            return EmptyUpdater()

        return LigamentsUpdater(
            self.name,
            properties=LineStripProperties(
                strip_names=self.model.ligament_names,
                color=np.array([255, 255, 0]),
                radius=0.01,
            ),
            update_callable=self.model.ligament_strips,
        )

    def create_segments_updater(self):
        segments = []

        for i, segment in enumerate(self.model.segments):
            segment_name = self.name + "/" + segment.name
            transform_callable = partial(
                self.model.segment_homogeneous_matrices_in_global,
                segment_index=segment.id,
            )

            mesh = (
                TransformableMeshUpdater.from_file(segment_name, segment.mesh_path, transform_callable)
                if segment.has_mesh
                else EmptyUpdater()
            )
            segments.append(SegmentUpdater(name=segment_name, transform_callable=transform_callable, mesh=mesh))
        return segments

    def create_muscles_updater(self):
        if self.model.nb_muscles == 0:
            return EmptyUpdater()
        return MusclesUpdater(
            self.name,
            properties=LineStripProperties(
                strip_names=self.model.muscle_names,
                color=np.array([255, 0, 0]),
                radius=0.004,
            ),
            update_callable=self.model.muscle_strips,
        )

    @property
    def nb_components(self):
        nb_components = 0
        for component in self.components:
            nb_components += component.nb_components()

    @property
    def components(self) -> list[Any]:
        all_segment_components = []
        for segment in self.segments:
            all_segment_components.extend(segment.components)
        return [self.markers, *all_segment_components, self.ligaments, self.muscles]

    @property
    def component_names(self) -> list[str]:
        return [component.name for component in self.components]

    def to_rerun(self, q: np.ndarray) -> None:
        for component in self.components:
            component.to_rerun(q)
