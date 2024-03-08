from functools import partial

import numpy as np

from .mesh import TransformableMesh
from .model_interface import BiorbdModel
from .model_markers import BiorbdModelMarkers
from .segment import BiorbdModelSegment
from ..abstract.abstract_class import Components
from ..abstract.linestrip import LineStripProperties
from ..abstract.markers import MarkerProperties
from ..biorbd_components.ligaments import BiorbdModelLigaments


class BiorbdModelSegments(Components):
    def __init__(self, name, model: BiorbdModel):
        self.name = name
        self.model = model
        self.markers = self.create_markers()
        self.ligaments = self.create_ligaments()
        self.segments = self.create_segments()

    def create_markers(self):
        return BiorbdModelMarkers(
            self.name,
            marker_properties=MarkerProperties(
                markers_names=self.model.marker_names, color=np.array([0, 0, 255]), radius=0.01
            ),
            callable_markers=self.model.markers,
        )

    def create_ligaments(self):
        return BiorbdModelLigaments(
            self.name,
            ligament_properties=LineStripProperties(
                strip_names=self.model.ligament_names,
                color=np.array([255, 255, 0]),
                radius=0.01,
            ),
            callable_ligaments=self.model.ligament_strips,
        )

    def create_segments(self):
        segments = []
        for i, (segment, mesh_path) in enumerate(zip(self.model.segments_with_mesh, self.model.mesh_paths)):
            segment_name = self.name + "/" + segment.name
            # transform_callable = partial(self.model.segment_homogeneous_matrices_in_global, segment_index=segment.id())
            transform_callable = partial(
                self.model.segment_homogeneous_matrices_in_global,
                segment_index=segment.id,
            )

            mesh = TransformableMesh.from_file(segment_name, mesh_path, transform_callable)
            segments.append(BiorbdModelSegment(name=segment_name, transform_callable=transform_callable, mesh=mesh))
        return segments

    @property
    def nb_components(self):
        nb_components = 0
        for component in self.components:
            nb_components += component.nb_components()

    @property
    def components(self) -> list[BiorbdModelSegment]:
        all_segment_components = []
        for segment in self.segments:
            all_segment_components.extend(segment.components)
        return [self.markers, *all_segment_components, self.ligaments]

    @property
    def component_names(self) -> list[str]:
        return [component.name for component in self.components]

    def to_rerun(self, q: np.ndarray) -> None:
        for component in self.components:
            component.to_rerun(q)
