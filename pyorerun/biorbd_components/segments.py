from functools import partial

import numpy as np

from .mesh import TransformableMesh
from .model_interface import BiorbdModel
from .model_markers import BiorbdModelMarkers
from .segment import BiorbdModelSegment
from ..abstract.abstract_class import Components
from ..abstract.markers import MarkerProperties


class BiorbdModelSegments(Components):
    def __init__(self, name, model: BiorbdModel):
        self.name = name
        self.model = model
        self.markers = BiorbdModelMarkers(
            name,
            marker_properties=MarkerProperties(
                markers_names=model.marker_names, color=np.array([0, 0, 255]), radius=0.01
            ),
            callable_markers=model.markers,
        )
        self.segments = []
        for i, (segment, mesh_path) in enumerate(zip(model.segments_with_mesh, model.mesh_paths)):
            segment_name = name + "/" + segment.name().to_string()
            transform_callable = partial(model.segment_homogeneous_matrices_in_global, segment_index=segment.id())
            mesh = TransformableMesh.from_file(segment_name, mesh_path, transform_callable)
            self.segments.append(
                BiorbdModelSegment(name=segment_name, transform_callable=transform_callable, mesh=mesh)
            )

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
        return [self.markers, *all_segment_components]

    @property
    def component_names(self) -> list[str]:
        return [component.name for component in self.components]

    def to_rerun(self, q: np.ndarray) -> None:
        for component in self.components:
            component.to_rerun(q)