from functools import partial

import numpy as np

from .abstract_class import Components
from .mesh import TransformableMesh
from .model_interface import BiorbdModel
from .model_markers import BiorbdModelMarkers
from .segment import BiorbdModelSegment


class BiorbdModelSegments(Components):
    def __init__(self, name, model: BiorbdModel):
        self.name = name
        self.model = model
        self.markers = BiorbdModelMarkers(name, model.marker_names, callable=model.markers, color=None, radius=None)

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

    def to_rerun(self, q: np.ndarray) -> None:
        for component in self.components:
            component.to_rerun(q)
