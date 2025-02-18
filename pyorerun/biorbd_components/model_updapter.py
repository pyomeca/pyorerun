from functools import partial
from typing import Any

import numpy as np

from .mesh import TransformableMeshUpdater
from .model_interface import BiorbdModel, BiorbdModelNoMesh
from .model_markers import MarkersUpdater
from .segment import SegmentUpdater
from ..abstract.abstract_class import Components
from ..abstract.empty_updater import EmptyUpdater
from ..abstract.linestrip import LineStripProperties
from ..abstract.markers import MarkerProperties
from ..biorbd_components.ligaments import LigamentsUpdater, MusclesUpdater, LineStripUpdaterFromGlobalTransform


class ModelUpdater(Components):
    def __init__(self, name, model: BiorbdModelNoMesh | BiorbdModel):
        self.name = name
        self.model = model
        self.markers = self.create_markers_updater()
        self.centers_of_mass = self.create_centers_of_mass_updater()
        self.soft_contacts = self.create_soft_contacts_updater()
        self.ligaments = self.create_ligaments_updater()
        self.segments = self.create_segments_updater()
        self.muscles = self.create_muscles_updater()

    @classmethod
    def from_file(cls, model_path: str):
        """
        This factory method is meant to be used with rerun easily. For example, to display a model in rerun,
        and add its custom experimental data.

        Parameters
        ----------
        model_path: str
            The path to the bioMod file, such as "path/to/model.bioMod".

        Returns
        -------
        ModelUpdater

        Examples
        --------
        >>> import rerun as rr
        >>> import numpy as np
        >>> from pyorerun import ModelUpdater

        >>> q = np.zeros(10)
        >>> model = ModelUpdater.from_file("path/to/model.bioMod")

        >>> rr.init("my_thing", spawn=True)
        >>> rr.set_time_sequence(timeline="step", sequence=0)
        >>> model.to_rerun(q)
        >>> rr.log("anything", rr.Anything())

        """
        model = BiorbdModel(model_path)
        if model.has_mesh or model.has_meshlines:
            return cls(model.name, model)

        return cls(model.name, BiorbdModelNoMesh(model_path))

    def create_markers_updater(self):
        if self.model.nb_markers == 0:
            return EmptyUpdater(self.name + "/markers")
        return MarkersUpdater(
            self.name,
            marker_properties=MarkerProperties(
                markers_names=self.model.marker_names,
                color=np.array(self.model.options.markers_color),
                radius=self.model.options.markers_radius,
            ),
            callable_markers=self.model.markers,
        )

    def create_centers_of_mass_updater(self):
        return MarkersUpdater(
            self.name + "/centers_of_mass",
            marker_properties=MarkerProperties(
                markers_names=self.model.segment_names_with_mass,
                color=np.array(self.model.options.centers_of_mass_color),
                radius=self.model.options.centers_of_mass_radius,
            ),
            callable_markers=self.model.centers_of_mass,
        )

    def create_soft_contacts_updater(self):
        if not self.model.has_soft_contacts:
            return EmptyUpdater(self.name + "/soft_contacts")
        return MarkersUpdater(
            self.name + "/soft_contacts",
            marker_properties=MarkerProperties(
                markers_names=self.model.soft_contacts_names,
                color=np.array(self.model.options.soft_contacts_color),
                radius=self.model.soft_contact_radii,
            ),
            callable_markers=self.model.soft_contacts,
        )

    def create_ligaments_updater(self):
        if self.model.nb_ligaments == 0:
            return EmptyUpdater(self.name + "/ligaments")

        return LigamentsUpdater(
            self.name,
            properties=LineStripProperties(
                strip_names=self.model.ligament_names,
                color=np.array(self.model.options.ligaments_color),
                radius=self.model.options.ligaments_radius,
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

            if segment.has_mesh:
                mesh_transform_callable = partial(
                    self.model.mesh_homogenous_matrices_in_global,
                    segment_index=segment.id,
                )
                mesh = TransformableMeshUpdater.from_file(segment_name, segment.mesh_path, mesh_transform_callable)
                mesh.set_transparency(self.model.options.transparent_mesh)
                mesh.set_color(self.model.options.mesh_color)

            elif segment.has_meshlines:
                mesh = LineStripUpdaterFromGlobalTransform(
                    segment_name + "/meshlines",
                    properties=LineStripProperties(
                        strip_names=self.model.muscle_names,
                        color=np.array((0, 0, 0)),
                        radius=0.001,
                    ),
                    strips=self.model.meshlines[i],
                    transform_callable=transform_callable,
                )
            else:
                mesh = EmptyUpdater(segment_name + "/mesh")

            segments.append(SegmentUpdater(name=segment_name, transform_callable=transform_callable, mesh=mesh))
        return segments

    def create_muscles_updater(self):
        if self.model.nb_muscles == 0:
            return EmptyUpdater(self.name + "/muscles")
        return MusclesUpdater(
            self.name,
            properties=LineStripProperties(
                strip_names=self.model.muscle_names,
                color=np.array(self.model.options.muscles_color),
                radius=self.model.options.muscles_radius,
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
        return [
            self.markers,
            self.centers_of_mass,
            self.soft_contacts,
            *all_segment_components,
            self.ligaments,
            self.muscles,
        ]

    @property
    def component_names(self) -> list[str]:
        return [component.name for component in self.components]

    def to_rerun(self, q: np.ndarray) -> None:
        """
        This function logs the components to rerun.

        Parameters
        ----------
        q: np.ndarray
            The generalized coordinates of the model one-dimensional array, i.e., q.shape = (n_q,).
        """
        for segment in self.segments:
            segment.mesh.initialize()

        for component in self.components:
            component.to_rerun(q)

    def to_component(self, q: np.ndarray) -> list:
        return [component.to_component(q) for component in self.components]

    def initialize(self):
        for segment in self.segments:
            segment.initialize()

    def to_chunk(self, q: np.ndarray) -> dict[str, list]:
        output = {}
        for component in self.components:
            output.update(component.to_chunk(q))
        # remove all empty components, this is the "empty" field
        output.pop("empty")
        return output
