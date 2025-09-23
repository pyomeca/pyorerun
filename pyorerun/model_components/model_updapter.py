from functools import partial
from typing import Any

import numpy as np

from .mesh import TransformableMeshUpdater
from .model_display_options import DisplayModelOptions
from .model_markers import MarkersUpdater, PersistentMarkersUpdater
from .segment import SegmentUpdater
from ..abstract.abstract_class import Components
from ..abstract.empty_updater import EmptyUpdater
from ..abstract.linestrip import LineStripProperties
from ..abstract.markers import MarkerProperties
from ..model_components.ligaments import LigamentsUpdater, MusclesUpdater, LineStripUpdaterFromGlobalTransform
from ..model_interfaces import AbstractModel, model_from_file


class ModelUpdater(Components):
    def __init__(
        self,
        name,
        model: AbstractModel,
        muscle_colors: np.ndarray = None,
    ):
        self.name = name
        self.model = model

        # Time dependant components
        self.markers = self.create_markers_updater()
        self.centers_of_mass = self.create_centers_of_mass_updater()
        self.soft_contacts = self.create_soft_contacts_updater()
        self.rigid_contacts = self.create_rigid_contacts_updater()
        self.ligaments = self.create_ligaments_updater()
        self.segments = self.create_segments_updater()
        self.muscles = self.create_muscles_updater(muscle_colors)

        # Persistent components
        self.persistent_markers = self.create_persistent_markers_updater()

    @classmethod
    def from_file(cls, model_path: str, options: DisplayModelOptions = None):
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
        model, no_mesh_instance = model_from_file(model_path, options=options)

        if model.has_mesh or model.has_meshlines:
            return cls(model.name, model)

        return cls(model.name, no_mesh_instance(model_path, options=options))

    def create_markers_updater(self):
        if self.model.nb_markers == 0:
            return EmptyUpdater(self.name + "/markers")
        return MarkersUpdater(
            self.name,
            marker_properties=MarkerProperties(
                markers_names=self.model.marker_names,
                color=np.array(self.model.options.markers_color),
                radius=self.model.options.markers_radius,
                show_labels=self.model.options.show_marker_labels,
            ),
            callable_markers=self.model.markers,
        )

    def create_centers_of_mass_updater(self):
        has_mass = self.model.segment_names_with_mass != tuple([])
        if has_mass:
            return MarkersUpdater(
                self.name + "/centers_of_mass",
                marker_properties=MarkerProperties(
                    markers_names=self.model.segment_names_with_mass,
                    color=np.array(self.model.options.centers_of_mass_color),
                    radius=self.model.options.centers_of_mass_radius,
                    show_labels=self.model.options.show_center_of_mass_labels,
                ),
                callable_markers=self.model.centers_of_mass,
            )
        else:
            return EmptyUpdater(self.name + "/centers_of_mass")

    def create_soft_contacts_updater(self):
        if not self.model.has_soft_contacts:
            return EmptyUpdater(self.name + "/soft_contacts")
        return MarkersUpdater(
            self.name + "/soft_contacts",
            marker_properties=MarkerProperties(
                markers_names=self.model.soft_contacts_names,
                color=np.array(self.model.options.soft_contacts_color),
                radius=self.model.soft_contact_radii,
                show_labels=self.model.options.show_contact_labels,
            ),
            callable_markers=self.model.soft_contacts,
        )

    def create_rigid_contacts_updater(self):
        if not self.model.has_rigid_contacts:
            return EmptyUpdater(self.name + "/rigid_contacts")
        return MarkersUpdater(
            self.name + "/rigid_contacts",
            marker_properties=MarkerProperties(
                markers_names=self.model.rigid_contacts_names,
                color=np.array(self.model.options.rigid_contacts_color),
                radius=0.01,
                show_labels=self.model.options.show_contact_labels,
            ),
            callable_markers=self.model.rigid_contacts,
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
                show_labels=self.model.options.show_ligament_labels,
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
                meshes = []
                for m_idx, m in enumerate(segment.mesh_path):
                    mesh_transform_callable = partial(
                        self.model.mesh_homogenous_matrices_in_global, segment_index=segment.id, mesh_index=m_idx
                    )
                    meshes.append(
                        TransformableMeshUpdater.from_file(
                            segment_name, m, mesh_transform_callable, segment.mesh_scale_factor[m_idx]
                        )
                    )
                    meshes[-1].set_transparency(self.model.options.transparent_mesh)
                    meshes[-1].set_color(self.model.options.mesh_color)

            elif segment.has_meshlines:
                meshes = [
                    LineStripUpdaterFromGlobalTransform(
                        segment_name + "/meshlines",
                        properties=LineStripProperties(
                            strip_names=self.model.muscle_names,
                            color=np.array((0, 0, 0)),
                            radius=0.001,
                        ),
                        strips=self.model.meshlines[i],
                        transform_callable=transform_callable,
                    )
                ]
            else:
                meshes = [EmptyUpdater(segment_name + "/mesh")]

            segments.append(SegmentUpdater(name=segment_name, transform_callable=transform_callable, meshes=meshes))
        return segments

    def create_muscles_updater(self, muscle_colors: np.ndarray = None):
        if self.model.nb_muscles == 0:
            return EmptyUpdater(self.name + "/muscles")
        colors = muscle_colors if muscle_colors is not None else np.array(self.model.options.muscles_color)
        return MusclesUpdater(
            self.name,
            properties=LineStripProperties(
                strip_names=self.model.muscle_names,
                color=colors,
                radius=self.model.options.muscles_radius,
                show_labels=self.model.options.show_muscle_labels,
            ),
            update_callable=self.model.muscle_strips,
        )

    def create_persistent_markers_updater(self):
        if self.model.nb_markers == 0 or self.model.options.persistent_markers is None:
            return EmptyUpdater(self.name + "/persistent_marker")
        else:
            return PersistentMarkersUpdater(
                self.name,
                marker_properties=MarkerProperties(
                    markers_names=self.model.marker_names,
                    color=np.array(self.model.options.markers_color),
                    radius=self.model.options.markers_radius,
                    show_labels=self.model.options.show_marker_labels,
                ),
                persistent_markers=self.model.options.persistent_markers,
                callable_markers=self.model.markers,
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
            self.rigid_contacts,
            *all_segment_components,
            self.ligaments,
            self.muscles,
        ]

    @property
    def persistent_components(self) -> list[Any]:
        return [self.persistent_markers]

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
            [mesh.initialize() for mesh in segment.meshes]

        for component in self.components:
            component.to_rerun(q)

    def to_rerun_persistent(self, q: np.ndarray) -> None:
        """
        This function logs the components to rerun.

        Parameters
        ----------
        q: np.ndarray
            The generalized coordinates of the model two-dimensional array, i.e., q.shape = (n_q, N_frames).
        """
        for persistent_component in self.persistent_components:
            persistent_component.to_rerun(q)

    def to_component(self, q: np.ndarray) -> list:
        components = []
        for component in self.components:
            components += [component.to_component(q)]
        for persistent_component in self.persistent_components:
            components += [persistent_component.to_component(q)]
        return components

    def initialize(self):
        for segment in self.segments:
            segment.initialize()

    def to_chunk(self, q: np.ndarray) -> dict[str, list]:
        output = {}
        for component in self.components:
            output.update(component.to_chunk(q))
        for persistent_component in self.persistent_components:
            output.update(persistent_component.to_chunk(q))
        # remove all empty components, this is the "empty" field
        output.pop("empty")
        return output
