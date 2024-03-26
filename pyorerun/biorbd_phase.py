import numpy as np

from .biorbd_components.model_interface import BiorbdModel
from .biorbd_components.model_marker_link_updapter import ModelMarkerLinksUpdater
from .biorbd_components.model_updapter import ModelUpdater


class BiorbdRerunPhase:
    """
    A class to animate a biorbd model in rerun.
    """

    def __init__(self, name, phase: int = 0):
        self.name = name
        self.phase = phase
        self.models = []
        self.rerun_models = []
        self.q = []
        self.tracked_markers = []
        self.rerun_links = []

    @property
    def _rerun_links_without_none(self):
        return [rr_link for rr_link in self.rerun_links if rr_link is not None]

    def add_animated_model(self, biomod: BiorbdModel, q: np.ndarray, tracked_markers: np.ndarray = None):
        self.models.append(biomod)
        self.rerun_models.append(ModelUpdater(name=f"{self.name}/{self.nb_models}_{biomod.name}", model=biomod))
        self.q.append(q)

        self.tracked_markers.append(tracked_markers if tracked_markers is not None else None)
        updater = (
            ModelMarkerLinksUpdater(name=f"{self.name}/{self.nb_models}_{biomod.name}", model=biomod)
            if tracked_markers is not None
            else None
        )
        self.rerun_links.append(updater)

    def to_rerun(self, frame: int):
        for i, rr_model in enumerate(self.rerun_models):
            rr_model.to_rerun(
                self.q[i][:, frame],
            )

        for i, rr_link in enumerate(self._rerun_links_without_none):
            rr_link.to_rerun(self.q[i][:, frame], self.tracked_markers[i][:, :, frame])

    @property
    def nb_models(self):
        return len(self.models)

    @property
    def component_names(self):
        all_component_names = []
        for model in self.rerun_models:
            all_component_names.extend(model.component_names)
        return all_component_names
