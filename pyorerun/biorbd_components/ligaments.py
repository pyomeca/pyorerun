import numpy as np
import rerun as rr

from ..abstract.linestrip import LineStrips, LineStripProperties


class BiorbdModelLigaments(LineStrips):
    def __init__(self, name, ligament_properties: LineStripProperties, callable_ligaments: callable):
        self.name = name + "/ligaments"
        self.ligament_properties = ligament_properties
        self.callable_ligament = callable_ligaments

    @property
    def nb_strips(self) -> int:
        return self.ligament_properties.nb_strips

    @property
    def nb_components(self) -> int:
        return 1

    def to_rerun(self, q: np.ndarray) -> None:
        rr.log(
            self.name,
            rr.LineStrips3D(
                strips=self.callable_ligament(q),
                radii=self.ligament_properties.radius_to_rerun(),
                colors=self.ligament_properties.color_to_rerun(),
                labels=self.ligament_properties.strip_names,
            ),
        )
