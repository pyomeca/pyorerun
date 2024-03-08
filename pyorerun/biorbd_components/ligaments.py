import numpy as np
import rerun as rr

from ..abstract.linestrip import LineStrips, LineStripProperties


class LineStripUpdater(LineStrips):
    def __init__(self, name, properties: LineStripProperties, update_callable: callable):
        self.name = name
        self.properties = properties
        self.update_callable = update_callable

    @property
    def nb_strips(self) -> int:
        return self.properties.nb_strips

    @property
    def nb_components(self) -> int:
        return 1

    def to_rerun(self, q: np.ndarray) -> None:
        rr.log(
            self.name,
            rr.LineStrips3D(
                strips=self.update_callable(q),
                radii=self.properties.radius_to_rerun(),
                colors=self.properties.color_to_rerun(),
                labels=self.properties.strip_names,
            ),
        )


class LigamentsUpdater(LineStripUpdater):
    def __init__(self, name, properties: LineStripProperties, update_callable: callable):
        super(LigamentsUpdater, self).__init__(
            name=name + "/ligaments", properties=properties, update_callable=update_callable
        )


class MusclesUpdater(LineStripUpdater):
    def __init__(self, name, properties: LineStripProperties, update_callable: callable):
        super(MusclesUpdater, self).__init__(
            name=name + "/muscles", properties=properties, update_callable=update_callable
        )
