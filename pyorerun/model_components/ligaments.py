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
            self.to_component(q),
        )

    def to_component(self, q: np.ndarray) -> rr.LineStrips3D:
        return rr.LineStrips3D(
            strips=self.update_callable(q),
            radii=self.properties.radius_to_rerun(),
            colors=self.properties.color_to_rerun(),
            labels=self.properties.strip_names,
            show_labels=self.properties.show_labels_to_rerun(),
        )

    def compute_strips(self, q: np.ndarray) -> list[list[list[list[float]]]]:
        """
        Returns
        -------
        list[list[list[list[float]]]]
            A nb_frames list of nb_strips list of 2 list of 3 list of float
        """
        nb_frames = q.shape[1]
        return [self.update_callable(q[:, f]) for f in range(nb_frames)]

    def to_chunk(self, q: np.ndarray) -> dict[str, list]:
        nb_frames = q.shape[1]

        strips_by_frame = self.compute_strips(q)
        # keep it for now
        # colors = [self.properties.color for _ in range(nb_frames)]
        # radii = [self.properties.radius for _ in range(nb_frames)]
        # return {
        #     f"{self.name}_{s}": [
        #         rr.LineStrips3D.indicator(),
        #         rr.components.LineStrip3DBatch([strips_by_frame[f][s] for f in range(nb_frames)]),
        #         rr.components.ColorBatch(colors),
        #         rr.components.RadiusBatch(radii),
        #         rr.components.TextBatch([self.properties.strip_names[s] for _ in range(nb_frames)]),
        #         rr.components.ShowLabelsBatch([False for _ in range(nb_frames)]),
        #     ]
        #     for s in range(self.nb_strips)
        # }
        # lets try a more advanced approach
        colors = self.properties.color_to_rerun(nb_frames)
        radii = [self.properties.radius for _ in range(nb_frames * self.nb_strips)]
        labels = [self.properties.strip_names[s] for _ in range(nb_frames) for s in range(self.nb_strips)]
        partition = [self.nb_strips for _ in range(nb_frames)]
        return {
            self.name: [
                rr.LineStrips3D.indicator(),
                rr.components.LineStrip3DBatch(
                    [strips_by_frame[f][s] for f in range(nb_frames) for s in range(self.nb_strips)]
                ).partition(partition),
                rr.components.ColorBatch(colors).partition(partition),
                rr.components.RadiusBatch(radii).partition(partition),
                rr.components.TextBatch(labels).partition(partition),
                rr.components.ShowLabelsBatch([False for _ in range(nb_frames * self.nb_strips)]).partition(partition),
            ]
        }


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


class ModelMarkerLinkUpdater(LineStripUpdater):
    def __init__(self, name, properties: LineStripProperties, update_callable: callable):
        super(ModelMarkerLinkUpdater, self).__init__(
            name=name + "/marker_links", properties=properties, update_callable=update_callable
        )

    def line_strips(self, q: np.ndarray, markers: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        q : np.ndarray
            Generalized coordinates, one dimension array, [N x 1]
        markers : np.ndarray
            Markers in the global reference frame, [3 x N_markers]

        Returns
        -------
        should return a [N x 2 x 3] array
        """
        output = np.zeros((self.properties.nb_strips, 2, 3))
        output[:, 0, :] = self.update_callable(q)
        output[:, 1, :] = markers.T

        return output

    def to_rerun(self, q: np.ndarray = None, markers: np.ndarray = None) -> None:
        rr.log(
            self.name,
            rr.LineStrips3D(
                strips=self.line_strips(q, markers),
                radii=self.properties.radius_to_rerun(),
                colors=self.properties.color_to_rerun(),
                # labels=self.properties.strip_names,
            ),
        )

    def compute_all_strips(self, q: np.ndarray, markers: np.ndarray) -> np.ndarray:
        nb_frames = q.shape[1]
        strips = np.zeros((self.nb_strips, 2, 3, nb_frames))
        for f in range(nb_frames):
            strips[:, :, :, f] = self.line_strips(q[:, f], markers[:, :, f])

        return strips

    def to_chunk(self, q: np.ndarray, markers: np.ndarray) -> dict[str, list]:
        nb_frames = q.shape[1]
        strips_by_frame = self.compute_all_strips(q, markers)

        # colors = [self.properties.color for _ in range(nb_frames)]
        # radii = [self.properties.radius for _ in range(nb_frames)]
        #
        # return {
        #     f"{self.name}_{s}": [
        #         rr.LineStrips3D.indicator(),
        #         rr.components.LineStrip3DBatch([strips_by_frame[s, :, :, f] for f in range(nb_frames)]),
        #         rr.components.ColorBatch(colors),
        #         rr.components.RadiusBatch(radii),
        #     ]
        #     for s in range(self.nb_strips)
        # }

        colors = [self.properties.color for _ in range(nb_frames * self.nb_strips)]
        radii = [self.properties.radius for _ in range(nb_frames * self.nb_strips)]
        labels = [self.properties.strip_names[s] for _ in range(nb_frames) for s in range(self.nb_strips)]
        partition = [self.nb_strips for _ in range(nb_frames)]

        return {
            self.name: [
                rr.LineStrips3D.indicator(),
                rr.components.LineStrip3DBatch(
                    [strips_by_frame[s, :, :, f] for f in range(nb_frames) for s in range(self.nb_strips)]
                ).partition(partition),
                rr.components.ColorBatch(colors).partition(partition),
                rr.components.RadiusBatch(radii).partition(partition),
                rr.components.TextBatch(labels).partition(partition),
                rr.components.ShowLabelsBatch([False for _ in range(nb_frames * self.nb_strips)]).partition(partition),
            ]
        }


class LineStripUpdaterFromGlobalTransform(LineStripUpdater):
    def __init__(self, name, properties: LineStripProperties, strips: np.ndarray, transform_callable: callable):
        super(LineStripUpdaterFromGlobalTransform, self).__init__(name, properties, update_callable=transform_callable)
        self.rerun_mesh = rr.LineStrips3D(
            strips=strips,
            radii=self.properties.radius_to_rerun(),
            colors=self.properties.color_to_rerun(),
        )

    def initialize(self):
        rr.log(
            self.name,
            self.rerun_mesh,
        )

    def to_rerun(self, q: np.ndarray) -> None:
        homogenous_matrices = self.update_callable(q)
        rr.log(
            self.name,
            rr.Transform3D(
                translation=homogenous_matrices[:3, 3],
                mat3x3=homogenous_matrices[:3, :3],
                # axis_length=1,
            ),
        )
        rr.log(
            self.name,
            self.rerun_mesh,
        )

    def compute_all_transforms(self, q: np.ndarray) -> np.ndarray:
        nb_frames = q.shape[1]
        homogenous_matrices = np.zeros((4, 4, nb_frames))
        for f in range(nb_frames):
            homogenous_matrices[:, :, f] = self.update_callable(q[:, f])

        return homogenous_matrices

    def to_chunk(self, q: np.ndarray) -> dict[str, list]:
        homogenous_matrices = self.compute_all_transforms(q)

        return {
            self.name: [
                rr.InstancePoses3D.indicator(),
                rr.components.PoseTranslation3DBatch(homogenous_matrices[:3, 3, :].T),
                rr.components.PoseTransformMat3x3Batch(
                    [homogenous_matrices[:3, :3, f] for f in range(homogenous_matrices.shape[2])]
                ),
                # rr.components.AxisLengthBatch([1] * homogenous_matrices.shape[2]),
            ]
        }
