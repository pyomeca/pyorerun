import numpy as np
import rerun as rr

from ..abstract.abstract_class import ExperimentalData


class DepthImage(ExperimentalData):
    def __init__(self, name, depth_image: np.ndarray):
        self.name: str = name + "/depth_image"
        self.depth_image: np.ndarray = depth_image

    @property
    def size_x(self):
        return self.depth_image.shape[0]

    @property
    def size_y(self):
        return self.depth_image.shape[1]

    @property
    def nb_frames(self):
        return self.depth_image.shape[2]

    @property
    def nb_components(self):
        return 1

    def to_rerun(self, frame: int) -> None:
        depth_image_frame = self.depth_image[:, :, frame]
        rr.log(
            self.name,
            rr.Pinhole(
                width=depth_image_frame.shape[1],
                height=depth_image_frame.shape[0],
                focal_length=200,
            ),
        )

        # Log the tensor.
        rr.log(f"{self.name}/depth", rr.DepthImage(depth_image_frame, meter=10_000.0))
