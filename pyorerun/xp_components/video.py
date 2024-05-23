import numpy as np
import rerun as rr

from ..abstract.abstract_class import ExperimentalData


class Video(ExperimentalData):
    def __init__(self, name: str, video_array: np.ndarray):
        self.name = name
        self.video = video_array

    @property
    def nb_frames(self):
        return self.video.shape[0]

    def nb_vertical_pixels(self):
        return self.video.shape[2]

    def nb_horizontal_pixels(self):
        return self.video.shape[1]

    @property
    def nb_components(self):
        return 1

    def to_rerun(self, frame: int) -> None:
        rr.log(
            self.name,
            rr.Image(
                self.video[frame, :, :, :],
            ),
        )
