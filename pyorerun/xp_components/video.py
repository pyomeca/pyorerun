import numpy as np
import rerun as rr

from ..abstract.abstract_class import ExperimentalData


class Video(ExperimentalData):
    def __init__(self, name: str, video_array: np.ndarray):
        self.name = name
        if video_array.dtype != np.uint8:
            raise ValueError("Video array should be a np.uint8")
        self.video = video_array  # shape (nb_frames, nb_vertical_pixels, nb_horizontal_pixels, nb_components)

    @property
    def nb_frames(self):
        return self.video.shape[0]

    def nb_vertical_pixels(self):
        return self.video.shape[1]

    def nb_horizontal_pixels(self):
        return self.video.shape[2]

    @property
    def nb_components(self):
        return 1

    def initialize(self):
        format_static = rr.components.ImageFormat(
            width=self.nb_horizontal_pixels(),
            height=self.nb_vertical_pixels(),
            color_model="RGB",
            channel_datatype="U8",
        )
        rr.log(
            self.name,
            [format_static, rr.Image.indicator()],
            static=True,
        )

    def to_rerun(self, frame: int) -> None:
        rr.log(
            self.name,
            self.to_component(frame),
        )

    def to_component(self, frame: int) -> rr.Image:
        return rr.Image(
            self.video[frame, :, :, :],
        )

    def to_chunk(self, **kwargs) -> dict[str, list]:
        return {
            self.name: [
                rr.components.ImageBufferBatch(self.video.reshape(self.nb_frames, -1)),
            ]
        }
