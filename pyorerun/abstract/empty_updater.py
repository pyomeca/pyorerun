import numpy as np


class EmptyUpdater:
    def __init__(self, name: str):
        self.name = name

    def to_rerun(self, q: np.ndarray) -> None:
        pass
