import numpy as np


class EmptyUpdater:
    def __init__(self, name: str):
        self.name = name

    def to_rerun(self, q: np.ndarray) -> None:
        pass

    def to_chunk(self, q: np.ndarray) -> dict[str, list]:
        return {"empty" : None}

