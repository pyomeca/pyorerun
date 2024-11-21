from .abstract.abstract_class import ExperimentalData


class XpRerunPhase:
    def __init__(self, name, phase: int):
        self.name = name
        self.xp_data = []
        self.phase = phase

    def add_data(self, xp_data: ExperimentalData):
        self.xp_data.append(xp_data)

    def to_rerun(self, frame: int):
        for data in self.xp_data:
            data.to_rerun(frame)

    def to_chunk(self):
        for data in self.xp_data:
            data.to_chunk()

    @property
    def component_names(self) -> list[str]:
        return [data.name for data in self.xp_data]
