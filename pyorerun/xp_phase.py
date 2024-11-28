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

    def to_chunk(self) -> dict[str, list]:
        output = {}
        for data in self.xp_data:
            output.update(data.to_chunk())
        return output

    @property
    def component_names(self) -> list[str]:
        return [data.name for data in self.xp_data]
