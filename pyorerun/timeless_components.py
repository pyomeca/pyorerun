from typing import Any


class TimelessRerunPhase:
    def __init__(self, name, phase: int):
        self.name = name
        self.phase = phase
        self.timeless_components = []

    def add_component(self, component: Any):
        self.timeless_components.append(component)

    def to_rerun(self):
        for data in self.timeless_components:
            data.to_rerun()

    @property
    def component_names(self) -> list[str]:
        return [data.name for data in self.timeless_components]
