from pyorerun.timeless_components import TimelessRerunPhase


class MockTimelessComponent:
    def __init__(self, name):
        self.name = name
        self.rerun_called = False

    def to_rerun(self):
        self.rerun_called = True



def test_init():
    phase = TimelessRerunPhase("test_phase", 1)
    assert phase.name == "test_phase"
    assert phase.phase == 1
    assert len(phase.timeless_components) == 0


def test_add_component():
    phase = TimelessRerunPhase("test_phase", 1)
    component = MockTimelessComponent("test_component")
    phase.add_component(component)
    assert len(phase.timeless_components) == 1
    assert phase.timeless_components[0] == component


def test_to_rerun():
    phase = TimelessRerunPhase("test_phase", 1)
    component1 = MockTimelessComponent("component1")
    component2 = MockTimelessComponent("component2")
    phase.add_component(component1)
    phase.add_component(component2)
    
    phase.to_rerun()
    assert component1.rerun_called
    assert component2.rerun_called


def test_component_names():
    phase = TimelessRerunPhase("test_phase", 1)
    component1 = MockTimelessComponent("component1")
    component2 = MockTimelessComponent("component2")
    phase.add_component(component1)
    phase.add_component(component2)
    
    assert phase.component_names == ["component1", "component2"]
