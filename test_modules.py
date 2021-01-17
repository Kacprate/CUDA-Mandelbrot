import pytest
import pygame

pygame.init()

# state machine

from modules.state_machine import State_Machine, State

@pytest.fixture(scope="session")
def smachine():
    first_state = State("rendering", [2, 3, 4])
    second_state = State("choosing_save_to_load", [1])
    state_machine = State_Machine(states={1: first_state, 
                                          2: second_state,
                                          3: State("choosing_save_to_save", [1]),
                                          4: State("choosing_save_to_remove", [1])})
    return state_machine, first_state

def test_statemachine_state():
    state = State("test", [1, 2, 3])
    assert state.name == "test"
    assert state.next == [1, 2, 3]
    assert state.id == None

def test_statemachine(smachine):
    state_machine, first_state = smachine

    assert state_machine.get_state() == first_state
    assert state_machine.get_state_id_from_name("choosing_save_to_load") == 2
    state_machine.change_state("choosing_save_to_save")
    assert state_machine.get_state().id == 3
    state_machine.change_state(1)
    assert state_machine.get_state().id == 1

def test_statemachine_fails(smachine):
    smachine, _ = smachine
    with pytest.raises(TypeError):
        state_machine = State_Machine("test")

    with pytest.raises(TypeError):
        state_machine = State_Machine({1: "test"})

    with pytest.raises(TypeError):
        smachine.change_state({})

    with pytest.raises(KeyError):
        smachine.change_state(5)

    with pytest.raises(ValueError):
        smachine.change_state(1)


# save_manager

from modules.save_manager import Save_Manager

@pytest.fixture(scope="module")
def smanager():
    save_manager = Save_Manager("./test_utils/saves.json")
    return save_manager

def test_savemanager_load(smanager):
    smanager.load()

def test_savemanager_setstate(smanager):
    assert smanager.set_state(0, {1: "test"})
    assert not smanager.set_state(smanager.slot_number + 1, "test")

def test_savemanager_getstate(smanager):
    assert smanager.get_state(0)[1] == "test"
    assert smanager.get_state(smanager.slot_number + 1) is None

def test_savemanager_save(smanager):
    smanager.save()
    with open(smanager.save_file_path) as f:
        assert f.read() == "[{\"1\": \"test\"}, null, null, null, null]"

def test_savemanager(smanager):
    smanager.load()
    assert smanager.get_state(0)['1'] == "test"


# renderer

from modules.renderer import Renderer

@pytest.fixture(scope="module")
def renderer():
    ren = Renderer("./config.json")
    return ren

def test_renderer(renderer):
    renderer = renderer
    state = renderer.get_state()
    renderer.load_state(state.copy())
    assert renderer.get_state() == state
    assert type(renderer.get_render_data(60)) == list
    for label in renderer.get_render_data(60):
        assert type(label) == str

@pytest.mark.parametrize("data", [(x, y, update) for x in [-10, 0, 10] for y in [-10, 0, 10] for update in [False, True]])
def test_renderer_step(renderer, data):
    renderer.step(10, 10, False)

def test_renderer_show(renderer):
    renderer.show_saves([{'timestamp': 123, 'center': {123, 123}}])
    renderer.show_info(60)