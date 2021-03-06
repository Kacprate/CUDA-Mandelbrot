class State:
    def __init__(self, name, next_states):
        self.name = name
        self.next = next_states
        self.id = None

class State_Machine:
    def __init__(self, states):
        if type(states) != dict:
            raise TypeError("states must be a dictionary")

        self.states = states

        self.names_to_id = {}
        for id, state in self.states.items():
            if type(state) != State:
                raise TypeError("All states in states dictionary must be of type State")
            self.names_to_id[state.name] = id
            state.id = id
            
        self.state = self.states[1]

    def get_state(self):
        return self.state

    def get_state_id_from_name(self, name):
        if name in self.names_to_id:
            return self.names_to_id[name]
        else:
            return None

    def change_state(self, state):
        if type(state) == str:
            id = self.get_state_id_from_name(state)
        elif type(state) == int:
            id = state
        else:
            raise TypeError(f"State must be an integer or string, got: {type(state)}")

        if id is None:
            raise KeyError(f"No state {state}")

        if id in self.state.next:
            self.state = self.states[id]
        else:
            raise ValueError(f"Can't go from state {self.state.name} to {self.states[id].name}")
