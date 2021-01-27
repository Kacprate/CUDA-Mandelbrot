import json
import logging

logger = logging.getLogger(__name__)

class Save_Manager:
    def __init__(self, save_file_path, slots=5):
        self.save_file_path = save_file_path
        self.saves = [None] * slots
        self.slot_number = slots

    def load(self):
        with open(self.save_file_path, 'r') as f:
            self.saves = json.loads(f.read())

    def save(self):
        with open(self.save_file_path, 'w') as f:
            json.dump(self.saves, f)

    def set_state(self, index, state):
        try:
            self.saves[index] = state
            return True
        except:
            logger.warning(f"Could not save state on index {index}")
            return False

    def get_state(self, index):
        try:
            state = self.saves[index]
            return state
        except:
            logger.warning(f"Could not load save {index}")
            return None