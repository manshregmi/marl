# models/a2c/critic.py
class TabularCritic:
    """Simple dictionary‑based value table."""
    def __init__(self):
        self.table = {}

    def get(self, key):
        return self.table.get(key, 0.0)

    def update(self, key, td_error, alpha):
        self.table[key] = self.get(key) + alpha * td_error

    def set_weights(self, new_table):
        self.table = new_table

    def get_weights(self):
        return self.table.copy()