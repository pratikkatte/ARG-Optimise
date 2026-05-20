import random


DEFAULT_TIME_INCREMENTS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]


class TimeEnvCategorical:
    """Categorical waiting-time helper for bottom-up ARG construction."""

    def __init__(self, increments=None):
        increments = DEFAULT_TIME_INCREMENTS if increments is None else increments
        self.increments = [float(x) for x in increments]
        if len(self.increments) == 0:
            raise ValueError("time increments must not be empty")
        if any(x <= 0 for x in self.increments):
            raise ValueError("time increments must all be positive")
        self.bins = len(self.increments)

    def time_action_to_delta(self, action):
        action = int(action)
        if action < 0 or action >= self.bins:
            raise ValueError(f"time_action must be in [0, {self.bins - 1}], got {action}")
        return self.increments[action]

    def delta_to_time_action(self, delta):
        delta = float(delta)
        distances = [abs(delta - increment) for increment in self.increments]
        return int(min(range(self.bins), key=lambda idx: distances[idx]))

    def generate_random_action(self):
        return random.randint(0, self.bins - 1)