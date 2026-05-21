import math
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
        if any(left >= right for left, right in zip(self.increments, self.increments[1:])):
            raise ValueError("time increments must be strictly increasing")
        self.bins = len(self.increments)
        self.boundaries = self._compute_boundaries(self.increments)

    def time_action_to_delta(self, action):
        action = self._validate_action(action)
        return self.increments[action]

    def delta_to_time_action(self, delta):
        delta = float(delta)
        distances = [abs(delta - increment) for increment in self.increments]
        return int(min(range(self.bins), key=lambda idx: distances[idx]))

    def generate_random_action(self):
        return random.randint(0, self.bins - 1)

    def sample_action_from_prior(self, rate, rng=None):
        probabilities = self.time_action_probabilities(rate)
        rng = random if rng is None else rng
        target = rng.random()
        cumulative = 0.0
        for action, probability in enumerate(probabilities):
            cumulative += probability
            if target <= cumulative:
                return action
        return self.bins - 1

    def time_action_log_probability(self, action, rate):
        action = self._validate_action(action)
        rate = self._validate_rate(rate)
        lower = self.boundaries[action]
        upper = self.boundaries[action + 1]
        if math.isinf(upper):
            return -rate * lower
        return -rate * lower + math.log(-math.expm1(-rate * (upper - lower)))

    def time_action_probabilities(self, rate):
        rate = self._validate_rate(rate)
        probabilities = []
        for action in range(self.bins):
            lower = self.boundaries[action]
            upper = self.boundaries[action + 1]
            if math.isinf(upper):
                probabilities.append(math.exp(-rate * lower))
            else:
                probabilities.append(math.exp(-rate * lower) - math.exp(-rate * upper))
        return probabilities

    def _validate_action(self, action):
        action = int(action)
        if action < 0 or action >= self.bins:
            raise ValueError(f"time_action must be in [0, {self.bins - 1}], got {action}")
        return action

    def _validate_rate(self, rate):
        rate = float(rate)
        if rate <= 0:
            raise ValueError("waiting-time rate must be positive")
        return rate

    def _compute_boundaries(self, increments):
        boundaries = [0.0]
        boundaries.extend(
            (left + right) / 2.0
            for left, right in zip(increments, increments[1:])
        )
        boundaries.append(math.inf)
        return boundaries
