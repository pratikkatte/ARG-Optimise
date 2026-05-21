import math
import random


DEFAULT_TIME_BINS = 32
DEFAULT_TIME_TAIL_PROBABILITY = 1e-4
DEFAULT_TIME_MODEL = "adaptive_exponential"
LEGACY_TIME_MODEL = "fixed_increments"
DEFAULT_TIME_INCREMENTS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]


class TimeEnvCategorical:
    """Categorical waiting-time helper for bottom-up ARG construction."""

    def __init__(
        self,
        increments=None,
        bins=DEFAULT_TIME_BINS,
        tail_probability=DEFAULT_TIME_TAIL_PROBABILITY,
        time_model=DEFAULT_TIME_MODEL,
    ):
        if increments is not None:
            self.time_model = LEGACY_TIME_MODEL
            self.increments = [float(x) for x in increments]
            if len(self.increments) == 0:
                raise ValueError("time increments must not be empty")
            if any(x <= 0 for x in self.increments):
                raise ValueError("time increments must all be positive")
            if any(left >= right for left, right in zip(self.increments, self.increments[1:])):
                raise ValueError("time increments must be strictly increasing")
            self.bins = len(self.increments)
            self.boundaries = self._compute_fixed_boundaries(self.increments)
            self.tail_probability = None
            self.probability_boundaries = None
            return

        if time_model != DEFAULT_TIME_MODEL:
            raise ValueError(
                f"unsupported time_model={time_model!r}; use {DEFAULT_TIME_MODEL!r} "
                "or provide legacy time_increments"
            )
        self.time_model = DEFAULT_TIME_MODEL
        self.increments = None
        self.bins = int(bins)
        if self.bins < 2:
            raise ValueError("adaptive time bins must be at least 2")
        self.tail_probability = float(tail_probability)
        if not 0.0 < self.tail_probability < 1.0:
            raise ValueError("time tail probability must be in (0, 1)")
        self.boundaries = None
        self.probability_boundaries = self._compute_probability_boundaries(
            self.bins,
            self.tail_probability,
        )

    @property
    def is_adaptive(self):
        return self.time_model == DEFAULT_TIME_MODEL

    def time_action_to_delta(self, action, rate=None):
        action = self._validate_action(action)
        if not self.is_adaptive:
            return self.increments[action]
        rate = self._validate_rate(rate)
        lower_p = self.probability_boundaries[action]
        upper_p = self.probability_boundaries[action + 1]
        return self._exponential_conditional_mean(lower_p, upper_p, rate)

    def delta_to_time_action(self, delta, rate=None):
        delta = float(delta)
        if self.is_adaptive:
            rate = self._validate_rate(rate)
            representatives = [
                self.time_action_to_delta(action, rate)
                for action in range(self.bins)
            ]
        else:
            representatives = self.increments
        distances = [abs(delta - representative) for representative in representatives]
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
        if self.is_adaptive:
            probability = self.probability_boundaries[action + 1] - self.probability_boundaries[action]
            return math.log(probability)
        lower = self.boundaries[action]
        upper = self.boundaries[action + 1]
        if math.isinf(upper):
            return -rate * lower
        return -rate * lower + math.log(-math.expm1(-rate * (upper - lower)))

    def time_action_probabilities(self, rate):
        rate = self._validate_rate(rate)
        if self.is_adaptive:
            return [
                self.probability_boundaries[action + 1] - self.probability_boundaries[action]
                for action in range(self.bins)
            ]
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
        if rate is None:
            raise ValueError("waiting-time rate is required")
        rate = float(rate)
        if rate <= 0:
            raise ValueError("waiting-time rate must be positive")
        return rate

    def _compute_fixed_boundaries(self, increments):
        boundaries = [0.0]
        boundaries.extend(
            (left + right) / 2.0
            for left, right in zip(increments, increments[1:])
        )
        boundaries.append(math.inf)
        return boundaries

    def _compute_probability_boundaries(self, bins, tail_probability):
        body_mass = 1.0 - tail_probability
        boundaries = [
            body_mass * action / float(bins - 1)
            for action in range(bins)
        ]
        boundaries.append(1.0)
        return boundaries

    def _exponential_conditional_mean(self, lower_p, upper_p, rate):
        lower_survival = 1.0 - lower_p
        upper_survival = max(0.0, 1.0 - upper_p)
        lower_time = -math.log(lower_survival) / rate
        lower_term = (lower_time + 1.0 / rate) * lower_survival
        if upper_survival == 0.0:
            upper_term = 0.0
        else:
            upper_time = -math.log(upper_survival) / rate
            upper_term = (upper_time + 1.0 / rate) * upper_survival
        probability = lower_survival - upper_survival
        return (lower_term - upper_term) / probability
