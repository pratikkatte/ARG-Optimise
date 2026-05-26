import math
import random


DEFAULT_TIME_BINS = 32
DEFAULT_TIME_DELTA_BIN_WIDTH = 0.001
DEFAULT_TIME_BIN_SCHEME = "TimeEnvFixedDelta"

class TimeEnvFixedDelta:
    """Fixed-width delta-time helper for bottom-up ARG construction.

    Actions 0..bins-2 represent finite intervals of width delta_bin_width.
    The final action is a tail bin [tail_start, infinity), so the discretized
    exponential waiting-time prior remains normalized without truncation.
    """

    def __init__(
        self,
        bins=DEFAULT_TIME_BINS,
        delta_bin_width=DEFAULT_TIME_DELTA_BIN_WIDTH,
    ):
        self.bins = int(bins)
        if self.bins < 2:
            raise ValueError("fixed delta time bins must be at least 2")
        self.delta_bin_width = float(delta_bin_width)
        if self.delta_bin_width <= 0.0:
            raise ValueError("time delta bin width must be positive")
        self.finite_bins = self.bins - 1
        self.tail_start = self.finite_bins * self.delta_bin_width

    def time_action_to_delta(self, action, rate=None):
        action = self._validate_action(action)
        lower_t, upper_t = self._time_bounds(action)
        return self._exponential_conditional_mean(lower_t, upper_t, rate)

    def delta_to_time_action(self, delta, rate=None):
        delta = float(delta)
        if delta < 0.0:
            raise ValueError("delta must be non-negative")
        if delta >= self.tail_start:
            return self.bins - 1
        return int(min(math.floor(delta / self.delta_bin_width), self.finite_bins - 1))

    def sample_action_from_prior(self, rate, rng=None):
        probabilities = self.time_action_probabilities(rate)
        rng = random if rng is None else rng
        return rng.choices(range(self.bins), weights=probabilities)[0]

    def time_action_log_probability(self, action, rate):
        # action = self._validate_action(action)
        lower_t, upper_t = self._time_bounds(action)
        if math.isinf(upper_t):
            return -rate * lower_t
        width = upper_t - lower_t
        interval_mass_from_zero = -math.expm1(-rate * width)
        return -rate * lower_t + math.log(interval_mass_from_zero)

    def time_action_probabilities(self, rate):
        return [
            math.exp(self.time_action_log_probability(action, rate))
            for action in range(self.bins)
        ]

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

    def _time_bounds(self, action):
        lower_t = action * self.delta_bin_width
        if action == self.bins - 1:
            return lower_t, math.inf
        return lower_t, lower_t + self.delta_bin_width

    def _exponential_conditional_mean(self, lower_t, upper_t, rate):
        if math.isinf(upper_t):
            return lower_t + 1.0 / rate
        width = upper_t - lower_t
        interval_mass_from_zero = -math.expm1(-rate * width)
        if interval_mass_from_zero <= 0.0:
            return lower_t
        tail_factor = math.exp(-rate * width)
        return lower_t + (1.0 / rate) - (width * tail_factor / interval_mass_from_zero)
