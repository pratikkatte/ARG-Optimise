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

    def time_action_to_delta(self, action, rate=None, max_delta=None):
        action = self._validate_action(action)
        lower_t, upper_t = self._time_bounds(action)
        if max_delta is not None:
            max_delta = self._validate_max_delta(max_delta)
            upper_t = min(upper_t, max_delta)
            if not lower_t < upper_t:
                raise ValueError(
                    "time action has no probability mass before max_delta"
                )
        return self._exponential_conditional_mean(lower_t, upper_t, rate)

    def delta_to_time_action(self, delta, rate=None):
        delta = float(delta)
        if delta < 0.0:
            raise ValueError("delta must be non-negative")
        if delta >= self.tail_start:
            return self.bins - 1
        return int(min(math.floor(delta / self.delta_bin_width), self.finite_bins - 1))

    def sample_action_from_prior(self, rate, rng=None, max_delta=None):
        probabilities = self.time_action_probabilities(
            rate,
            max_delta=max_delta,
        )
        if sum(probabilities) <= 0.0:
            raise ValueError("no event-time probability mass is available")
        rng = random if rng is None else rng
        return rng.choices(range(self.bins), weights=probabilities)[0]

    def time_action_log_probability(self, action, rate, max_delta=None):
        action = self._validate_action(action)
        rate = self._validate_rate(rate)
        lower_t, upper_t = self._time_bounds(action)
        if max_delta is not None:
            max_delta = self._validate_max_delta(max_delta)
            upper_t = min(upper_t, max_delta)
            if not lower_t < upper_t:
                return -math.inf
        if math.isinf(upper_t):
            return -rate * lower_t
        width = upper_t - lower_t
        interval_mass_from_zero = -math.expm1(-rate * width)
        return -rate * lower_t + math.log(interval_mass_from_zero)

    def time_action_probabilities(self, rate, max_delta=None):
        return [
            math.exp(
                self.time_action_log_probability(
                    action,
                    rate,
                    max_delta=max_delta,
                )
            )
            for action in range(self.bins)
        ]

    def bounded_waiting_distribution(self, rate, max_delta):
        """Return event-bin masses and no-event survival to ``max_delta``.

        The event masses are not conditioned on an event occurring before the
        bound. Consequently, their sum plus the returned survival mass is one.
        This is the distribution needed when a deterministic fixed ARG event
        competes with the next sampled coalescence/recombination event.
        """

        rate = self._validate_rate(rate)
        max_delta = self._validate_max_delta(max_delta)
        event_masses = tuple(
            self.time_action_probabilities(rate, max_delta=max_delta)
        )
        survival_mass = math.exp(-rate * max_delta)
        total = sum(event_masses) + survival_mass
        if not math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(
                "bounded waiting-time distribution is not normalized"
            )
        return event_masses, survival_mass

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

    def _validate_max_delta(self, max_delta):
        max_delta = float(max_delta)
        if not math.isfinite(max_delta) or max_delta < 0.0:
            raise ValueError("max_delta must be finite and non-negative")
        return max_delta

    def _time_bounds(self, action):
        lower_t = action * self.delta_bin_width
        if action == self.bins - 1:
            return lower_t, math.inf
        return lower_t, lower_t + self.delta_bin_width

    def _exponential_conditional_mean(self, lower_t, upper_t, rate):
        rate = self._validate_rate(rate)
        if math.isinf(upper_t):
            return lower_t + 1.0 / rate
        width = upper_t - lower_t
        interval_mass_from_zero = -math.expm1(-rate * width)
        if interval_mass_from_zero <= 0.0:
            return lower_t
        tail_factor = math.exp(-rate * width)
        return lower_t + (1.0 / rate) - (width * tail_factor / interval_mass_from_zero)
