import math
import random


DEFAULT_TIME_BINS = 32
DEFAULT_TIME_DELTA_BIN_WIDTH = 0.001
DEFAULT_TIME_BIN_SCHEME = "TimeEnvFixedDelta"
TIME_POLICIES = ("categorical", "cwr_exponential")
CONTINUOUS_TIME_SCHEME = "cwr_exponential_v1"
CONTINUOUS_TIME_UNITS = "2Ne"


def validate_time_policy(policy):
    if policy not in TIME_POLICIES:
        raise ValueError(f"Unknown time_policy: {policy!r}")
    return policy


def validate_temperature(random_spec, *, time_component=False):
    temperature = 1.0 if random_spec is None else float(random_spec["T"])
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    time_temperature = temperature if random_spec is None else float(random_spec.get('time_T', temperature))
    if not math.isfinite(time_temperature) or time_temperature <= 0:
        raise ValueError('waiting-time temperature must be finite and positive')
    return time_temperature if time_component else temperature


def checkpoint_time_policy(metadata):
    """Validate redundant timing metadata; missing legacy modes are categorical."""
    timing = metadata.get("time", {})
    policy = validate_time_policy(metadata.get("time_policy", timing.get("time_policy", "categorical")))
    model_policy = metadata.get("model", {}).get("time_policy", "categorical")
    if model_policy != policy or timing.get("time_policy", policy) != policy:
        raise ValueError("Checkpoint environment and model time_policy disagree")
    if policy == "cwr_exponential":
        for key, expected in (("time_scheme", CONTINUOUS_TIME_SCHEME),
                              ("time_units", CONTINUOUS_TIME_UNITS)):
            if metadata.get(key, timing.get(key)) != expected or timing.get(key, expected) != expected:
                raise ValueError(f"Unsupported continuous {key}; expected {expected!r}")
    return policy


class TimeEnvCwrExponential:
    """Continuous CwR waiting-time prior in internal 2Ne units (not the policy)."""

    @property
    def metadata(self):
        return {"time_policy": "cwr_exponential", "time_scheme": CONTINUOUS_TIME_SCHEME,
                "time_units": CONTINUOUS_TIME_UNITS}

    @staticmethod
    def positive(value, name):
        if value is None or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"continuous {name} must be finite and positive")
        return float(value)

    def log_density(self, delta_t, rate):
        rate = self.positive(rate, "rate")
        delta_t = self.positive(delta_t, "wait")
        score = math.log(rate) - rate * delta_t
        if not math.isfinite(score):
            raise ValueError("non-finite continuous prior log density")
        return score

    def event_time(self, current_time, delta_t):
        delta_t = self.positive(delta_t, "wait")
        current_time = float(current_time)
        event_time = current_time + delta_t
        if not math.isfinite(current_time) or current_time < 0 or not math.isfinite(event_time) or event_time <= current_time:
            raise ValueError("continuous event time must be finite and strictly increasing in float64")
        return event_time

    def sample_from_prior(self, rate, rng=None):
        rate = self.positive(rate, "rate")
        rng = random if rng is None else rng
        return self.positive(rng.expovariate(rate), "sampled wait")

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
