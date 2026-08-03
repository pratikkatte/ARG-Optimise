import math
import random


DEFAULT_TIME_SCHEME = "ContinuousCWRConditionalCDF"
DEFAULT_TIME_DENSITY = "BernsteinBeta"
DEFAULT_TIME_BASIS_COMPONENTS = 16
TIME_REFERENCE_MEASURE = "delta_t_over_2Ne"
DEFAULT_DEMOGRAPHY_MODEL = "constant_ne"


class ContinuousCoalescentTime:
    """Exact constant-Ne CWR waiting-time transformations.

    Between ARG events the total event rate is constant in ``2Ne`` time units,
    so the next waiting time is exponential with density
    ``rate * exp(-rate * delta)``. When ``max_delta`` is supplied, the event
    time is represented by its conditional-CDF quantile before that fixed
    boundary. The complementary survival mass belongs to the deterministic
    fixed-ancestor transition.
    """

    scheme = DEFAULT_TIME_SCHEME
    density = DEFAULT_TIME_DENSITY
    reference_measure = TIME_REFERENCE_MEASURE
    demography_model = DEFAULT_DEMOGRAPHY_MODEL

    def cdf(self, delta, rate):
        """Return the exponential waiting-time CDF at ``delta``."""

        delta = self._validate_delta(delta)
        rate = self._validate_rate(rate)
        return -math.expm1(-rate * delta)

    def inverse_cdf(self, probability, rate):
        """Invert the exponential CDF using a stable open-upper endpoint."""

        probability = float(probability)
        if (
            not math.isfinite(probability)
            or probability < 0.0
            or probability >= 1.0
        ):
            raise ValueError(
                "CDF probability must be finite and lie in [0, 1)"
            )
        rate = self._validate_rate(rate)
        return -math.log1p(-probability) / rate

    def generated_probability(self, rate, max_delta=None):
        rate = self._validate_rate(rate)
        if max_delta is None:
            return 1.0
        max_delta = self._validate_max_delta(max_delta)
        return self.cdf(max_delta, rate)

    def survival_probability(self, rate, max_delta):
        rate = self._validate_rate(rate)
        max_delta = self._validate_max_delta(max_delta)
        return math.exp(-rate * max_delta)

    def survival_log_probability(self, rate, max_delta):
        rate = self._validate_rate(rate)
        max_delta = self._validate_max_delta(max_delta)
        return -rate * max_delta

    def bounded_waiting_distribution(self, rate, max_delta):
        """Return exact generated-event and fixed-boundary masses."""

        generated = self.generated_probability(rate, max_delta=max_delta)
        survival = self.survival_probability(rate, max_delta)
        if not math.isclose(
            generated + survival,
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "bounded continuous waiting-time distribution is not normalized"
            )
        return generated, survival

    def sample_prior_quantile(self, rng=None):
        rng = random if rng is None else rng
        return self._open_unit_interval(float(rng.random()))

    def quantile_to_delta(self, quantile, rate, max_delta=None):
        quantile = self._validate_quantile(quantile)
        rate = self._validate_rate(rate)
        generated = self.generated_probability(rate, max_delta=max_delta)
        if generated <= 0.0:
            raise ValueError(
                "no continuous event-time probability mass is available"
            )
        delta = max(
            self.inverse_cdf(quantile * generated, rate),
            math.nextafter(0.0, 1.0),
        )
        if max_delta is not None:
            max_delta = self._validate_max_delta(max_delta)
            # For extremely small lambda * H, multiplying an open quantile by
            # the generated mass can round back to the endpoint. Preserve the
            # strict biological ordering required by tskit.
            delta = min(delta, math.nextafter(max_delta, 0.0))
        return delta

    def clamp_delta_before_absolute_boundary(
        self,
        delta,
        current_time,
        boundary_time,
    ):
        """Keep ``current_time + delta`` strictly below an absolute boundary.

        ``quantile_to_delta`` already returns a delta strictly below a relative
        horizon.  At large absolute times, however, adding that open delta can
        round back onto the boundary.  Canonicalize only that numerical edge
        case to the immediately preceding representable event time.
        """

        delta = self._validate_delta(delta)
        if not delta > 0.0:
            raise ValueError("event delta must be strictly positive")
        current_time = float(current_time)
        boundary_time = float(boundary_time)
        if not math.isfinite(current_time) or not math.isfinite(boundary_time):
            raise ValueError("absolute event-time bounds must be finite")
        if not current_time < boundary_time:
            raise ValueError("absolute boundary must be after current_time")
        if current_time + delta < boundary_time:
            return delta

        strict_event_time = math.nextafter(boundary_time, current_time)
        strict_delta = strict_event_time - current_time
        if not strict_delta > 0.0:
            raise ValueError(
                "no representable continuous event time exists before boundary"
            )
        # Subtraction followed by addition can itself round upward on unusual
        # exponent boundaries.  Step the delta inward until the invariant is
        # satisfied; this normally executes zero times.
        while current_time + strict_delta >= boundary_time:
            strict_delta = math.nextafter(strict_delta, 0.0)
            if not strict_delta > 0.0:
                raise ValueError(
                    "no representable continuous event time exists before boundary"
                )
        return min(delta, strict_delta)

    def event_time_after_delta(
        self,
        delta,
        current_time,
        boundary_time=None,
    ):
        """Convert a positive wait to a strictly later representable time.

        A positive subnormal ``delta`` can disappear when added to a much
        larger absolute ARG time.  Event times, unlike relative waits, must be
        strictly ordered, so promote that numerical edge case to the next
        representable float.  When a fixed-ancestor boundary is supplied, the
        promoted time must still remain strictly before it.
        """

        delta = self._validate_delta(delta)
        current_time = float(current_time)
        if not math.isfinite(current_time):
            raise ValueError("current_time must be finite")

        if boundary_time is not None:
            boundary_time = float(boundary_time)
            delta = self.clamp_delta_before_absolute_boundary(
                delta,
                current_time,
                boundary_time,
            )

        event_time = current_time + delta
        if not event_time > current_time:
            event_time = math.nextafter(current_time, math.inf)

        if boundary_time is not None and not event_time < boundary_time:
            raise ValueError(
                "no representable continuous event time exists before boundary"
            )
        return event_time

    def delta_to_quantile(self, delta, rate, max_delta=None):
        delta = self._validate_delta(delta)
        rate = self._validate_rate(rate)
        if max_delta is not None:
            max_delta = self._validate_max_delta(max_delta)
            if not delta < max_delta:
                raise ValueError(
                    "continuous event time must be strictly before max_delta"
                )
        generated = self.generated_probability(rate, max_delta=max_delta)
        if generated <= 0.0:
            raise ValueError(
                "no continuous event-time probability mass is available"
            )
        quantile = self.cdf(delta, rate) / generated
        return self._open_unit_interval(quantile)

    def waiting_time_log_density(self, delta, rate, max_delta=None):
        """Canonical unconditional exponential log density in scaled time."""

        delta = self._validate_delta(delta)
        rate = self._validate_rate(rate)
        if max_delta is not None:
            max_delta = self._validate_max_delta(max_delta)
            if not delta < max_delta:
                return -math.inf
        return math.log(rate) - rate * delta

    def prior_log_density(self, delta, rate, max_delta=None):
        """Public hazard-interface name for the biological prior density."""

        return self.waiting_time_log_density(
            delta,
            rate,
            max_delta=max_delta,
        )

    def log_abs_quantile_jacobian(self, delta, rate, max_delta=None):
        """Return ``log |du / d(delta)|`` for the conditional-CDF map."""

        delta = self._validate_delta(delta)
        rate = self._validate_rate(rate)
        generated = self.generated_probability(rate, max_delta=max_delta)
        if generated <= 0.0:
            return -math.inf
        if max_delta is not None and not delta < float(max_delta):
            return -math.inf
        return math.log(rate) - rate * delta - math.log(generated)

    @staticmethod
    def _validate_rate(rate):
        rate = float(rate)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("waiting-time rate must be finite and positive")
        return rate

    @staticmethod
    def _validate_max_delta(max_delta):
        max_delta = float(max_delta)
        if not math.isfinite(max_delta) or max_delta < 0.0:
            raise ValueError("max_delta must be finite and non-negative")
        return max_delta

    @staticmethod
    def _validate_delta(delta):
        delta = float(delta)
        if not math.isfinite(delta) or delta < 0.0:
            raise ValueError("delta must be finite and non-negative")
        return delta

    @staticmethod
    def _validate_quantile(quantile):
        quantile = float(quantile)
        if not math.isfinite(quantile) or not 0.0 < quantile < 1.0:
            raise ValueError("time quantile must lie strictly inside (0, 1)")
        return quantile

    @staticmethod
    def _open_unit_interval(value):
        return min(
            max(float(value), math.nextafter(0.0, 1.0)),
            math.nextafter(1.0, 0.0),
        )
