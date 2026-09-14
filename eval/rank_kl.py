"""KL of an empirical rank histogram against its uniform-rank reference."""
import math

import numpy as np


def rank_kl_divergence(probabilities, uniform_probabilities=None):
    """Return D_KL(observed || reference) in nats, with no pseudocounts.

    Zero observed masses contribute zero. Supply the histogram's
    ``uniform_probabilities`` for grouped ranks: unequal bin sizes have
    unequal mass under the discrete uniform rank distribution. Without a
    reference, every bin is assumed to have the same null mass. Counts are
    accepted and normalized. This is a descriptive discrepancy, not a p-value.
    """
    observed = np.asarray(probabilities, dtype=np.float64)
    if (observed.ndim != 1 or not observed.size or not np.isfinite(observed).all()
            or np.any(observed < 0) or not np.any(observed > 0)):
        raise ValueError('Rank masses must be a finite nonnegative nonzero vector')
    reference = (np.ones_like(observed) if uniform_probabilities is None
                 else np.asarray(uniform_probabilities, dtype=np.float64))
    if (reference.shape != observed.shape or not np.isfinite(reference).all()
            or np.any(reference <= 0)):
        raise ValueError('Uniform reference masses must be finite, positive and match rank bins')
    # Scaling first avoids overflow when callers supply large weighted counts.
    p = observed / observed.max(); p /= p.sum()
    log_q = np.log(reference) - math.log(float(reference.max()))
    log_q -= math.log(float(np.sum(reference / reference.max())))
    positive = p > 0
    value = math.fsum((p[positive]*(np.log(p[positive])-log_q[positive])).tolist())
    return max(0., value)  # Roundoff can put a matching histogram just below zero.
