"""Array validation shared by the truth-based TMRCA calibration metrics."""
import numpy as np


def validate_tmrca_arrays(truth, posterior, span_weights=None):
    """Return float64 times and normalized weights over (position, pair) cells.

    Draws have equal weight. Positions have equal weight unless genomic spans
    are supplied; haplotype pairs always have equal weight within a position.
    """
    truth = np.asarray(truth, dtype=np.float64)
    posterior = np.asarray(posterior, dtype=np.float64)
    if (truth.ndim != 2 or not truth.size or posterior.ndim != 3
            or posterior.shape[0] == 0 or posterior.shape[1:] != truth.shape):
        raise ValueError('Expected truth [positions, pairs] and posterior [draws, positions, pairs]')
    if (not np.isfinite(truth).all() or not np.isfinite(posterior).all()
            or np.any(truth < 0) or np.any(posterior < 0)):
        raise ValueError('TMRCA values must be finite and nonnegative')
    spans = (np.ones(truth.shape[0], dtype=np.float64) if span_weights is None
             else np.asarray(span_weights, dtype=np.float64))
    if (spans.shape != (truth.shape[0],) or not np.isfinite(spans).all()
            or np.any(spans < 0) or not np.any(spans > 0)):
        raise ValueError('Span weights must be finite, nonnegative, nonzero and match positions')
    scaled = spans / spans.max()
    weights = np.broadcast_to((scaled / scaled.sum())[:, None] / truth.shape[1], truth.shape)
    return truth, posterior, weights
