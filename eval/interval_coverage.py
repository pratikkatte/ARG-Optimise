"""Empirical equal-tail TMRCA interval coverage, following SINGER Figure 4c."""
import numpy as np

from eval._calibration import validate_tmrca_arrays


def tmrca_interval_coverage(truth, posterior, span_weights=None, levels=(.5, .7, .9)):
    """Coverage and width of empirical equal-tail intervals at each level.

    Inputs are truth [positions, pairs] and draws [draws, positions, pairs],
    in the same time units. NumPy linear quantiles define the endpoints;
    coverage includes both endpoints. One draw gives a zero-width interval,
    whose descriptive coverage is not evidence of posterior calibration.
    """
    truth, posterior, weights = validate_tmrca_arrays(truth, posterior, span_weights)
    levels = np.asarray(levels, dtype=np.float64)
    if (levels.ndim != 1 or not levels.size or not np.isfinite(levels).all()
            or np.any(levels <= 0) or np.any(levels >= 1)
            or len(np.unique(levels)) != len(levels)):
        raise ValueError('Coverage levels must be unique finite probabilities strictly between 0 and 1')
    # Preserve the exact historical 0.05 and 0.95 floating-point endpoints.
    tails = (100. - 100.*levels) / 200.
    quantiles = np.quantile(posterior, np.concatenate([tails, 1.-tails]), axis=0, method='linear')
    intervals = []
    total_weight = float(weights.sum())
    for index, level in enumerate(levels):
        low, high = quantiles[index], quantiles[index+len(levels)]
        covered = (low <= truth) & (truth <= high)
        coverage = float(np.clip(np.sum(weights*covered)/total_weight, 0., 1.))
        intervals.append(dict(level=float(level), coverage=coverage,
            coverage_error=coverage-float(level),
            mean_width=float(np.sum(weights*(high-low))/total_weight),
            lower_quantile=float(tails[index]), upper_quantile=float(1.-tails[index]),
            lower=low.tolist(), upper=high.tolist()))
    return dict(draw_count=int(posterior.shape[0]), cell_count=int(truth.size),
        quantile_method='linear', endpoints='inclusive', intervals=intervals)
