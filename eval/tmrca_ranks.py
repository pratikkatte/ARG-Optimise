"""Truth TMRCA rank histograms, following SINGER Figure 4b.

This is a Python implementation of the paper's diagnostic, not a source port:
the inspected SINGER repository does not contain the benchmark calculation.
See docs/EVALUATION.md for source provenance and finite-rank conventions.
"""
import numpy as np

from eval._calibration import validate_tmrca_arrays


def tmrca_rank_histogram(truth, posterior, span_weights=None, bins=20):
    """Rank true times among M draws, with all M+1 ranks (0 through M).

    Exact ties share their mass uniformly over the possible tie-broken ranks.
    This averages over random tie breaking without consuming RNG state. Keep
    the reported tie fraction: pervasive ties can hide errors in a histogram.

    Consecutive integer ranks are grouped into at most ``bins`` bins. Their
    null masses reflect the number of ranks in each bin; they need not be
    equal when M+1 is not divisible by the bin count. Weights represent genomic
    spans, not independent observations or importance weights over ARG draws.
    """
    truth, posterior, weights = validate_tmrca_arrays(truth, posterior, span_weights)
    if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError('Rank bins must be a positive integer')
    draws = posterior.shape[0]
    lower = np.zeros(truth.shape, dtype=np.int64)
    ties = np.zeros_like(lower)
    # Only allocate one position-by-pair comparison at a time.
    for sample in posterior:
        lower += sample < truth
        ties += sample == truth
    upper = lower + ties
    mass = (weights / (ties + 1)).ravel()
    # A difference array distributes each cell's mass over [lower, upper].
    delta = np.bincount(lower.ravel(), weights=mass, minlength=draws+2)
    delta -= np.bincount((upper+1).ravel(), weights=mass, minlength=draws+2)
    probabilities = np.maximum(np.cumsum(delta)[:-1], 0.)
    probabilities /= probabilities.sum()
    count = min(int(bins), draws+1)
    edges = np.arange(count+1, dtype=np.int64) * (draws+1) // count
    binned = np.add.reduceat(probabilities, edges[:-1])
    binned /= binned.sum()
    return dict(draw_count=int(draws), cell_count=int(truth.size),
        requested_bin_count=int(bins), bin_count=count,
        bin_edges=(edges.astype(float)-.5).tolist(),
        probabilities=binned.tolist(),
        uniform_probabilities=(np.diff(edges)/(draws+1)).tolist(),
        rank_probabilities=probabilities.tolist(),
        tie_cell_fraction=float(np.clip(np.sum(weights*(ties > 0))/weights.sum(), 0., 1.)),
        tie_handling='Uniform mass over exact-tie ranks; deterministic average of random tie breaking',
        rank_definition='Number of posterior draws below truth, with ranks 0 through M inclusive')
