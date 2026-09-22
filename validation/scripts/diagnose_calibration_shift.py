"""Attribute a fixed-bank log-weight variance change to policy score components.

This is an algebraic decomposition of score changes, not a causal ablation or
an importance-sampling estimate. Both inputs must score exactly the same bank.
"""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np


def diagnose(before, after):
    if before['bank_sha256'] != after['bank_sha256']:
        raise ValueError('Comparisons must use the identical held-out histories')
    a, b = before['records'], after['records']
    reward = np.array([r['log_reward'] for r in a])
    np.testing.assert_allclose(reward, [r['log_reward'] for r in b], atol=1e-10, rtol=0)
    initial = reward - np.array([r['log_policy_density'] for r in a])
    final = reward - np.array([r['log_policy_density'] for r in b])
    change = initial - final
    cov = lambda x, y: float(np.mean((x - x.mean()) * (y - y.mean())))
    components = {}
    for key in ('event', 'lineages', 'breakpoint', 'time'):
        delta = np.array([y['log_policy_factors'][key] - x['log_policy_factors'][key] for x, y in zip(a, b)])
        components[key] = dict(mean_score_change=float(delta.mean()),
            score_change_std=float(delta.std()),
            contribution_to_variance_reduction=2 * cov(initial, delta) - cov(change, delta))
    reduction = float(initial.var() - final.var())
    np.testing.assert_allclose(sum(r['contribution_to_variance_reduction'] for r in components.values()),
                               reduction, atol=1e-7, rtol=1e-9)
    lengths = np.array([r['event_count'] for r in a], dtype=float)
    correlation = lambda x: None if x.std() == 0 or lengths.std() == 0 else float(np.corrcoef(x, lengths)[0, 1])
    return dict(histories=len(a),bank_sha256=before['bank_sha256'],
        initial_log_weight_std=float(initial.std()),final_log_weight_std=float(final.std()),
        variance_reduction=reduction,initial_length_correlation=correlation(initial),
        final_length_correlation=correlation(final),components=components,
        interpretation='Positive contributions reduce fixed-bank residual variance; this is not causal attribution or ESS.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', required=True)
    parser.add_argument('--after', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with gzip.open(args.before, 'rt') as f:
        before = json.load(f)
    with gzip.open(args.after, 'rt') as f:
        after = json.load(f)
    result = diagnose(before, after)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
