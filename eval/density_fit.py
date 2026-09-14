"""Full-history density calibration and deterministic evaluation-bank selection."""
from collections import Counter, defaultdict, deque
import math
import random
import numpy as np

def fit_stats(x, y, intercept=None):
    """Free regression and slope-one residuals, sharing one intercept across strata."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError('Density coordinates must be matching one-dimensional arrays')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Density coordinates must be finite')
    if intercept is not None and not math.isfinite(float(intercept)):
        raise ValueError('Density intercept must be finite')
    n = len(x)
    if not n:
        return dict(count=0, finite_count=0, status='empty', pearson=None, slope=None,
                    free_intercept=None, slope_one_intercept=intercept, rmse=None,
                    residual_mean=None, residual_std=None)
    a = float(np.mean(y-x)) if intercept is None else float(intercept)
    residual = y-x-a
    dx, dy = x-x.mean(), y-y.mean()
    vx, vy = float(dx@dx), float(dy@dy)
    slope = float(dx@dy/vx) if n >= 2 and vx > 0 else None
    pearson = float(np.clip(dx@dy/math.sqrt(vx*vy), -1., 1.)) if n >= 3 and vx > 0 and vy > 0 else None
    return dict(count=n, finite_count=n,
        status='ok' if pearson is not None else 'insufficient_or_zero_variance',
        pearson=pearson, slope=slope,
        free_intercept=float(y.mean()-slope*x.mean()) if slope is not None else None,
        slope_one_intercept=a, rmse=float(np.sqrt(np.mean(residual**2))),
        residual_mean=float(residual.mean()), residual_std=float(residual.std(ddof=0)))


def density_summary(rows):
    result = {}
    for name, xkey in [('raw', 'log_reward'), ('prior_relative', 'log_likelihood')]:
        x = np.array([r[xkey] for r in rows], dtype=np.float64)
        y = np.array([r['log_policy_density'] - r.get('log_backward_probability', 0.)
                      - (r['log_prior'] if name == 'prior_relative' else 0.) for r in rows])
        whole = fit_stats(x, y)
        groups = {}
        for key in ('stratum', 'event_count', 'recombinations'):
            groups[key] = {str(value): fit_stats(x[mask], y[mask], whole['slope_one_intercept'])
                for value in sorted({r.get(key, 'unstratified') for r in rows}, key=str)
                for mask in [np.array([r.get(key, 'unstratified') == value for r in rows])]}
        result[name] = dict(global_fit=whole, groups=groups)
    if rows:
        offsets = np.array([r['log_reward']-r['log_likelihood']-r['log_prior'] for r in rows])
        if not np.allclose(offsets, offsets[0], atol=1e-7, rtol=0):
            raise ValueError('Density rows must share one target reward offset')
        np.testing.assert_allclose(result['raw']['global_fit']['rmse'],
                                   result['prior_relative']['global_fit']['rmse'], atol=1e-7)
    return result


def select_bank(candidates, size, forbidden):
    if size < 1:
        raise ValueError('Bank histories per stratum must be positive')
    unique = {}
    overlaps = []
    for row in candidates:
        if row['fingerprint'] in forbidden:
            overlaps.append(row['fingerprint'])
        else:
            unique.setdefault(row['fingerprint'], row)
    ordered = sorted(unique.values(), key=lambda r: (r['log_reward'], r['fingerprint']))
    bank, strata = [], {}
    rng = random.Random(91919191)
    for name, indices in zip(('low', 'medium', 'high'), np.array_split(np.arange(len(ordered)), 3)):
        population = [ordered[int(i)] for i in indices]
        if len(population) < size:
            raise ValueError(f'Insufficient unique candidates: {name} has {len(population)} histories, '
                             f'but bank_per_stratum={size}. Reduce bank_per_stratum or increase bank_candidates.')
        buckets = defaultdict(list)
        for row in population:
            buckets[(row['topology_sha256'], row['recombinations'], row['event_count'])].append(row)
        keys = sorted(buckets)
        rng.shuffle(keys)
        for values in buckets.values():
            rng.shuffle(values)
        queue, selected = deque(keys), []
        while len(selected) < size:
            key = queue.popleft()
            selected.append(dict(buckets[key].pop(), stratum=name))
            if buckets[key]:
                queue.append(key)
        bank.extend(selected)
        strata[name] = dict(candidate_count=len(population), count=len(selected),
            minimum_log_reward=population[0]['log_reward'], maximum_log_reward=population[-1]['log_reward'],
            minimum_fingerprint=population[0]['fingerprint'],
            topology_count=len({r['topology_sha256'] for r in selected}),
            recombination_counts=dict(Counter(str(r['recombinations']) for r in selected)),
            event_counts=dict(Counter(str(r['event_count']) for r in selected)),
            sources=dict(Counter(r['provenance']['source'] for r in selected)))
    return dict(records=bank, strata=strata, overlap_fingerprints=overlaps,
                candidate_count=len(candidates), unique_eligible_count=len(ordered),
                duplicates=len(candidates)-len(overlaps)-len(ordered))
