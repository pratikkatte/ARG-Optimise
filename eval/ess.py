"""Stable importance diagnostics for fresh draws from the scored proposal."""
import math
import numpy as np
from scipy.special import logsumexp


def _vector(values, name):
    if hasattr(values, 'detach'):
        values = values.detach().cpu().double().numpy()
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError(f'{name} must be a nonempty finite vector')
    return values


def log_importance_weights(log_reward, log_pf, log_pb):
    reward, pf, pb = [_vector(x, name) for x, name in
                      ((log_reward, 'log_reward'), (log_pf, 'log_pf'), (log_pb, 'log_pb'))]
    if not reward.shape == pf.shape == pb.shape:
        raise ValueError('Reward, forward and backward scores must have matching shapes')
    return _vector(reward + pb - pf, 'log_weights')


def normalized_weights(log_weights):
    values = _vector(log_weights, 'log_weights')
    # Subtract first: exp(w - logsumexp(w)) loses precision for large offsets.
    weights = np.exp(values - values.max())
    return weights / weights.sum()


def importance_stats(log_weights, *, fresh=True, reward_constant=None):
    values = _vector(log_weights, 'log_weights')
    n = len(values)
    result = dict(episodes=n, log_weight_mean=float(values.mean()),
                  log_weight_std=float(values.std(ddof=0)), ess=None,
                  ess_fraction=None, max_normalized_weight=None,
                  status='ok' if fresh else 'not_applicable_fixed_bank')
    if fresh:
        weights = normalized_weights(values)
        ess = float(1. / np.square(weights).sum())
        result.update(ess=ess, ess_fraction=ess/n,
                      max_normalized_weight=float(weights.max()))
    if reward_constant is not None:
        if not math.isfinite(float(reward_constant)):
            raise ValueError('Reward offset must be finite')
        result['log_evidence'] = (float(logsumexp(values)-math.log(n)-reward_constant)
                                  if fresh else None)
    return result


def importance(rows):
    """Legacy report adapter; rows must be fresh policy draws."""
    if not rows:
        raise ValueError('Importance evaluation requires samples')
    constants = _vector([r['reward_constant'] for r in rows], 'reward constants')
    if not np.all(constants == constants[0]):
        raise ValueError('Cannot pool samples with different reward offsets')
    stats = importance_stats([r['log_weight'] for r in rows], reward_constant=constants[0])
    return dict(ess_fraction=stats['ess_fraction'], max_weight=stats['max_normalized_weight'],
                log_weight_std=stats['log_weight_std'], log_evidence=stats['log_evidence'])
