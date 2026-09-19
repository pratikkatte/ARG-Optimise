"""Full timed-history distribution metrics using a known normalizing constant."""
import math
import numpy as np
from scipy.special import logsumexp
from scipy.stats import ks_2samp, wasserstein_distance


def mixed_mean(values_p, values_q):
    values_p, values_q = np.asarray(values_p), np.asarray(values_q)
    mean = .5 * (values_p.mean() + values_q.mean())
    se = math.sqrt(.25 * (values_p.var(ddof=1)/len(values_p) + values_q.var(ddof=1)/len(values_q)))
    return dict(estimate=float(mean), standard_error=se,
                ci95=[float(mean-1.96*se), float(mean+1.96*se)])


def distribution_metrics(reference, policy, reference_log_q, policy_log_q, log_z):
    """p is the normalized full-history target, q the unweighted learned policy.

    Expectations use independent samples from p and q with equal stratum weight.
    TV integrand |p-q|/(p+q) is bounded, and includes all history dimensions.
    The normalizing constant is computed by a separate finite-state CTMC.
    """
    def log_p(rows):
        return np.array([r['log_prior']+r['log_likelihood']-log_z for r in rows])
    p_ref, p_policy = log_p(reference), log_p(policy)
    q_ref, q_policy = np.asarray(reference_log_q), np.asarray(policy_log_q)
    if not all(np.isfinite(x).all() for x in (p_ref, p_policy, q_ref, q_policy)):
        raise ValueError('full-history comparison requires finite matched density scores')
    delta_ref, delta_policy = p_ref-q_ref, p_policy-q_policy
    tv = mixed_mean(np.abs(np.tanh(delta_ref/2)), np.abs(np.tanh(delta_policy/2)))
    def hellinger(delta):
        absolute = np.abs(delta/2)
        return 1 - 2*np.exp(-absolute)/(1+np.exp(-2*absolute))
    h2 = mixed_mean(hellinger(delta_ref), hellinger(delta_policy))
    def js(delta):
        log_a = -np.logaddexp(0., -delta)
        log_b = -np.logaddexp(0., delta)
        return np.exp(log_a)*(math.log(2)+log_a) + np.exp(log_b)*(math.log(2)+log_b)
    jensen = mixed_mean(js(delta_ref), js(delta_policy))
    log_weights = p_policy-q_policy
    weights = np.exp(log_weights-logsumexp(log_weights))
    counts = np.arange(max(r['recombinations'] for r in reference+policy)+1)
    freq_ref = np.array([sum(r['recombinations']==k for r in reference)/len(reference) for k in counts])
    freq_policy = np.array([sum(r['recombinations']==k for r in policy)/len(policy) for k in counts])
    times_ref, times_policy = np.array([r['tmrca'] for r in reference]), np.array([r['tmrca'] for r in policy])
    return dict(full_history_tv=tv, squared_hellinger=h2, jensen_shannon_nats=jensen,
        forward_kl_nats=dict(estimate=float(delta_ref.mean()), standard_error=float(delta_ref.std(ddof=1)/math.sqrt(len(delta_ref)))),
        reverse_kl_nats=dict(estimate=float(-delta_policy.mean()), standard_error=float(delta_policy.std(ddof=1)/math.sqrt(len(delta_policy)))),
        importance_ess_fraction=float(1/(weights@weights)/len(policy)),
        policy_importance_normalization=float(np.exp(logsumexp(log_weights)-math.log(len(policy)))),
        reference_samples=len(reference), policy_samples=len(policy),
        recombination_count=dict(counts=counts.tolist(), reference_probabilities=freq_ref.tolist(),
            policy_probabilities=freq_policy.tolist(), empirical_tv=float(.5*np.abs(freq_ref-freq_policy).sum())),
        tmrca=dict(units='2Ne', reference_mean=times_ref.mean(0).tolist(), policy_mean=times_policy.mean(0).tolist(),
            reference_covariance=np.cov(times_ref.T).tolist(), policy_covariance=np.cov(times_policy.T).tolist(),
            ks=[float(ks_2samp(times_ref[:,i],times_policy[:,i]).statistic) for i in (0,1)],
            wasserstein=[float(wasserstein_distance(times_ref[:,i],times_policy[:,i])) for i in (0,1)]))
