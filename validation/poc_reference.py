"""Independent two-locus coalescent reference for the posterior POC.

No neural policy, training reward, or environment transitions are used here.
Each locus has length one and one polarized singleton SNP. Mutation marks in
the finite-state solver integrate the likelihood over ALL history lengths.
The rejection sampler retains the original ordered, timed ARG history.
"""
from collections import defaultdict, deque
from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import solve


def terminal(lineages, all_bits):
    return all(sum(bool(x[j]) for x in lineages) == 1 and
               all(x[j] in (0, all_bits) for x in lineages) for j in (0, 1))


def transitions(lineages, link_rate):
    """Physical coalescences have rate 1; each two-locus lineage splits at c."""
    result = []
    for i in range(len(lineages)):
        for j in range(i + 1, len(lineages)):
            merged = tuple(lineages[i][k] | lineages[j][k] for k in (0, 1))
            child = [x for k, x in enumerate(lineages) if k not in (i, j)] + [merged]
            result.append((1., ('coal', i, j), child))
    if link_rate:
        for i, (left, right) in enumerate(lineages):
            if left and right:
                child = [x for k, x in enumerate(lineages) if k != i] + [(left, 0), (0, right)]
                result.append((link_rate, ('recomb', i), child))
    return result


def exact_evidence(n, kappa, link_rate, targets=(1, 2)):
    """Solve the absorbing mutation-mark CTMC in float64, without event caps.

    Killing rate kappa * proper lineage count accounts for no other mutations.
    Mark j arrives at rate kappa on the branch matching observed singleton j.
    Reaching local MRCAs with both marks present contributes one; other terminal
    states contribute zero. Repeated physical recombination cycles are included
    through the linear system, rather than enumerated or truncated histories.
    """
    if n < 2 or kappa <= 0 or link_rate < 0 or any(t not in [1 << i for i in range(n)] for t in targets):
        raise ValueError('POC requires positive mutation rate and two singleton SNPs')
    all_bits = (1 << n) - 1
    initial = tuple((1 << i, 1 << i) for i in range(n))
    states = [initial]
    index = {initial: 0}
    queue = deque(states)
    edges = []
    while queue:
        state = queue.popleft()
        aggregated = defaultdict(float)
        for rate, _, child in transitions(state, link_rate):
            child = tuple(sorted(child))
            aggregated[child] += rate
            if not terminal(child, all_bits) and child not in index:
                index[child] = len(states)
                states.append(child)
                queue.append(child)
        edges.append(aggregated)
    size = len(states)
    matrix = np.zeros((4 * size, 4 * size))
    rhs = np.zeros(4 * size)
    for i, (state, outgoing) in enumerate(zip(states, edges)):
        physical_rate = sum(outgoing.values())
        proper = sum(bits not in (0, all_bits) for x in state for bits in x)
        matching = [sum(x[j] == targets[j] for x in state) for j in (0, 1)]
        for marks in range(4):
            row = marks * size + i
            matrix[row, row] = physical_rate + kappa * proper
            for child, rate in outgoing.items():
                if terminal(child, all_bits):
                    rhs[row] += rate * (marks == 3)
                else:
                    matrix[row, marks * size + index[child]] -= rate
            for j in (0, 1):
                if not marks & (1 << j):
                    matrix[row, (marks | (1 << j)) * size + i] -= kappa * matching[j]
    solution = solve(matrix, rhs)
    residual = np.max(np.abs(matrix @ solution - rhs))
    if not 0 < solution[0] < 1 / (4 * math.e**2) or residual > 1e-10:
        raise ArithmeticError('invalid finite-state reference solution')
    return dict(evidence=float(solution[0]), log_evidence=float(math.log(solution[0])),
                physical_transient_states=size, marked_transient_states=4 * size,
                linear_system_max_residual=float(residual), condition_number=float(np.linalg.cond(matrix)),
                method='absorbing mutation-mark CTMC; all recombination counts included')


@dataclass
class ReferenceHistory:
    actions: list
    log_prior: float
    log_likelihood: float
    tmrca: list
    recombinations: int

    def as_dict(self):
        return dict(actions=self.actions, log_prior=self.log_prior, log_likelihood=self.log_likelihood,
                    tmrca=self.tmrca, recombinations=self.recombinations)


def sample_prior(rng, n, kappa, link_rate, targets=(1, 2)):
    """Direct Hudson event simulation on two loci, independent of env implementation."""
    all_bits = (1 << n) - 1
    lineages = [(1 << i, 1 << i) for i in range(n)]
    actions, lengths, roots = [], np.zeros(2), np.full(2, np.nan)
    exposure = time = log_prior = 0.
    recombinations = 0
    while not terminal(lineages, all_bits):
        choices = transitions(lineages, link_rate)
        rates = np.array([x[0] for x in choices])
        total = float(rates.sum())
        dt = float(rng.exponential(1 / total))
        if not dt > 0:
            raise ArithmeticError('nonpositive reference waiting time')
        time += dt
        exposure += dt * sum(bits not in (0, all_bits) for x in lineages for bits in x)
        lengths += dt * np.array([sum(x[j] == targets[j] for x in lineages) for j in (0, 1)])
        choice = int(np.searchsorted(rates.cumsum(), rng.random() * total, side='right'))
        rate, event, lineages = choices[choice]
        log_prior += math.log(rate) - total * dt
        action = dict(event_type=event[0], active_lineage_i=event[1], delta_t=dt, time_action=None)
        if event[0] == 'coal':
            action['active_lineage_j'] = event[2]
        else:
            action.update(material_count=2, span_start=0, span_end=1, breakpoint=1)
            recombinations += 1
        actions.append(action)
        for j in (0, 1):
            if np.isnan(roots[j]) and sum(bool(x[j]) for x in lineages) == 1:
                roots[j] = time
    likelihood = 2 * math.log(kappa) + float(np.log(lengths).sum()) - kappa * exposure
    return ReferenceHistory(actions, log_prior, likelihood, roots.tolist(), recombinations)


def rejection_sample(count, seed, n, kappa, link_rate, targets=(1, 2)):
    rng = np.random.default_rng(seed)
    log_bound = -math.log(4) - 2
    accepted = []
    proposals = 0
    while len(accepted) < count:
        history = sample_prior(rng, n, kappa, link_rate, targets)
        log_accept = history.log_likelihood - log_bound
        if log_accept > 1e-12:
            raise ArithmeticError('rejection envelope violated')
        proposals += 1
        if math.log(rng.random()) < log_accept:
            accepted.append(history.as_dict())
    return accepted, dict(accepted=count, proposals=proposals, acceptance_rate=count / proposals,
                          likelihood_bound=math.exp(log_bound), seed=seed,
                          reference='independent exact posterior rejection sampling')


def trajectory(record):
    from env.actions import CoalescenceChoice, RecombinationChoice
    from env.env import SimpleTrajectory
    result = SimpleTrajectory()
    for value in record['actions']:
        value = dict(value)
        kind = value.pop('event_type')
        cls = CoalescenceChoice if kind == 'coal' else RecombinationChoice
        result.actions.append(cls(**value))
    return result
