"""Independent scalar observation oracle for vectorization/cache regression tests."""
import math

import numpy as np
import torch

from env import priors
from policy.observations import PackedObservations, PolicyBatch, LINEAGE_DIM, STATE_DIM


def pack_states(env, states, device='cpu', cache=None, *, static_rows=None):
    if not states:
        raise ValueError('At least one ARG state is required')
    n, length = env.num_sequences, env.sequence_length
    positions = env.snp_data.positions
    snps, intervals, lineage_rows, state_rows = [], [], [], []
    snp_lengths, interval_lengths, offsets = [], [], [0]
    actions, rates, hazards = [], [], []
    for state in states:
        env._check_state(state)
        choices = env.enumerate_policy_actions(state)
        physical = env.enumerate_prior_options(state).rates
        total = priors.total_event_rate(physical, validate=not state.is_done)
        allowed = (float(len(choices[0])),
                   2*env.population_size*env.recombination_rate*sum(a.breakpoint_count for a in choices[1]))
        if not state.is_done and sum(allowed) <= 0:
            raise ValueError('Nonterminal state has no compatible action')
        actions.append(choices); rates.append(physical); hazards.append(allowed)
        completed = np.count_nonzero(~np.isnan(state.completed_site_lengths))
        state_rows.append([
            math.log1p(state.current_time), math.log1p(len(state.active_lineages)),
            state.total_active_blocks/length, completed/max(env.num_variants, 1) if env.num_variants else 1.,
            math.log1p(env.kappa*state.exposure), state.accumulated_log_prior/length,
            state.partial_log_likelihood/max(env.num_variants, 1), math.log1p(total),
            physical['lambda_coal']/total if total else 0.,
            physical['lambda_recomb']/total if total else 0., env.kappa*length, env.rho])
        for node in state.active_lineages:
            if node.messages is None or node.snp_indices is None:
                raise ValueError('Active lineage is missing infinite-sites messages')
            include = static_rows is None or len(lineage_rows) in static_rows
            snp_lengths.append(len(node.snp_indices) if include else 0)
            interval_lengths.append(len(node.descendants.segments) if include else 0)
            if include:
                for row, index in enumerate(node.snp_indices):
                    x = float(positions[index])
                    bits = node.descendants.at(x)
                    a, d, m = node.messages[row]
                    left = x-(positions[index-1] if index else 0.)
                    right = (positions[index+1] if index+1 < len(positions) else length)-x
                    snps.append([a, d, math.log1p(m), x/length, left/length, right/length,
                                 float(bits == env.all_samples), *env.snp_data.genotypes[:, index],
                                 *[(bits >> sample) & 1 for sample in range(n)]])
                for left, right, bits in node.descendants.segments:
                    intervals.append([left/length, right/length, (right-left)/length,
                                      float(bits == env.all_samples),
                                      *[(bits >> sample) & 1 for sample in range(n)]])
            material = node.material_segments
            lineage_rows.append([math.log1p(node.time), math.log1p(state.current_time-node.time),
                                 material.count/length,
                                 (material.segments[-1][1]-material.segments[0][0])/length,
                                 math.log1p(len(node.descendants.segments)),
                                 math.log1p(len(node.snp_indices)), float(bool(len(node.snp_indices)))])
        offsets.append(len(lineage_rows))
    def tensor(rows, width):
        array = np.asarray(rows, dtype=np.float32).reshape(-1, width)
        if not np.isfinite(array).all():
            raise FloatingPointError('Nonfinite neural observation; restore or diagnose the state')
        return torch.from_numpy(array).to(device=device)
    packed = PackedObservations(tensor(snps, 7+2*n), tensor(intervals, 4+n),
                                tuple(snp_lengths), tuple(interval_lengths),
                                tensor(lineage_rows, LINEAGE_DIM), tensor(state_rows, STATE_DIM), tuple(offsets))
    return PolicyBatch(packed, tuple(actions), tuple(rates), tuple(hazards))
