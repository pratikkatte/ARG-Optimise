"""Lossless sparse observations; learned representations are never cached here."""
from dataclasses import dataclass
import math
from collections import OrderedDict
import weakref
import numpy as np
import torch
from env import priors

FEATURE_VERSION = 'infinite-sites-sparse-v1'
LINEAGE_DIM = 7
STATE_DIM = 12


class RawObservationCache:
    """Bounded cache tied to immutable message arrays, never learned tensors."""
    def __init__(self, max_bytes=64*1024*1024):
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = 0

    def get(self, env, node, compute):
        source = node.messages
        if source.flags.writeable:
            return compute()
        key = id(source)
        signature = (env.dataset_fingerprint, node.descendants.segments, tuple(node.snp_indices))
        entry = self.entries.get(key)
        if entry is not None and entry[0]() is source and entry[1] == signature:
            self.entries.move_to_end(key); self.hits += 1
            return entry[2]
        self.misses += 1
        result = compute()
        size = sum(a.nbytes for a in result)
        if key in self.entries:
            self.bytes -= self.entries.pop(key)[3]
        if size > self.max_bytes:
            return result
        while self.entries and self.bytes+size > self.max_bytes:
            _, old = self.entries.popitem(last=False); self.bytes -= old[3]
        cache_ref = weakref.ref(self)
        def discard(ref):
            cache = cache_ref()
            if cache is not None and key in cache.entries and cache.entries[key][0] is ref:
                cache.bytes -= cache.entries.pop(key)[3]
        self.entries[key] = (weakref.ref(source, discard), signature, result, size)
        self.bytes += size
        return result


@dataclass(frozen=True)
class PackedObservations:
    snps: torch.Tensor
    intervals: torch.Tensor
    snp_lengths: tuple
    interval_lengths: tuple
    lineage_scalars: torch.Tensor
    state_scalars: torch.Tensor
    state_offsets: tuple

    @property
    def counts(self):
        return tuple(b-a for a, b in zip(self.state_offsets, self.state_offsets[1:]))


@dataclass(frozen=True)
class PolicyBatch:
    observations: PackedObservations
    actions: tuple
    physical_rates: tuple
    allowed_hazards: tuple


def pack_states(env, states, device='cpu', cache=None):
    if not states:
        raise ValueError('At least one ARG state is required')
    n, length = env.num_sequences, env.sequence_length
    positions = env.snp_data.positions
    left_gaps = positions - np.r_[0., positions[:-1]] if len(positions) else positions
    right_gaps = np.r_[positions[1:], length] - positions if len(positions) else positions
    snps, intervals, lineage_rows, state_rows = [], [], [], []
    snp_lengths, interval_lengths, offsets = [], [], [0]
    actions, rates, hazards = [], [], []
    def bits(mask):
        return [(int(mask) >> k) & 1 for k in range(n)]
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
            snp_lengths.append(len(node.snp_indices))
            interval_lengths.append(len(node.descendants.segments))
            def static_features():
                site_rows, material_rows = [], []
                for index, (a, d, m) in zip(node.snp_indices, node.messages):
                    descendants = node.descendants.at(float(positions[index]))
                    site_rows.append([a, d, math.log1p(m), positions[index]/length,
                                      left_gaps[index]/length, right_gaps[index]/length,
                                      float(descendants == env.all_samples)] +
                                     bits(env.derived_sets[index]) + bits(descendants))
                for left, right, descendants in node.descendants.segments:
                    material_rows.append([left/length, right/length, (right-left)/length,
                                          float(descendants == env.all_samples)] + bits(descendants))
                result = (np.asarray(site_rows, dtype=np.float32).reshape(-1,7+2*n),
                          np.asarray(material_rows, dtype=np.float32).reshape(-1,4+n))
                for value in result:
                    value.setflags(write=False)
                return result
            site_rows, material_rows = (static_features() if cache is None else cache.get(env,node,static_features))
            snps.append(site_rows); intervals.append(material_rows)
            material = node.material_segments
            lineage_rows.append([math.log1p(node.time), math.log1p(state.current_time-node.time),
                                 material.count/length,
                                 (material.segments[-1][1]-material.segments[0][0])/length,
                                 math.log1p(len(node.descendants.segments)),
                                 math.log1p(len(node.snp_indices)), float(bool(len(node.snp_indices)))])
        offsets.append(len(lineage_rows))
    def tensor(rows, width):
        value = torch.tensor(rows, dtype=torch.float32, device=device).reshape(-1, width)
        if not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite neural observation; restore or diagnose the state')
        return value
    packed = PackedObservations(tensor(np.concatenate(snps), 7+2*n), tensor(np.concatenate(intervals), 4+n),
                                tuple(snp_lengths), tuple(interval_lengths),
                                tensor(lineage_rows, LINEAGE_DIM), tensor(state_rows, STATE_DIM), tuple(offsets))
    return PolicyBatch(packed, tuple(actions), tuple(rates), tuple(hazards))
