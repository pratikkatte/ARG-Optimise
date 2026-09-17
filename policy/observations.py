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


@dataclass(frozen=True)
class _DatasetFeatures:
    geometry: np.ndarray
    observed: np.ndarray
    bit_shifts: np.ndarray


# SNPData has identity equality and immutable arrays. Values must not retain the
# key: constants disappear when their dataset does, independently of raw rows.
_DATASET_FEATURES = weakref.WeakKeyDictionary()


def _dataset_features(data):
    features = _DATASET_FEATURES.get(data)
    if features is None:
        positions, length = data.positions, int(data.sequence_length)
        left = positions - np.r_[0., positions[:-1]] if len(positions) else positions
        right = np.r_[positions[1:], length] - positions if len(positions) else positions
        geometry = np.column_stack((positions/length, left/length, right/length)).astype(np.float32)
        observed = np.array(data.genotypes.T, dtype=np.float32, order='C')
        shifts = np.arange(min(data.num_haplotypes, 64), dtype=np.uint64)
        for array in (geometry, observed, shifts):
            array.setflags(write=False)
        features = _DatasetFeatures(geometry, observed, shifts)
        _DATASET_FEATURES[data] = features
    return features


def _descendant_bits(masks, n, shifts):
    # Truncate only the decoded columns, just as the original range(n) did.
    sample_mask = (1 << n)-1
    if n <= 64:
        values = np.fromiter((mask & sample_mask for mask in masks), dtype=np.uint64, count=len(masks))
        return ((values[:, None] >> shifts) & np.uint64(1)).astype(np.float32)
    width = (n+7)//8
    raw = b''.join((mask & sample_mask).to_bytes(width, 'little') for mask in masks)
    octets = np.frombuffer(raw, dtype=np.uint8).reshape(len(masks), width)
    return np.unpackbits(octets, axis=1, bitorder='little')[:, :n].astype(np.float32)


def _static_features(env, node, constants):
    n, length = env.num_sequences, env.sequence_length
    segments = node.descendants.segments
    masks = tuple(segment[2] for segment in segments)
    decoded = _descendant_bits(masks, n, constants.bit_shifts)
    complete = np.fromiter((mask == env.all_samples for mask in masks),
                           dtype=np.float32, count=len(masks))
    left = np.fromiter((s[0] for s in segments), dtype=np.float64, count=len(segments))
    right = np.fromiter((s[1] for s in segments), dtype=np.float64, count=len(segments))
    material = np.empty((len(segments), 4+n), dtype=np.float32)
    if length <= 2**53:
        material[:, 0] = left/length
        material[:, 1] = right/length
        material[:, 2] = (right-left)/length
    else:
        # Preserve Python integer subtraction/division even for huge coordinates.
        material[:, :3] = np.asarray([(l/length, r/length, (r-l)/length)
                                      for l, r, _ in segments], dtype=np.float32).reshape(-1, 3)
    material[:, 3] = complete
    material[:, 4:] = decoded

    indices = node.snp_indices
    sites = np.empty((len(indices), 7+2*n), dtype=np.float32)
    sites[:, :2] = node.messages[:, :2]
    # Keep libm's scalar rounding and domain behavior; vectorize row assembly.
    sites[:, 2] = np.fromiter(map(math.log1p, node.messages[:, 2]),
                              dtype=np.float64, count=len(indices))
    sites[:, 3:6] = constants.geometry[indices]
    sites[:, 7:7+n] = constants.observed[indices]
    sites[:, 6] = 0
    sites[:, 7+n:] = 0
    if len(segments) and len(indices):
        positions = env.snp_data.positions[indices]
        segment_indices = np.searchsorted(left, positions, side='right')-1
        covered = (segment_indices >= 0) & (positions < right[segment_indices])
        if length > 2**53:
            # Integer/float boundary comparisons must agree with at(float(x)).
            segment_indices = np.fromiter(
                (next((i for i, (l, r, _) in enumerate(segments) if l <= float(x) < r), -1)
                 for x in positions), dtype=np.intp, count=len(positions))
            covered = segment_indices >= 0
        selected = segment_indices[covered]
        sites[covered, 6] = complete[selected]
        sites[covered, 7+n:] = decoded[selected]
    for array in (sites, material):
        array.setflags(write=False)
    return sites, material


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
    constants = _dataset_features(env.snp_data)
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
            snp_lengths.append(len(node.snp_indices))
            interval_lengths.append(len(node.descendants.segments))
            def static_features():
                return _static_features(env, node, constants)
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
        # Concatenation (or conversion of scalar lists) owns a fresh batch buffer;
        # no tensor aliases immutable cache entries or reusable scratch storage.
        array = np.asarray(rows, dtype=np.float32).reshape(-1, width)
        value = torch.from_numpy(array).to(device=device)
        if not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite neural observation; restore or diagnose the state')
        return value
    packed = PackedObservations(tensor(np.concatenate(snps), 7+2*n), tensor(np.concatenate(intervals), 4+n),
                                tuple(snp_lengths), tuple(interval_lengths),
                                tensor(lineage_rows, LINEAGE_DIM), tensor(state_rows, STATE_DIM), tuple(offsets))
    return PolicyBatch(packed, tuple(actions), tuple(rates), tuple(hazards))
