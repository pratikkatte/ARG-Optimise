"""Independent infinite-sites likelihood on completed candidate ancestry.

No JC69, incremental caches, or candidate mutation records enter this evaluator.
The score is a polarized mutation-pattern density up to fixed data-only factors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import tskit

from .snp_data import SNPData


@dataclass(frozen=True, eq=False)
class InfiniteSitesResult:
    log_likelihood: float
    exposure: float
    compatible_branch_lengths: np.ndarray
    incompatible_site_ids: tuple[int, ...]

    @property
    def zero_likelihood(self):
        return self.log_likelihood == -math.inf


def evaluate_infinite_sites(tree_sequence, data: SNPData, *, mutation_rate, sample_nodes=None,
                            time_scale=None) -> InfiniteSitesResult:
    """Score observed SNPs, returning exposure/branch lengths in generations.

    A tree sequence with time_units='2Ne' requires time_scale=2*Ne. Branch
    differences are computed before scaling to preserve very short branches.
    Mutation rates always have units per generation, regardless of input units.

    ``sample_nodes[i]`` identifies genotype row i; the default is ts.samples()
    order. An explicit mapping must be a permutation of exactly those samples.
    Exposure has units generation-bp; compatible branch lengths are generations.
    The caller must supply mutation_rate in mutations per bp per generation.

    Incomplete ancestry and unsupported inputs raise ValueError. A completed
    but incompatible topology instead returns -inf with its incompatible sites.
    """
    if not isinstance(tree_sequence, tskit.TreeSequence) or not isinstance(data, SNPData):
        raise ValueError("expected a tskit.TreeSequence and SNPData")
    ts = tree_sequence
    if ts.time_units == 'generations' and time_scale is None:
        time_scale = 1.
    elif ts.time_units != '2Ne' or time_scale is None:
        raise ValueError("candidate ancestry requires time_units='generations', or '2Ne' with explicit time_scale")
    if isinstance(time_scale, bool) or not math.isfinite(float(time_scale)) or time_scale <= 0:
        raise ValueError('time_scale must be finite and positive')
    if ts.sequence_length != data.sequence_length:
        raise ValueError("candidate and observations must have the same physical sequence_length")
    try:
        rate = float(mutation_rate)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("mutation_rate must be finite and nonnegative") from None
    if isinstance(mutation_rate, (bool, np.bool_)) or not math.isfinite(rate) or rate < 0:
        raise ValueError("mutation_rate must be finite and nonnegative")
    if ts.num_samples != data.num_haplotypes:
        raise ValueError("candidate sample count does not match genotype rows")
    try:
        samples = tuple(ts.samples()) if sample_nodes is None else tuple(sample_nodes)
    except TypeError:
        raise ValueError("sample_nodes must be an ordered sequence of sample node IDs") from None
    if (len(samples) != data.num_haplotypes
            or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in samples)
            or len(set(samples)) != len(samples) or set(samples) != set(ts.samples())):
        raise ValueError("sample_nodes must be a permutation of exactly the candidate sample nodes")
    if np.any(ts.nodes_time[list(samples)] != 0):
        raise ValueError("only contemporaneous samples at time zero are supported")

    # Python integer bitsets preserve exact descendant identity at any sample size.
    sample_bits = {int(node): 1 << row for row, node in enumerate(samples)}
    all_samples = (1 << len(samples)) - 1
    targets = [sum(1 << int(row) for row in np.flatnonzero(data.genotypes[:, col]))
               for col in range(data.num_variants)]
    lengths = np.zeros(data.num_variants, dtype=np.float64)
    interval_exposures = []
    for tree in ts.trees():
        if tree.num_roots != 1 or tree.num_samples(tree.root) != len(samples):
            raise ValueError(f"incomplete ancestry on interval [{tree.interval.left}, {tree.interval.right})")
        descendants, by_pattern, proper_branches = {}, {}, []
        for node in tree.nodes(order="postorder"):
            bits = sample_bits.get(node, 0)
            for child in tree.children(node):
                bits |= descendants[child]
            descendants[node] = bits
            # A full-sample descendant set identifies stems above the local MRCA.
            if tree.parent(node) != tskit.NULL and bits not in (0, all_samples):
                # Subtract in the original units BEFORE scaling. Scaling two
                # absolute float64 timestamps first loses short branches.
                branch = float(tree.branch_length(node)) * time_scale
                if not math.isfinite(branch) or branch <= 0:
                    raise ValueError("candidate branches must have finite positive durations")
                proper_branches.append(branch)
                by_pattern.setdefault(bits, []).append(branch)
        interval_exposures.append(tree.span * math.fsum(proper_branches))
        start, end = np.searchsorted(data.positions, [tree.interval.left, tree.interval.right], side="left")
        for index in range(int(start), int(end)):
            lengths[index] = math.fsum(by_pattern.get(targets[index], ()))

    exposure = math.fsum(interval_exposures)
    if not math.isfinite(exposure):
        raise ValueError("genomic branch exposure overflowed float64")
    incompatible = tuple(data.site_ids[i] for i in np.flatnonzero(lengths == 0))
    if incompatible or (rate == 0 and data.num_variants):
        log_likelihood = -math.inf
    else:
        # Separate logs avoid underflow in rate * very short compatible branches.
        log_likelihood = -rate * exposure
        if data.num_variants:
            log_likelihood += data.num_variants * math.log(rate) + math.fsum(map(math.log, lengths))
        if not math.isfinite(log_likelihood):
            raise ValueError("finite compatible log likelihood overflowed float64")
    lengths.setflags(write=False)
    return InfiniteSitesResult(float(log_likelihood), exposure, lengths, incompatible)
