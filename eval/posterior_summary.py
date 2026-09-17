"""Shared tree-sequence summaries, exact span weighting, and posterior comparisons."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import torch
import tskit
from eval.interval_coverage import tmrca_interval_coverage
from eval.rank_kl import rank_kl_divergence
from eval.tmrca_ranks import tmrca_rank_histogram
from utils import load_sequences, read_fasta

# Retain the historical reference identifier for legacy report readers.
COPIED_FROM_SHA256 = '472b144be54e4b0f674baf90a946710b8ddf35af1b50fa70f31cd259b21f395c'
IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
CALIBRATION_SHA256 = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ('_calibration.py', 'tmrca_ranks.py', 'rank_kl.py', 'interval_coverage.py')}

@dataclass(frozen=True)
class TruthInterval:
    left: float
    right: float
    tcoal_2ne: float


@dataclass(frozen=True)
class PairSegment:
    pair: tuple[int, int]
    left: float
    right: float
    truth: float
    posterior_mean: float
    posterior_median: float

    @property
    def length(self) -> float:
        return self.right - self.left


def iter_pairs(nspl: int, skip: int) -> list[tuple[int, int]]:
    return [
        (s1, s2)
        for s1 in range(0, nspl - 1, skip)
        for s2 in range(s1 + 1, nspl, skip)
    ]


def _truth_value_at(
    intervals: list[TruthInterval], position: float, start_idx: int
) -> tuple[float, int]:
    idx = start_idx
    while idx < len(intervals) and position >= intervals[idx].right:
        idx += 1
    if idx >= len(intervals):
        return float("nan"), idx
    if intervals[idx].left <= position < intervals[idx].right:
        return intervals[idx].tcoal_2ne, idx
    return float("nan"), idx


def _posterior_values_at(
    trees: list[tskit.TreeSequence],
    pair: tuple[int, int],
    position: float,
    scale: float,
) -> tuple[float, ...]:
    left_idx, right_idx = pair
    vals: list[float] = []
    for ts in trees:
        samples = ts.samples()
        n = len(samples)
        if left_idx < 0 or right_idx < 0 or left_idx >= n or right_idx >= n:
            raise ValueError(
                f"Pair {pair} is out of range for tree sequence with {n} samples"
            )
        tree = ts.at(position)
        try:
            tmrca = tree.tmrca(samples[left_idx], samples[right_idx]) / scale
        except ValueError:
            tmrca = float("nan")
        vals.append(float(tmrca))
    return tuple(vals)


def combine_pair_segments(
    pair: tuple[int, int],
    truth_intervals: list[TruthInterval],
    trees: list[tskit.TreeSequence],
    ne: float,
) -> list[PairSegment]:
    sequence_length = float(trees[0].sequence_length)
    for ts in trees[1:]:
        if not math.isclose(float(ts.sequence_length), sequence_length):
            raise ValueError("Tree samples have inconsistent sequence lengths")

    truth_intervals = sorted(truth_intervals, key=lambda iv: (iv.left, iv.right))
    breakpoints = {0.0, sequence_length}
    breakpoints.update(interval.left for interval in truth_intervals)
    breakpoints.update(interval.right for interval in truth_intervals)
    for ts in trees:
        breakpoints.update(float(bp) for bp in ts.breakpoints())

    sorted_bp = sorted(bp for bp in breakpoints if 0.0 <= bp <= sequence_length)
    segments: list[PairSegment] = []
    truth_idx = 0
    scale = 2.0 * ne
    for left, right in zip(sorted_bp[:-1], sorted_bp[1:]):
        if right <= left:
            continue
        mid = (left + right) / 2.0
        truth_val, truth_idx = _truth_value_at(truth_intervals, mid, truth_idx)
        if not math.isfinite(truth_val):
            continue
        posterior_vals = _posterior_values_at(trees, pair, mid, scale)
        finite_vals = [x for x in posterior_vals if math.isfinite(x)]
        if not finite_vals:
            continue
        segments.append(
            PairSegment(
                pair=pair,
                left=left,
                right=right,
                truth=truth_val,
                posterior_mean=float(np.mean(finite_vals)),
                posterior_median=float(np.median(finite_vals)),
            )
        )
    return segments


def collect_segments_from_trees(
    *,
    truth_tracks: dict[tuple[int, int], list[TruthInterval]],
    pairs: list[tuple[int, int]],
    inferred_trees: list[tskit.TreeSequence],
    ne: float,
    verbose: bool,
) -> list[PairSegment]:
    all_segments: list[PairSegment] = []
    for pi, pair in enumerate(pairs, start=1):
        if pair not in truth_tracks:
            print(f"skip pair {pair}: no truth track", file=sys.stderr)
            continue
        if verbose:
            print(f"pair {pi}/{len(pairs)} {pair}: aligning segments ...", flush=True)
        all_segments.extend(
            combine_pair_segments(pair, truth_tracks[pair], inferred_trees, ne)
        )
    if not all_segments:
        raise RuntimeError("No aligned segments were generated.")
    return all_segments


def segments_to_dataframe(segments: list[PairSegment]) -> pd.DataFrame:
    columns = [
        "chr",
        "start",
        "end",
        "Simulated",
        "PosteriorMean",
        "PosteriorMedian",
        "len",
    ]
    if not segments:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(
        {
            "chr": "1",
            "start": [int(round(s.left)) for s in segments],
            "end": [int(round(s.right)) for s in segments],
            "Simulated": [s.truth for s in segments],
            "PosteriorMean": [s.posterior_mean for s in segments],
            "PosteriorMedian": [s.posterior_median for s in segments],
            "len": [s.length for s in segments],
        },
        columns=columns,
    )


def _finite_weighted_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (
        np.isfinite(df["Simulated"].to_numpy(dtype=float))
        & np.isfinite(df["PosteriorMean"].to_numpy(dtype=float))
        & np.isfinite(df["len"].to_numpy(dtype=float))
        & (df["len"].to_numpy(dtype=float) > 0)
    )
    return (
        df.loc[mask, "Simulated"].to_numpy(dtype=float),
        df.loc[mask, "PosteriorMean"].to_numpy(dtype=float),
        df.loc[mask, "len"].to_numpy(dtype=float),
    )


def common_metric_values(df: pd.DataFrame, legacy_mse: float) -> dict[str, float | str]:
    sim, post, weights = _finite_weighted_arrays(df)
    total_weight = float(weights.sum()) if len(weights) else 0.0
    if total_weight <= 0:
        return {
            "n_segments": int(len(df)),
            "total_length": 0.0,
            "weighted_mse": float("nan"),
            "weighted_rmse": float("nan"),
            "weighted_mae": float("nan"),
            "weighted_bias": float("nan"),
            "legacy_mseall": legacy_mse,
        }
    diff = post - sim
    mse = float(np.sum((diff**2) * weights) / total_weight)
    return {
        "n_segments": int(len(df)),
        "total_length": total_weight,
        "weighted_mse": mse,
        "weighted_rmse": math.sqrt(mse),
        "weighted_mae": float(np.sum(np.abs(diff) * weights) / total_weight),
        "weighted_bias": float(np.sum(diff * weights) / total_weight),
        "legacy_mseall": legacy_mse,
    }


def aligned_pair_times(truth, posterior, ne, truth_samples=None, posterior_samples=None):
    """Build the exact union-of-breakpoints data used by the copied reference.

    Explicit sample maps are verified before evaluation. All samples must be
    valid complete genealogies; invalid samples are counted by the caller.
    """
    if not math.isfinite(ne) or ne <= 0 or not posterior:
        raise ValueError('Positive Ne and at least one posterior sample required')
    truth_samples = list(truth.samples()) if truth_samples is None else list(truth_samples)
    n = len(truth_samples)
    if n < 2 or len(set(truth_samples)) != n or set(truth_samples) != set(truth.samples()):
        raise ValueError('Truth identity map must be a bijection over all sample nodes')
    if posterior_samples is None:
        posterior_samples = [list(ts.samples()) for ts in posterior]
    if len(posterior_samples) != len(posterior):
        raise ValueError('One sample identity map is required per posterior ARG')
    all_trees = [truth, *posterior]
    maps = [truth_samples, *posterior_samples]
    length = float(truth.sequence_length)
    for ts, sample_ids in zip(all_trees, maps):
        validate_tree_sequence(ts, sample_ids, length, n)
        if ts.time_units != 'generations' or ts.sequence_length != length:
            raise ValueError('TMRCA evaluation requires matching genomes and generation times')
        if len(sample_ids) != n or len(set(sample_ids)) != n or set(sample_ids) != set(ts.samples()):
            raise ValueError('Sample identity maps must be bijections with equal sample counts')
    boundaries = np.unique(np.concatenate([np.asarray(ts.breakpoints(as_array=True)) for ts in all_trees]))
    positions = (boundaries[1:]+boundaries[:-1])/2
    pairs = iter_pairs(n, 1)
    values = np.empty((len(all_trees), len(positions), len(pairs)), dtype=np.float64)
    for row, (ts, samples) in enumerate(zip(all_trees, maps)):
        for tree in ts.trees():
            if tree.num_roots != 1:
                raise ValueError('TMRCA evaluation requires complete marginal trees')
            first = int(np.searchsorted(positions, tree.interval.left, side='left'))
            stop = int(np.searchsorted(positions, tree.interval.right, side='left'))
            local = np.array([tree.tmrca(samples[a], samples[b])/(2*ne) for a,b in pairs])
            if not np.isfinite(local).all():
                raise ValueError('Nonfinite pairwise TMRCA')
            values[row, first:stop] = local
    return boundaries, pairs, values[0], values[1:]


def tmrca_calibration_metrics(expected, values, spans=None, rank_bins=20):
    """Shared array dispatch used by exact-span and legacy-grid evaluation."""
    ranks = tmrca_rank_histogram(expected, values, spans, bins=rank_bins)
    ranks['kl_from_uniform'] = rank_kl_divergence(ranks['probabilities'], ranks['uniform_probabilities'])
    coverage = tmrca_interval_coverage(expected, values, spans)
    metrics = dict(eval_truth_tmrca_rank_kl=ranks['kl_from_uniform'],
                   eval_truth_tmrca_rank_tie_fraction=ranks['tie_cell_fraction'])
    for interval in coverage['intervals']:
        level = round(100*interval['level'])
        metrics[f'eval_truth_interval_{level}_coverage'] = interval['coverage']
        metrics[f'eval_pair_tmrca_interval_{level}_width_mean'] = interval['mean_width']
    metrics['eval_pair_tmrca_interval_width_mean'] = metrics['eval_pair_tmrca_interval_90_width_mean']
    details = dict(rank_histogram=ranks, interval_coverage=coverage,
        weighting='Exact genomic spans and equal pairs' if spans is not None else 'Equal grid positions and pairs',
        interpretation='Single-dataset descriptive calibration; linked positions and pairs are dependent. '
                       'Uniform ranks require repeated datasets from the assumed generative model and posterior draws. '
                       'Sampling repeats from one dataset are not independent simulated datasets.',
        kl_definition='D_KL(observed binned ranks || discrete-uniform rank bin masses), natural logarithm')
    return metrics, details


def point_accuracy_metrics(truth, posterior, ne, truth_samples=None, posterior_samples=None, *, rank_bins=20):
    """Exact genomic-span weighted metrics in 2Ne units, plus spread/coverage."""
    boundaries, pairs, expected, values = aligned_pair_times(
        truth, posterior, ne, truth_samples, posterior_samples)
    means = values.mean(axis=0)
    medians = np.median(values, axis=0)
    spread = values.std(axis=0, ddof=0)
    spans = np.diff(boundaries)
    calibration_metrics, calibration = tmrca_calibration_metrics(expected, values, spans, rank_bins)
    interval_90 = calibration['interval_coverage']['intervals'][-1]
    low, high = np.asarray(interval_90['lower']), np.asarray(interval_90['upper'])
    # Match the pair-major ordering of collect_segments_from_trees exactly.
    segments = [PairSegment(pair, float(left), float(right), float(expected[j,i]),
                            float(means[j,i]), float(medians[j,i]))
                for i,pair in enumerate(pairs)
                for j,(left,right) in enumerate(zip(boundaries[:-1],boundaries[1:]))]
    frame = segments_to_dataframe(segments)
    common = common_metric_values(frame, legacy_mse=float('nan'))
    # Plot-bin-rounded legacy MSE is deliberately not used as the training metric.
    common.pop('legacy_mseall')
    weights = spans[:,None]/(spans.sum()*len(pairs))
    average = lambda x: float(np.sum(x*weights))
    metrics = dict(eval_truth_pair_tmrca_rmse=common['weighted_rmse'],
        eval_truth_pair_tmrca_mse=common['weighted_mse'],
        eval_truth_pair_tmrca_mae=common['weighted_mae'],
        eval_truth_pair_tmrca_bias=common['weighted_bias'],
        eval_pair_tmrca_sample_std_mean=average(spread),
        eval_pair_tmrca_sample_std_rms=math.sqrt(average(spread**2)),
        eval_pair_tmrca_sample_mean=average(means),
        eval_truth_pair_tmrca_mean=average(expected), **calibration_metrics)
    details = dict(method='Shared exact interval alignment and span-weighted RMSE',
        copied_source_sha256=COPIED_FROM_SHA256, time_units='2 Ne', time_divisor=2*ne,
        boundaries=boundaries.tolist(), pairs=[list(pair) for pair in pairs],
        truth=expected.tolist(), mean=means.tolist(), median=medians.tolist(),
        std=spread.tolist(), q05=low.tolist(), q95=high.tolist(), common_metrics=common,
        tmrca_calibration=calibration)
    return metrics, details, frame



def distribution_summary(values, prefix):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    result = {prefix + '_count': int(values.size), prefix + '_finite_count': int(finite.size)}
    if not finite.size:
        return {**result, **{prefix + '_' + k: None for k in
                            ('mean', 'median', 'std', 'min', 'q01', 'q05', 'q25', 'q75', 'q95', 'q99', 'max')}}
    result.update({prefix + '_mean': float(finite.mean()), prefix + '_std': float(finite.std(ddof=0))})
    for label, q in (('min', 0), ('q01', .01), ('q05', .05), ('q25', .25), ('median', .5),
                     ('q75', .75), ('q95', .95), ('q99', .99), ('max', 1)):
        result[prefix + '_' + label] = float(np.quantile(finite, q))
    return result


def topology_signature(tree, samples):
    """Rooted nontrivial clade bitsets; suppress unary nodes and ignore times/IDs."""
    indices = {int(node): i for i, node in enumerate(samples)}
    clades = set()
    for node in tree.nodes():
        leaves = list(tree.samples(node))
        if 1 < len(leaves) < len(samples):
            clades.add(sum(1 << indices[int(leaf)] for leaf in leaves))
    return tuple(sorted(clades))


def rooted_rf(first, second, sample_count):
    return len(set(first) ^ set(second)) / max(2 * (sample_count - 2), 1)


def mean_pairwise_rf(signatures, sample_count):
    """Exact mean across unordered pairs, computed from clade frequencies."""
    n = len(signatures)
    if n < 2:
        return None
    frequencies = Counter(clade for signature in signatures for clade in signature)
    differences = sum(count * (n - count) for count in frequencies.values())
    return differences / (n * (n - 1) / 2) / max(2 * (sample_count - 2), 1)


def topology_segments(ts, samples):
    return [(float(t.interval.left), float(t.interval.right), topology_signature(t, samples)) for t in ts.trees()]


def genome_rf(first, second, sample_count, length):
    """Exact span-weighted distance on the union of both breakpoint sets."""
    i = j = 0
    total = covered = 0.
    while i < len(first) and j < len(second):
        left = max(first[i][0], second[j][0])
        right = min(first[i][1], second[j][1])
        if right > left:
            total += (right - left) * rooted_rf(first[i][2], second[j][2], sample_count)
            covered += right - left
        a, b = first[i][1], second[j][1]
        i += a <= b
        j += b <= a
    if not np.isclose(covered, length):
        raise ValueError('Tree intervals do not cover the complete genome')
    return total / length


class TerminalSamplingEvaluator:
    """Reuse terminal rollout states; never sample actions or update any model."""

    def __init__(self, truth, truth_samples, sample_names, population_size, grid_size=100,
                 provenance=None, tmrca_method='grid'):
        if tmrca_method not in ('grid', 'point_accuracy'):
            raise ValueError('Unknown TMRCA evaluation method')
        self.tmrca_method = tmrca_method
        if truth.time_units != 'generations':
            raise ValueError('Truth tree times must explicitly be in generations')
        if grid_size < 1 or population_size <= 0:
            raise ValueError('Positive grid size and population size required')
        self.truth = truth
        self.truth_samples = list(map(int, truth_samples))
        self.n = len(self.truth_samples)
        if self.n < 2 or len(set(self.truth_samples)) != self.n or set(self.truth_samples) != set(truth.samples()):
            raise ValueError('Truth sample identities must be a bijection over all truth samples')
        if len(sample_names) != self.n or len(set(sample_names)) != self.n:
            raise ValueError('Sample names must be unique and match the sample count')
        self.scale = 2 * population_size
        if any(tree.num_roots != 1 for tree in truth.trees()):
            raise ValueError('Truth must have complete marginal trees')
        if np.any(truth.tables.nodes.time[self.truth_samples] != 0):
            raise ValueError('Expected contemporary truth samples')
        self.positions = (np.arange(grid_size) + .5) * truth.sequence_length / grid_size
        self.pairs = list(itertools.combinations(range(self.n), 2))
        self.expected = self.pair_times(truth, self.truth_samples)
        self.truth_segments = topology_segments(truth, self.truth_samples)
        self.protocol = dict(
            version=1, grid_positions=self.positions.tolist(), grid_weights=[1/grid_size]*grid_size,
            grid_definition='Midpoints of equal-width genomic bins; same grid at all checkpoints',
            sample_names=list(sample_names), gfn_sample_nodes=list(range(self.n)),
            truth_sample_nodes=self.truth_samples, pair_indices=[list(p) for p in self.pairs],
            input_time_units='generations', reported_time_units='2 Ne', time_divisor=self.scale,
            tmrca_definition='Equal genomic-grid and haplotype-pair weights; unweighted fresh-policy samples',
            sample_spread_definition='Empirical population standard deviation (ddof=0); linear quantiles',
            interval_definition='Descriptive 5th–95th sample percentiles, not a calibrated posterior guarantee',
            rf_definition='Nontrivial rooted clade symmetric difference / (2*(n-2)); unary nodes suppressed',
            truth_rf_integration='Exact genomic span weights on the union of tree breakpoints',
            diversity_definition='Topology frequencies and exact mean pairwise local RF on the fixed grid',
            **(provenance or {}))
        if tmrca_method == 'point_accuracy':
            self.protocol.update(version=2, tmrca_method=tmrca_method,
                                 tmrca_definition='Exact genomic span and haplotype-pair weights; unweighted fresh-policy samples',
                                 tmrca_implementation='eval/posterior_summary.py',
                                 tmrca_copied_reference_sha256=COPIED_FROM_SHA256)
        self.protocol['metric_implementation'] = 'eval/posterior_summary.py'
        self.protocol['metric_implementation_sha256'] = IMPLEMENTATION_SHA256
        self.protocol['calibration_implementation_sha256'] = CALIBRATION_SHA256
        self.protocol['calibration_definition'] = dict(rank_bins=20, ranks='0 through draw count inclusive',
            ties='Uniform mass over exact-tie ranks', kl='observed || uniform rank bin mass, in nats',
            coverage_levels=[.5, .7, .9], quantile_method='linear', endpoints='inclusive')
        self.protocol['sha256'] = hashlib.sha256(json.dumps(self.protocol, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dataset(cls, dataset_path, env, grid_size=100, tmrca_method='grid'):
        from env.snp_data import load_snp_dataset
        dataset_path = Path(dataset_path)
        metadata = json.loads((dataset_path/'metadata.json').read_text())
        observed = load_snp_dataset(dataset_path)
        if (observed.haplotype_ids != env.snp_data.haplotype_ids
                or not np.array_equal(observed.positions, env.snp_data.positions)
                or not np.array_equal(observed.genotypes, env.snp_data.genotypes)
                or observed.sequence_length != env.sequence_length):
            raise ValueError('Evaluation observations differ from the model environment')
        truth_path = dataset_path/metadata['files']['ground_truth_trees']
        truth = tskit.load(truth_path)
        samples = metadata['sample_nodes_in_haplotype_order']
        if len(samples) != env.num_sequences or truth.sequence_length != env.sequence_length:
            raise ValueError('Truth dimensions differ from observations')
        variants = {v.site.id: (v.site.position, v.site.ancestral_state, v.genotypes.copy(), tuple(v.alleles))
                    for v in truth.variants(samples=samples)}
        for j, site_id in enumerate(observed.site_ids):
            position, ancestral, genotypes, alleles = variants[site_id]
            if (position != observed.positions[j] or ancestral != observed.ancestral_states[j]
                    or not np.array_equal(genotypes, observed.genotypes[:, j])
                    or alleles[1] != observed.derived_states[j]):
                raise ValueError('Truth sample mapping does not reproduce observed SNPs')
        return cls(truth, samples, observed.haplotype_ids, env.population_size, grid_size,
                   dict(environment_fingerprint=env.dataset_fingerprint,
                        truth_sha256=hashlib.sha256(truth_path.read_bytes()).hexdigest(),
                        verified_exported_sites=observed.num_variants), tmrca_method=tmrca_method)

    def pair_times(self, ts, samples):
        return np.array([[tree.tmrca(samples[a], samples[b])/self.scale for a, b in self.pairs]
                         for position in self.positions for tree in [ts.at(position)]])

    def summarize_trees(self, tree_sequences):
        nvalid = len(tree_sequences)
        if not nvalid:
            return {}, {}
        signatures, times, truth_distances, tree_counts = [], [], [], []
        for ts in tree_sequences:
            samples = ts.samples()
            times.append(self.pair_times(ts, samples))
            cache, grid = {}, []
            for position in self.positions:
                tree = ts.at(position)
                if tree.index not in cache:
                    cache[tree.index] = topology_signature(tree, samples)
                grid.append(cache[tree.index])
            signatures.append(grid)
            truth_distances.append(genome_rf(topology_segments(ts, samples), self.truth_segments,
                                             self.n, self.truth.sequence_length))
            tree_counts.append(ts.num_trees)
        values = np.asarray(times)
        mean, spread = values.mean(0), values.std(0, ddof=0)
        low, high = np.quantile(values, [.05, .95], axis=0)
        catalog = {}
        def key(signature):
            name = ','.join(map(str, signature)) or 'star'
            catalog[name] = list(signature)
            return name
        grid_details, richness, modal, pairwise, entropies = [], [], [], [], []
        for col, position in enumerate(self.positions):
            local = [row[col] for row in signatures]
            counts = Counter(local)
            frequency = np.array(list(counts.values()))/nvalid
            entropy = float(-np.sum(frequency*np.log(frequency)))
            distance = mean_pairwise_rf(local, self.n)
            richness.append(len(counts)); modal.append(float(frequency.max()))
            entropies.append(entropy)
            if distance is not None:
                pairwise.append(distance)
            grid_details.append(dict(position=float(position), unique_topologies=len(counts),
                                     mean_pairwise_rf=distance,
                                     frequencies={key(sig):dict(count=count, frequency=count/nvalid)
                                                  for sig, count in sorted(counts.items())}))
        path_counts = Counter(tuple(row) for row in signatures)
        truth_grid = [topology_signature(self.truth.at(position), self.truth_samples) for position in self.positions]
        metrics = dict(
            eval_topology_unique_grid_paths=len(path_counts),
            eval_topology_unique_local_mean=float(np.mean(richness)),
            eval_topology_unique_local_min=int(min(richness)), eval_topology_unique_local_max=int(max(richness)),
            eval_topology_modal_frequency_mean=float(np.mean(modal)),
            eval_topology_entropy_mean=float(np.mean(entropies)),
            eval_pairwise_local_rf_mean=float(np.mean(pairwise)) if pairwise else None,
            eval_truth_pair_tmrca_rmse=float(np.sqrt(np.mean((mean-self.expected)**2))),
            eval_pair_tmrca_sample_std_mean=float(spread.mean()),
            eval_pair_tmrca_sample_std_rms=float(np.sqrt(np.mean(spread**2))),
            eval_truth_rooted_rf_mean=float(np.mean(truth_distances)),
            eval_truth_rooted_rf_grid_mean=float(np.mean([
                rooted_rf(sig, target, self.n) for row in signatures for sig, target in zip(row, truth_grid)])),
            eval_pair_tmrca_sample_mean=float(values.mean()), eval_truth_pair_tmrca_mean=float(self.expected.mean()),
            **distribution_summary(tree_counts, 'eval_marginal_tree_count'))
        details = dict(topology_catalog=catalog, topology_grid=grid_details,
                       grid_path_frequencies=[dict(signatures=[key(s) for s in path], count=count,
                                                   frequency=count/nvalid) for path, count in path_counts.items()],
                       pair_tmrca_grid=dict(truth=self.expected.tolist(), mean=mean.tolist(),
                                           std=spread.tolist(), q05=low.tolist(), q95=high.tolist()),
                       per_arg_truth_rooted_rf=truth_distances, per_arg_marginal_tree_counts=tree_counts)
        if self.tmrca_method == 'point_accuracy':
            exact, point_details, _ = point_accuracy_metrics(
                self.truth, tree_sequences, self.scale / 2,
                truth_samples=self.truth_samples,
                posterior_samples=[list(ts.samples()) for ts in tree_sequences])
            metrics.update(exact)
            details['pair_tmrca_exact'] = point_details
        else:
            calibration_metrics, calibration = tmrca_calibration_metrics(self.expected, values)
            metrics.update(calibration_metrics)
            details['tmrca_calibration'] = calibration
        return metrics, details

    @torch.no_grad()
    def __call__(self, env, outputs, trajectories):
        states = outputs['states']
        trees, valid_indices, invalid = [], [], Counter()
        for index, state in enumerate(states):
            try:
                if not state.is_done or not np.isfinite(state.log_reward):
                    raise ValueError('Nonterminal state or nonfinite reward')
                ts = env.save_to_tree_sequence(state)
                if ts.time_units != 'generations' or ts.sequence_length != self.truth.sequence_length:
                    raise ValueError('Sample time units or sequence length mismatch')
                if not np.array_equal(ts.samples(), np.arange(self.n)):
                    raise ValueError('Exported sample identity/order mismatch')
                if not np.isfinite(ts.tables.nodes.time).all() or np.any(ts.tables.nodes.time[ts.samples()] != 0):
                    raise ValueError('Invalid node times or noncontemporary tips')
                if any(tree.num_roots != 1 for tree in ts.trees()):
                    raise ValueError('Incomplete marginal tree')
                trees.append(ts); valid_indices.append(index)
            except (ValueError, tskit.LibraryError) as exc:
                raise ValueError('Invalid evaluation draw; no samples may be discarded') from exc
        metrics, details = self.summarize_trees(trees)
        recombinations = [sum(a.event_type=='recomb' for a in traj.actions) for traj in trajectories]
        rewards = outputs['log_rewards'].detach().cpu().numpy()
        metrics.update(distribution_summary(rewards, 'eval_log_reward'))
        metrics.update(distribution_summary(recombinations, 'eval_recombination_count'))
        metrics.update(eval_terminal_sample_count=len(states), eval_valid_arg_count=len(trees),
                       eval_invalid_arg_count=len(states)-len(trees), eval_valid_arg_fraction=len(trees)/len(states))
        details.update(protocol=self.protocol, valid_sample_indices=valid_indices, invalid_reasons=dict(invalid),
                       log_rewards=rewards.tolist(), recombination_counts=recombinations)
        return metrics, details


def validate_tree_sequence(ts, sample_ids, length, sample_count):
    """Validate units and a caller-supplied identity mapping; never guess identities."""
    if ts.time_units != 'generations' or ts.sequence_length != length:
        raise ValueError('Tree sequences require matching sequence lengths and generation times')
    if (len(sample_ids) != sample_count or len(set(sample_ids)) != sample_count
            or set(sample_ids) != set(ts.samples())):
        raise ValueError('Sample identity maps must be bijections over all sample nodes')
    if not np.isfinite(ts.tables.nodes.time).all() or np.any(ts.tables.nodes.time[list(sample_ids)] != 0):
        raise ValueError('Tree times must be finite with contemporary samples')
    if any(tree.num_roots != 1 for tree in ts.trees()):
        raise ValueError('Tree sequences require complete single-root marginal trees')


def posterior_covariance(values, positions, lags=(0, 1, 2, 4, 8, 16, 32, 64)):
    """Posterior covariance across draws, not spatial covariance of posterior means."""
    values = np.asarray(values, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if (positions.ndim != 1 or not len(positions) or not np.isfinite(positions).all()
            or np.any(np.diff(positions) <= 0) or values.ndim != 3
            or values.shape[1] != len(positions) or not values.shape[0] or not values.shape[2]
            or not np.isfinite(values).all()):
        raise ValueError('Covariance requires finite [draw, position, pair] values')
    if len(values) < 2:
        return dict(status='requires_multiple_draws', curves=[])
    centered = values-values.mean(axis=0, keepdims=True)
    rows = []
    for lag in lags:
        if not math.isfinite(lag) or lag < 0 or int(lag) != lag:
            raise ValueError('Covariance lags must be nonnegative integers')
        lag = int(lag)
        if lag >= len(positions):
            continue
        width = len(positions)-lag
        covariance = np.sum(centered[:, :width]*centered[:, lag:], axis=0)/(len(values)-1)
        distance = float(np.mean(positions[lag:]-positions[:width]))
        rows.append(dict(lag=int(lag), distance_bp=distance, covariance=float(covariance.mean())))
    return dict(status='ok', definition='Sample covariance across draws, ddof=1; average over positions and haplotype pairs',
                units='(2 Ne)^2', curves=rows)


def summarize_ensemble(trees, sample_maps, ne, grid_size=100):
    """Reusable grid features and a medoid chosen without simulated truth."""
    if not trees or len(trees) != len(sample_maps):
        raise ValueError('Provide at least one tree sequence and one identity map per draw')
    if not math.isfinite(ne) or ne <= 0 or grid_size < 1:
        raise ValueError('Positive finite Ne and grid size are required')
    n, length = len(sample_maps[0]), trees[0].sequence_length
    if n < 2:
        raise ValueError('At least two haplotypes are required')
    positions = (np.arange(grid_size)+.5)*length/grid_size
    pairs = list(itertools.combinations(range(n), 2))
    times = np.empty((len(trees), grid_size, len(pairs)), dtype=np.float64)
    signatures = []
    for row, (ts, samples) in enumerate(zip(trees, sample_maps)):
        validate_tree_sequence(ts, samples, length, n)
        local, cache = [], {}
        for col, position in enumerate(positions):
            tree = ts.at(position)
            if tree.index not in cache:
                cache[tree.index] = (topology_signature(tree, samples),
                    [tree.tmrca(samples[a], samples[b])/(2*ne) for a, b in pairs])
            sig, times[row, col] = cache[tree.index]
            local.append(sig)
        signatures.append(local)
    frequencies = [Counter(c for row in signatures for c in row[col]) for col in range(grid_size)]
    medoid_distances = np.zeros(len(trees))
    if len(trees) > 1:
        for row, sigs in enumerate(signatures):
            medoid_distances[row] = sum(
                sum(counts.values()) + len(sig)*len(trees)-2*sum(counts[c] for c in sig)
                for sig, counts in zip(sigs, frequencies)) / (
                    grid_size*(len(trees)-1)*max(2*(n-2), 1))
    return dict(positions=positions, pairs=pairs, times=times, signatures=signatures,
                sample_count=n, draw_count=len(trees), sequence_length=length,
                medoid_index=int(np.argmin(medoid_distances)),
                medoid_mean_grid_rf=float(medoid_distances.min()),
                covariance=posterior_covariance(times, positions))


def compare_ensembles(first, reference):
    """Clade probability and TMRCA marginal agreement on the same genomic grid."""
    if (first['sample_count'] != reference['sample_count'] or first['pairs'] != reference['pairs']
            or not np.array_equal(first['positions'], reference['positions'])):
        raise ValueError('Posterior comparison requires matching haplotypes and genomic grids')
    squared, wasserstein = [], []
    for col in range(len(first['positions'])):
        a = Counter(c for row in first['signatures'] for c in row[col])
        b = Counter(c for row in reference['signatures'] for c in row[col])
        clades = set(a) | set(b)
        squared.append(float(np.mean([(a[c]/first['draw_count']-b[c]/reference['draw_count'])**2
                                      for c in clades])) if clades else 0.)
        wasserstein.append([float(wasserstein_distance(first['times'][:, col, pair],
                            reference['times'][:, col, pair])) for pair in range(len(first['pairs']))])
    return dict(clade_probability_rmse=math.sqrt(float(np.mean(squared))),
                pair_tmrca_wasserstein=float(np.mean(wasserstein)),
                per_position_clade_mse=squared,
                per_position_pair_tmrca_wasserstein=wasserstein,
                time_units='2 Ne', clade_definition='Union of observed nontrivial rooted clades per position; equal position weights')


def ensemble_truth_metrics(trees, sample_maps, truth, truth_samples, ne, features=None, *, rank_bins=20):
    metrics, details, _ = point_accuracy_metrics(truth, trees, ne, truth_samples, sample_maps, rank_bins=rank_bins)
    truth_segments = topology_segments(truth, truth_samples)
    distances = [genome_rf(topology_segments(ts, samples), truth_segments,
                          len(truth_samples), truth.sequence_length) for ts, samples in zip(trees, sample_maps)]
    if features is None:
        features = summarize_ensemble(trees, sample_maps, ne)
    metrics.update(eval_truth_rooted_rf_mean=float(np.mean(distances)),
                   eval_truth_rooted_rf_medoid=float(distances[features['medoid_index']]))
    return metrics, dict(**details, per_arg_truth_rooted_rf=distances,
                         medoid_index=features['medoid_index'],
                         medoid_selection='Minimum mean grid RF to ensemble; ties use first draw')
