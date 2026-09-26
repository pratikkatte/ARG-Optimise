#!/usr/bin/env python3
"""Validate saved, trusted ARGinfer pickles without running inference."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
import tempfile
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'arginfer_eval_matplotlib'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(Path(tempfile.gettempdir()) / 'arginfer_eval_numba'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'arginfer_eval_cache'))

import numpy as np
import pandas as pd
import tskit

from eval.tmrca_ranks import tmrca_rank_histogram
from eval.rank_kl import rank_kl_divergence

LEVELS = (.5, .7, .9, .95)
GLOBAL_TRACES = ('log_likelihood', 'log_prior', 'log_posterior', 'recombinations',
                 'marginal_trees', 'mean_pair_tmrca', 'mean_root_time',
                 'mean_total_branch_length')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + '\n')


def inventory(directory, burnin=0, expected_thin=1000):
    files = []
    for path in directory.glob('*.arg'):
        # ARGinfer overwrites this scratch snapshot while proposing MCMC moves.
        # It is not an iteration-indexed posterior draw.
        if path.name == 'arg.arg':
            continue
        match = re.fullmatch(r'arg(\d+)\.arg', path.name)
        require(match is not None, f'Unrecognized ARG filename: {path}')
        files.append((int(match[1]), path))
    files.sort()
    require(len(files) > burnin,
            'No retained numbered ARG files after burn-in; --arg-dir must point to the job directory')
    iterations = [index for index, _ in files]
    require(len(set(iterations)) == len(iterations), 'Duplicate saved iterations')
    require(all(b-a == expected_thin for a, b in zip(iterations, iterations[1:])),
            f'Saved iteration gap differs from --expected-thin {expected_thin}')
    return files[burnin:], len(files)


def arg_to_ts(arg, n, length):
    """Emit ancestral edges, splitting a recombination child's material at its breakpoint.

    Roots have no stored segments; their children's segments still emit incoming
    edges. Suppressing unary recombination nodes gives marginal genealogies.
    No mutation coordinates or branch times are rounded or rescaled.
    """
    nodes = arg.nodes
    leaves = sorted(k for k, v in nodes.items() if v.left_child is None)
    require(leaves == list(range(n)), 'ARG leaves must be input haplotype rows 0..n-1')
    tables = tskit.TableCollection(length)
    tables.time_units = 'generations'
    mapping = {}
    for key in sorted(nodes):
        node = nodes[key]
        require(key == node.index, 'Node dictionary identity mismatch')
        require(math.isfinite(node.time) and node.time >= 0, 'Invalid node time')
        require(key not in leaves or node.time == 0, 'Noncontemporary ARG sample')
        mapping[key] = tables.nodes.add_row(flags=int(key in leaves), time=node.time)
    for key, node in nodes.items():
        left, right = node.left_parent, node.right_parent
        if left is None:
            require(right is None and node.first_segment is None, 'Invalid ARG root')
            continue
        require(right is not None and node.first_segment is not None, 'Missing parent/material')
        for parent in (left, right):
            require(nodes.get(parent.index) is parent, 'Dangling parent reference')
            require(parent.time > node.time, 'Nonpositive branch length')
            require(parent.left_child is node or parent.right_child is node,
                    'Parent/child references disagree')
        split = left.index != right.index
        if split:
            require(left.time == right.time and node.breakpoint is not None,
                    'Invalid recombination parents')
            require(0 < node.breakpoint < length, 'Invalid recombination breakpoint')
        else:
            require(node.breakpoint is None, 'Breakpoint without two parents')
        seg = node.first_segment
        seen, last_right, previous = set(), 0., None
        while seg is not None:
            require(id(seg) not in seen, 'Cyclic segment list')
            seen.add(id(seg))
            require(seg.prev is previous and seg.node is node, 'Invalid segment links')
            require(0 <= last_right <= seg.left < seg.right <= length,
                    'Overlapping or out-of-range ancestral segments')
            intervals = [(seg.left, seg.right, left)]
            if split:
                intervals = [(seg.left, min(seg.right, node.breakpoint), left),
                             (max(seg.left, node.breakpoint), seg.right, right)]
            for a, b, parent in intervals:
                if b > a:
                    tables.edges.add_row(a, b, mapping[parent.index], mapping[key])
            last_right, previous, seg = seg.right, seg, seg.next
    tables.sort()
    raw = tables.tree_sequence()
    samples = [mapping[i] for i in range(n)]
    for tree in raw.trees():
        require(tree.num_roots == 1, f'Incomplete marginal tree at {tree.interval}')
    ts = raw.simplify(samples=samples, keep_unary=False)
    return ts, raw, mapping


def signature(tree, samples):
    indices = {int(node): i for i, node in enumerate(samples)}
    bits, clades = {}, set()
    full = (1 << len(samples)) - 1
    for node in tree.nodes(order='postorder'):
        value = 1 << indices[node] if node in indices else 0
        for child in tree.children(node):
            value |= bits[child]
        bits[node] = value
        if value != full and value.bit_count() > 1:
            clades.add(value)
    return tuple(sorted(clades))


@dataclass
class Features:
    boundaries: np.ndarray
    times: np.ndarray  # pair times followed by root time; generations
    branch: np.ndarray
    topologies: list

    def at(self, positions):
        return np.searchsorted(self.boundaries[1:], positions, side='right')


def extract_features(ts, samples=None):
    samples = list(ts.samples()) if samples is None else list(samples)
    require(set(samples) == set(ts.samples()) and len(samples) == ts.num_samples,
            'Sample map must be a bijection')
    require(ts.time_units == 'generations', 'Tree times must be in generations')
    # Compare sampled genealogies, not representation-specific ARG scaffolding.
    # GFN can retain unary ancestry above the local sample MRCA. Such nodes must
    # not contribute to root TMRCA, total branch length, or marginal-tree counts.
    ts = ts.simplify(samples=samples, keep_unary=False, keep_input_roots=False)
    samples = list(ts.samples())
    pairs = list(itertools.combinations(samples, 2))
    times, branch, topologies = [], [], []
    for tree in ts.trees():
        require(tree.num_roots == 1, 'Incomplete tree sequence')
        times.append([tree.tmrca(a, b) for a, b in pairs] + [tree.time(tree.root)])
        branch.append(tree.total_branch_length)
        topologies.append(signature(tree, samples))
    return Features(np.asarray(list(ts.breakpoints())), np.asarray(times),
                    np.asarray(branch), topologies)


def validate_inputs(dataset, input_dir):
    from validation.scripts.prepare_arginfer_inputs import convert
    metadata = json.loads((dataset / 'metadata.json').read_text())
    name = metadata['dataset_name']
    expected, n, _ = convert(dataset, name)
    manifest = json.loads((input_dir / 'manifest.json').read_text())
    for filename in ('haplotypes.txt', 'ancestral.txt', 'positions.txt', 'samples.tsv'):
        require((input_dir / filename).read_text() == expected[filename],
                f'ARGinfer input does not match dataset: {filename}')
    expected_manifest = json.loads(expected['manifest.json'])
    for key in ('source_sha256', 'num_haplotypes', 'num_snps', 'sequence_length',
                'population_size', 'mutation_rate', 'recombination_rate', 'coordinates'):
        require(manifest[key] == expected_manifest[key], f'Input manifest mismatch: {key}')
    haplotypes = [line.split() for line in expected['haplotypes.txt'].splitlines()]
    ancestral = expected['ancestral.txt'].splitlines()
    positions = [int(x) for x in expected['positions.txt'].splitlines()]
    data = {pos: {i for i in range(n) if haplotypes[i][j] != ancestral[j]}
            for j, pos in enumerate(positions)}
    truth_path = dataset / metadata['files']['ground_truth_trees']
    truth = tskit.load(truth_path)
    require(truth.num_samples == n and truth.sequence_length == manifest['sequence_length'],
            'Truth dimensions do not match ARGinfer input')
    sample_map = metadata['sample_nodes_in_haplotype_order']
    extract_features(truth, sample_map)  # validate explicit identity order and times
    # Verify truth genotypes in the declared haplotype order, at exact truth sites.
    variants = list(truth.variants(samples=sample_map))
    require(len(variants) == len(positions), 'Truth/input mutation count differs')
    for j, variant in enumerate(variants):
        alleles = [variant.alleles[g] for g in variant.genotypes]
        require(alleles == [h[j] for h in haplotypes] and
                variant.site.ancestral_state == ancestral[j], 'Truth/input haplotype identity mismatch')
    return metadata, manifest, truth, data


def check_mutations(arg, raw, mapping, data, n):
    assigned = {}
    sample_lookup = {mapping[i]: i for i in range(n)}
    for key, node in arg.nodes.items():
        for position in node.snps:
            require(position in data and position not in assigned, 'Missing/duplicate/unknown stored SNP')
            tree = raw.at(position)
            descendants = {sample_lookup[x] for x in tree.samples(mapping[key])}
            require(descendants == data[position], f'Mutation descendants disagree at {position}')
            assigned[position] = key
    require(set(assigned) == set(data), 'Stored mutations do not cover all observed SNPs')


def tree_log_likelihood(ts, data, mutation_rate):
    """Independent infinite-sites score, using all branches compatible with each SNP."""
    total = sum(tree.span*tree.total_branch_length for tree in ts.trees())
    result = -mutation_rate*total + len(data)*math.log(mutation_rate)
    samples = list(ts.samples())
    for position, derived in data.items():
        tree = ts.at(position)
        target = {samples[i] for i in derived}
        compatible_length = sum(tree.branch_length(node) for node in tree.nodes()
                                if set(tree.samples(node)) == target)
        require(compatible_length > 0, f'No compatible mutation branch at {position}')
        result += math.log(compatible_length)
    return result


def topology_stats(signatures, truth, levels=LEVELS):
    counts = Counter(signatures)
    n = len(signatures)
    truth_set = set(truth)
    rf = sum(count * len(set(sig) ^ truth_set) for sig, count in counts.items()) / n
    mass = counts[truth] / n
    # Frequency-threshold credible sets include all ties at the boundary.
    ordered = sorted(counts.values(), reverse=True)
    result = dict(rf=rf, truth_probability=mass, unique_topologies=len(counts),
                  topology_entropy=-sum((c/n)*math.log(c/n) for c in counts.values()),
                  clade_brier=0., true_clade_support=0.)
    clades = set(truth)
    for sig in counts:
        clades.update(sig)
    clade_counts = Counter()
    for sig, count in counts.items():
        for clade in sig:
            clade_counts[clade] += count
    support = {c: clade_counts[c]/n for c in clades}
    if clades:
        result['clade_brier'] = sum((p-int(c in truth_set))**2 for c, p in support.items())/len(clades)
    result['true_clade_support'] = (sum(support[c] for c in truth_set)/len(truth_set)
                                    if truth_set else 1.)
    for level in levels:
        cutoff = ordered[min(int(np.searchsorted(np.cumsum(ordered), level*n)), len(ordered)-1)]
        selected = [s for s, c in counts.items() if c >= cutoff]
        prefix = f'credible_{round(100*level)}'
        result[prefix + '_truth_covered'] = int(counts[truth] >= cutoff)
        result[prefix + '_size'] = len(selected)
        result[prefix + '_mass'] = sum(counts[s] for s in selected)/n
    return result


def summarize_posterior(truth, posterior, pairs, output, chunk_size=64, rank_bins=20):
    """Exact span weighting with bounded draw-by-position working arrays."""
    boundaries = np.unique(np.concatenate([truth.boundaries, *[f.boundaries for f in posterior]]))
    spans = np.diff(boundaries)
    positions = (boundaries[1:] + boundaries[:-1])/2
    length, draws, columns = spans.sum(), len(posterior), len(pairs)+1
    labels = [f'pair_{a}_{b}' for a, b in pairs] + ['root']
    per_column = np.zeros((columns, 7 + 2*len(LEVELS)))
    ranks = [np.zeros(draws+1), np.zeros(draws+1)]
    tie_mass = np.zeros(2)
    topo_sum, cache = Counter(), {}
    count_rows = 0
    with gzip.open(output/'genomic_profile.csv.gz', 'wt', newline='') as handle:
        writer = None
        for start in range(0, len(positions), chunk_size):
            stop = min(start+chunk_size, len(positions))
            pos, weights = positions[start:stop], spans[start:stop]
            ti = truth.at(pos)
            expected = truth.times[ti]
            indices = [f.at(pos) for f in posterior]
            values = np.stack([f.times[idx] for f, idx in zip(posterior, indices)])
            means, medians = values.mean(axis=0), np.median(values, axis=0)
            mean_error, median_error = means-expected, medians-expected
            arrays = [expected, means, medians, mean_error, abs(mean_error), mean_error**2, abs(median_error)]
            coverages = []
            for level in LEVELS:
                tail = (100-100*level)/200
                low, high = np.quantile(values, [tail, 1-tail], axis=0)
                covered = (low <= expected) & (expected <= high)
                arrays.extend([covered, high-low])
                coverages.append((low, high))
            per_column += np.stack([np.sum(a*weights[:, None], axis=0) for a in arrays], axis=1)/length
            for group, selection in enumerate((slice(None, -1), slice(-1, None))):
                result = tmrca_rank_histogram(expected[:, selection], values[:, :, selection], weights, bins=rank_bins)
                fraction = weights.sum()/length
                ranks[group] += np.asarray(result['rank_probabilities'])*fraction
                tie_mass[group] += result['tie_cell_fraction']*fraction
            for offset, weight in enumerate(weights):
                truth_sig = truth.topologies[ti[offset]]
                sample_sigs = tuple(f.topologies[idx[offset]] for f, idx in zip(posterior, indices))
                # Adjacent branch-time intervals often have the same topology ensemble.
                key = (truth_sig, sample_sigs)
                if key not in cache:
                    stats = topology_stats(sample_sigs, truth_sig)
                    if len(cache) >= 128:
                        cache.clear()
                    cache[key] = stats
                stats = cache[key]
                for name, value in stats.items():
                    topo_sum[name] += value*weight/length
                row = dict(left=boundaries[start+offset], right=boundaries[start+offset+1],
                           truth_mean_pair_tmrca=expected[offset, :-1].mean(),
                           posterior_mean_pair_tmrca=means[offset, :-1].mean(),
                           truth_root_time=expected[offset, -1], posterior_mean_root_time=means[offset, -1],
                           **stats)
                for level, (low, high) in zip(LEVELS, coverages):
                    row[f'root_q{(1-level)/2:.3f}'] = low[offset, -1]
                    row[f'root_q{(1+level)/2:.3f}'] = high[offset, -1]
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                count_rows += 1
            if start == 0 or stop == len(positions) or start//chunk_size % 20 == 0:
                print(f'Exact genomic summaries: {stop}/{len(positions)} intervals', flush=True)
    names = ['truth_mean', 'posterior_mean', 'posterior_median_mean', 'mean_bias',
             'mean_mae', 'mean_mse', 'median_mae']
    for level in LEVELS:
        names.extend([f'coverage_{round(100*level)}', f'width_{round(100*level)}'])
    frame = pd.DataFrame(per_column, columns=names, index=labels)
    frame['mean_rmse'] = np.sqrt(frame['mean_mse'])
    frame.index.name = 'quantity'
    frame.to_csv(output/'tmrca_by_pair.csv')
    pair_summary = frame.iloc[:-1].mean().to_dict()
    pair_summary['mean_rmse'] = math.sqrt(pair_summary['mean_mse'])
    rank_results = {}
    for group, label in enumerate(('pair_tmrca', 'root_time')):
        probabilities = ranks[group]/ranks[group].sum()
        edges = np.arange(min(rank_bins, draws+1)+1)*(draws+1)//min(rank_bins, draws+1)
        binned = np.add.reduceat(probabilities, edges[:-1])
        uniform = np.diff(edges)/(draws+1)
        rank_results[label] = dict(rank_probabilities=probabilities, probabilities=binned,
                                  uniform_probabilities=uniform, bin_edges=edges-.5,
                                  kl_from_uniform=rank_kl_divergence(binned, uniform),
                                  tie_cell_fraction=tie_mass[group])
    return dict(pair_tmrca=pair_summary, root_time=frame.iloc[-1].to_dict(),
                ranks=rank_results, topology=dict(topo_sum), aligned_intervals=count_rows)


def diagnose(frame, min_ess=400):
    import arviz as az
    rows = []
    for name in frame.columns:
        x = frame[name].to_numpy(dtype=float)
        row = dict(metric=name, draws=len(x), mean=x.mean(), sd=x.std(ddof=1),
                   ess_bulk=None, ess_tail=None, ess_mean=None, ess_fraction=None,
                   mcse_mean=None, mcse_over_sd=None, lag1_autocorrelation=None,
                   rhat=None)
        if not np.isfinite(x).all():
            row['status'] = 'nonfinite'
        elif len(x) < 8:
            row['status'] = 'too_few_draws'
        elif np.ptp(x) == 0:
            row['status'] = 'constant_uninformative'
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                for method in ('bulk', 'tail', 'mean'):
                    row['ess_'+method] = float(az.ess(x[None, :], method=method))
                row['mcse_mean'] = np.asarray(az.mcse(x[None, :], method='mean')).item()
            row['ess_fraction'] = row['ess_bulk']/len(x)
            row['mcse_over_sd'] = row['mcse_mean']/row['sd']
            row['lag1_autocorrelation'] = float(np.corrcoef(x[:-1], x[1:])[0, 1])
            valid = all(math.isfinite(row[k]) for k in ('ess_bulk', 'ess_tail', 'ess_mean', 'mcse_mean'))
            row['status'] = ('undefined_diagnostic' if not valid else
                             'low_ess' if min(row['ess_bulk'], row['ess_tail']) < min_ess else
                             'high_mcse' if row['mcse_over_sd'] > .05 else 'thresholds_met_single_chain')
        midpoint = len(x)//2
        row['early_half_mean'] = x[:midpoint].mean() if midpoint else None
        row['late_half_mean'] = x[midpoint:].mean()
        row['half_mean_difference_over_sd'] = ((row['late_half_mean']-row['early_half_mean'])/row['sd']
                                               if row['sd'] > 0 and midpoint else None)
        rows.append(row)
    result = pd.DataFrame(rows).set_index('metric')
    for name in ('ess_bulk', 'ess_tail', 'ess_mean', 'ess_fraction', 'mcse_mean',
                 'mcse_over_sd', 'lag1_autocorrelation', 'rhat', 'half_mean_difference_over_sd'):
        result[name] = pd.to_numeric(result[name], errors='raise').astype(float)
    return result


def make_plots(frame, diagnostics, summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.signal import correlate
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), constrained_layout=True)
    for ax, name in zip(axes.flat, GLOBAL_TRACES):
        ax.plot(frame.index, frame[name], linewidth=.6)
        ax.set_title(name)
        ax.set_xlabel('MCMC iteration')
    fig.savefig(output/'traces.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), constrained_layout=True)
    for ax, name in zip(axes.flat, GLOBAL_TRACES):
        x = frame[name].to_numpy()-frame[name].mean()
        if np.dot(x, x) > 0:
            acf = correlate(x, x, mode='full', method='fft')[len(x)-1:]/np.dot(x, x)
            count = min(101, len(x))
            ax.plot(np.arange(count), acf[:count])
        ax.set_title(name)
        ax.set_xlabel('Lag in saved draws')
    fig.savefig(output/'autocorrelation.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, label in zip(axes[0], ('pair_tmrca', 'root_time')):
        rank = summary['ranks'][label]
        ax.bar(np.arange(len(rank['probabilities'])), rank['probabilities'])
        ax.plot(rank['uniform_probabilities'], color='black', linestyle='--')
        ax.set_title(f'{label}: truth rank (KL={rank["kl_from_uniform"]:.3g})')
        ax.set_xlabel('Rank bin')
    for label in ('pair_tmrca', 'root_time'):
        axes[1, 0].plot(LEVELS, [summary[label][f'coverage_{round(100*l)}'] for l in LEVELS], 'o-', label=label)
    axes[1, 0].plot(LEVELS, LEVELS, 'k--')
    axes[1, 0].legend()
    axes[1, 0].set(xlabel='Nominal interval probability', ylabel='Truth coverage')
    axes[1, 1].plot(LEVELS, [summary['topology'][f'credible_{round(100*l)}_truth_covered'] for l in LEVELS], 'o-')
    axes[1, 1].plot(LEVELS, LEVELS, 'k--')
    axes[1, 1].set(xlabel='Topology credible-set probability', ylabel='Truth topology coverage')
    fig.savefig(output/'calibration.png', dpi=160)
    plt.close(fig)
    profile = pd.read_csv(output/'genomic_profile.csv.gz')
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    edges = np.r_[profile.left.to_numpy(), profile.right.iloc[-1]]
    for ax, names in zip(axes[:2], [('truth_mean_pair_tmrca', 'posterior_mean_pair_tmrca'),
                                    ('truth_root_time', 'posterior_mean_root_time')]):
        for name in names:
            ax.stairs(profile[name], edges, baseline=None, label=name)
        ax.legend()
        ax.set_ylabel('Generations')
    axes[2].stairs(profile.truth_probability, edges, baseline=None,
                   label='True topology posterior probability')
    axes[2].legend()
    axes[2].set_xlabel('Genomic coordinate (bp)')
    axes[2].set_xlim(edges[0], edges[-1])
    axes[2].set_ylim(0, 1.05)
    fig.savefig(output/'genomic_profiles.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    chosen = diagnostics.loc[list(GLOBAL_TRACES)]
    ax.bar(np.arange(len(chosen))-.18, chosen.ess_bulk, .36, label='Bulk ESS')
    ax.bar(np.arange(len(chosen))+.18, chosen.ess_tail, .36, label='Tail ESS')
    ax.set_xticks(np.arange(len(chosen)), chosen.index, rotation=25, ha='right')
    ax.legend()
    fig.savefig(output/'ess.png', dpi=160)
    plt.close(fig)


def run(args):
    import arginfer
    import arviz as az
    import bintrees
    require(__debug__, 'Do not run with python -O: ARGinfer uses assertions for validation')
    require(args.burnin_samples >= 0 and args.expected_thin > 0 and args.chunk_size > 0 and
            args.rank_bins > 0 and args.diagnostic_positions > 0 and args.min_ess > 0,
            'Invalid nonpositive setting or negative burn-in')
    dataset = args.dataset_dir.resolve()
    input_dir = (args.input_dir or dataset.parent/'arginfer_inputs').resolve()
    metadata, manifest, truth_ts, data = validate_inputs(dataset, input_dir)
    files, available = inventory(args.arg_dir, args.burnin_samples, args.expected_thin)
    require(len(files) >= 8, 'At least eight retained samples are required')
    output = args.output_dir.resolve()
    require(not output.exists(), f'Output already exists; choose a new --output-dir: {output}')
    output.mkdir(parents=True)
    n, length = manifest['num_haplotypes'], manifest['sequence_length']
    pairs = list(itertools.combinations(range(n), 2))
    truth = extract_features(truth_ts, metadata['sample_nodes_in_haplotype_order'])
    positions = (np.arange(args.diagnostic_positions)+.5)*length/args.diagnostic_positions
    native_data = {p: bintrees.AVLTree({i: i for i in samples}) for p, samples in data.items()}
    posterior, records, source_files, breakpoint_rows, local_signatures = [], [], [], [], []
    write_json(output/'settings.json', dict(arguments={k: str(v) if isinstance(v, Path) else v
                                                      for k, v in vars(args).items()},
                                           diagnostic_positions=positions, input_manifest=manifest))
    try:
        for i, (iteration, path) in enumerate(files):
            # Only local, trusted ARGinfer outputs should be unpickled.
            with path.open('rb') as handle:
                arg = pickle.load(handle)
            arg.verify()
            ts, raw, mapping = arg_to_ts(arg, n, length)
            check_mutations(arg, raw, mapping, data, n)
            features = extract_features(ts)
            native_length = arg.total_branch_length()
            span_mean = lambda values: float(np.dot(np.diff(features.boundaries), values)/length)
            tree_length = span_mean(features.branch)*length
            require(math.isclose(native_length, tree_length, rel_tol=1e-9, abs_tol=1e-6),
                    'ARG material length differs from marginal-tree branch length')
            likelihood = arg.log_likelihood(manifest['mutation_rate'], native_data)
            likelihood_error = abs(likelihood-tree_log_likelihood(ts, data, manifest['mutation_rate']))
            require(likelihood_error <= 1e-8*max(1., abs(likelihood)),
                    'Native likelihood differs from independent marginal-tree likelihood')
            prior = arg.log_prior(n, length, manifest['recombination_rate'], manifest['population_size'])
            recombinations = sum(node.breakpoint is not None for node in arg.nodes.values())
            require(recombinations == arg.num_ancestral_recomb + arg.num_nonancestral_recomb,
                    'Recombination count disagrees with native prior traversal')
            breakpoints = [node.breakpoint for node in arg.nodes.values() if node.breakpoint is not None]
            row = dict(iteration=iteration, log_likelihood=likelihood, log_prior=prior,
                       log_posterior=likelihood+prior, recombinations=recombinations,
                       ancestral_recombinations=arg.num_ancestral_recomb,
                       nonancestral_recombinations=arg.num_nonancestral_recomb,
                       marginal_trees=ts.num_trees, unique_recombination_breakpoints=len(set(breakpoints)),
                       arg_nodes=len(arg.nodes), mean_pair_tmrca=span_mean(features.times[:, :-1].mean(axis=1)),
                       mean_root_time=span_mean(features.times[:, -1]),
                       mean_total_branch_length=tree_length/length,
                       mutation_count=len(data), material_length_absolute_error=abs(native_length-tree_length),
                       log_likelihood_absolute_error=likelihood_error)
            idx = features.at(positions)
            for j, values in enumerate(features.times[idx]):
                for k, (a, b) in enumerate(pairs):
                    row[f'pos{j}_pair_{a}_{b}'] = values[k]
                row[f'pos{j}_root_time'] = values[-1]
            local_signatures.append([features.topologies[k] for k in idx])
            require(all(math.isfinite(x) for x in row.values()), f'Nonfinite summary in {path}')
            records.append(row)
            posterior.append(features)
            breakpoint_rows.extend(dict(iteration=iteration, position=p) for p in breakpoints)
            source_files.append(dict(path=str(path.resolve()), iteration=iteration, sha256=sha256(path)))
            if i == 0 or (i+1) % 100 == 0 or i+1 == len(files):
                print(f'Validated and extracted {i+1}/{len(files)} ARGs', flush=True)
        frame = pd.DataFrame(records).set_index('iteration')
        # Monitor every observed local clade, plus true topology membership.
        local_columns = {}
        for j, position in enumerate(positions):
            sigs = [s[j] for s in local_signatures]
            all_clades = sorted(set().union(*map(set, sigs)))
            for clade in all_clades:
                local_columns[f'pos{j}_clade_{clade}'] = [int(clade in s) for s in sigs]
            true_sig = truth.topologies[truth.at([position])[0]]
            local_columns[f'pos{j}_truth_topology'] = [int(s == true_sig) for s in sigs]
        frame = frame.join(pd.DataFrame(local_columns, index=frame.index, dtype=np.int8))
        frame.to_csv(output/'traces.csv.gz')
        pd.DataFrame(breakpoint_rows, columns=['iteration', 'position']).to_csv(output/'recombination_breakpoints.csv.gz', index=False)
        summary = summarize_posterior(truth, posterior, pairs, output, args.chunk_size, args.rank_bins)
        # Count/branch-length posterior intervals and truth reference (full ARG count is separate).
        truth_globals = dict(marginal_trees=len(truth.branch),
                             mean_pair_tmrca=np.average(truth.times[:, :-1].mean(axis=1), weights=np.diff(truth.boundaries)),
                             mean_root_time=np.average(truth.times[:, -1], weights=np.diff(truth.boundaries)),
                             mean_total_branch_length=np.average(truth.branch, weights=np.diff(truth.boundaries)))
        structural = {}
        for name in GLOBAL_TRACES + ('ancestral_recombinations', 'nonancestral_recombinations', 'arg_nodes'):
            x = frame[name].to_numpy()
            structural[name] = dict(mean=x.mean(), median=np.median(x), sd=x.std(ddof=1),
                                    q025=np.quantile(x, .025), q975=np.quantile(x, .975),
                                    truth=truth_globals.get(name))
        full_path = dataset / f'{metadata["dataset_name"]}.full.trees'
        if full_path.is_file():
            full = tskit.load(full_path)
            # msprime marks both parental nodes of a recombination with bit 17.
            import msprime
            rec_nodes = int(np.count_nonzero(full.tables.nodes.flags & msprime.NODE_IS_RE_EVENT))
            require(rec_nodes % 2 == 0, 'Odd number of full-truth recombination parent nodes')
            structural['recombinations']['truth'] = rec_nodes//2
        print('Computing single-chain ESS and Monte Carlo errors', flush=True)
        residuals = ['material_length_absolute_error', 'log_likelihood_absolute_error']
        diagnostics = diagnose(frame.drop(columns=residuals), args.min_ess)
        diagnostics.to_csv(output/'diagnostics.csv')
        late = diagnose(frame.iloc[len(frame)//2:].drop(columns=residuals), args.min_ess)
        late.to_csv(output/'late_half_diagnostics.csv')
        evolution = []
        for count in sorted({max(8, len(frame)//4), max(8, len(frame)//2), len(frame)}):
            result = diagnose(frame.iloc[:count][list(GLOBAL_TRACES)], args.min_ess).reset_index()
            result['prefix_draws'] = count
            evolution.append(result)
        pd.concat(evolution).to_csv(output/'ess_evolution.csv', index=False)
        summary.update(dataset=metadata['dataset_name'], draws=len(files), available_draws=available,
                       first_iteration=files[0][0], last_iteration=files[-1][0],
                       saved_iteration_spacing=args.expected_thin, structural=structural,
                       diagnostic_status_counts=diagnostics.status.value_counts().to_dict(),
                       global_diagnostics=diagnostics.loc[list(GLOBAL_TRACES)].reset_index().to_dict('records'),
                       time_units='generations', time_divisor_for_2Ne=2*manifest['population_size'],
                       weighting='Exact genomic spans; equal haplotype pairs; equal posterior draws',
                       topology_definition='Rooted nontrivial clades; unary nodes suppressed; branch times ignored',
                       topology_credible_sets='Highest empirical frequency, including all boundary ties',
                       integrity=dict(validated_args=len(files), mutation_compatible_args=len(files),
                                      max_material_length_error=frame.material_length_absolute_error.max(),
                                      max_log_likelihood_error=frame.log_likelihood_absolute_error.max()),
                       limitations=[
                           'One chain: R-hat unavailable; ESS/trace stability does not establish full posterior convergence.',
                           'Rank/coverage are descriptive for this dataset; linked positions and pairs are dependent.',
                           'Constant traces have undefined ESS and are not evidence of convergence.',
                           'ARGinfer used integer VCF positions; truth retains original genomic coordinates.',
                           'Native ARGinfer log likelihood/prior are recomputed; their additive constants are not evidence estimates.',
                           'Recombination event counts differ from topology-changing breakpoint counts.',
                           'Acceptance rates cannot be recovered from saved ARGs alone.'])
        write_json(output/'summary.json', summary)
        provenance_paths = [Path(__file__), dataset/'metadata.json',
                            dataset/metadata['files']['ground_truth_trees'],
                            dataset/metadata['files']['vcf']]
        provenance_paths += list(input_dir.glob('*.txt')) + [input_dir/'samples.tsv', input_dir/'manifest.json']
        provenance_paths += [ROOT/'eval'/name for name in ('tmrca_ranks.py', 'rank_kl.py', '_calibration.py')]
        import arginfer.argbook
        provenance_paths.append(Path(arginfer.argbook.__file__))
        if full_path.is_file():
            provenance_paths.append(full_path)
        write_json(output/'provenance.json', dict(args=source_files,
                   files={str(p): sha256(p) for p in provenance_paths},
                   versions=dict(python=sys.version, numpy=np.__version__, tskit=tskit.__version__,
                                 arviz=az.__version__, arginfer=arginfer.__version__)))
        make_plots(frame, diagnostics, summary, output)
        text = [f'# ARGinfer validation: {metadata["dataset_name"]}', '',
                f'{len(files)} posterior draws, iterations {files[0][0]}–{files[-1][0]}, spacing {args.expected_thin}.', '',
                'Times are in generations. Accuracy and coverage use exact genomic spans.', '',
                f'Pairwise TMRCA posterior-mean RMSE: {summary["pair_tmrca"]["mean_rmse"]:.6g}.',
                f'90% TMRCA interval coverage: {summary["pair_tmrca"]["coverage_90"]:.3%}.',
                f'90% topology credible-set coverage: {summary["topology"]["credible_90_truth_covered"]:.3%}.', '',
                '| Quantity | Bulk ESS | Tail ESS | Status |', '|---|---:|---:|---|']
        for name in GLOBAL_TRACES:
            row = diagnostics.loc[name]
            text.append(f'| {name} | {row.ess_bulk:.1f} | {row.ess_tail:.1f} | {row.status} |')
        text += ['', *[f'- {s}' for s in summary['limitations']], '',
                 'See summary.json, diagnostics.csv, late_half_diagnostics.csv, ess_evolution.csv,',
                 'tmrca_by_pair.csv, genomic_profile.csv.gz, traces.csv.gz and the PNG plots.']
        (output/'report.md').write_text('\n'.join(text)+'\n')
        (output/'SUCCESS').write_text('All saved ARGs validated; report generation completed.\n')
        print(f'Report: {output / "report.md"}', flush=True)
    except Exception as exc:
        write_json(output/'failure.json', dict(error_type=type(exc).__name__, error=str(exc),
                   current_arg=str(path), validated_args=len(records),
                   last_valid_iteration=records[-1]['iteration'] if records else None))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arg-dir', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--input-dir', type=Path, help='Default: <dataset-directory>/../arginfer_inputs')
    parser.add_argument('--output-dir', type=Path, required=True, help='Must not exist')
    parser.add_argument('--burnin-samples', type=int, default=0, help='Additional saved draws to discard; original burn-in already removed')
    parser.add_argument('--expected-thin', type=int, default=1000)
    parser.add_argument('--chunk-size', type=int, default=64)
    parser.add_argument('--rank-bins', type=int, default=20)
    parser.add_argument('--diagnostic-positions', type=int, default=5, help='Evenly spaced midpoint locations for local ESS')
    parser.add_argument('--min-ess', type=float, default=400)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
