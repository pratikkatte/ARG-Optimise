"""Reproduce Section 5.1 metrics from saved ARGFlow draws and simulated truth."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import pickle
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import tskit
import yaml

from paper.scripts.paper_datasets.cli import load_config

from env.snp_data import load_snp_dataset
from validation.scripts.evaluate_arginfer import (
    extract_features, inventory, validate_inputs, arg_to_ts, check_mutations)

METRICS = ('pair_tmrca_rmse', 'rooted_rf', 'clade_brier',
           'tmrca_coverage', 'tmrca_interval_width')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def time_metrics(truth, draws, ne, metrics, level=.9, chunk_size=64, progress=False):
    """Equal pair weighting and exact span integrals, with bounded working memory."""
    require(bool(draws), 'No draws supplied')
    require(ne > 0 and math.isfinite(ne), 'Ne must be finite and positive')
    require(0 < level < 1, 'Credible level must be between zero and one')
    require(chunk_size >= 1, 'chunk_size must be positive')
    boundaries = np.unique(np.concatenate([truth.boundaries, *[d.boundaries for d in draws]]))
    spans = np.diff(boundaries)
    pairs = truth.times.shape[1] - 1
    accum = np.zeros((pairs, 3))  # MSE, coverage, width, already in 2Ne units.
    selected = set(metrics)
    intervals = bool(selected & {'tmrca_coverage', 'tmrca_interval_width'})
    for start in range(0, len(spans), chunk_size):
        stop = min(start + chunk_size, len(spans))
        # Half-open intervals: evaluating at the left endpoint avoids midpoint rounding.
        pos = boundaries[start:stop]
        weights = spans[start:stop, None] / spans.sum()
        actual = truth.times[truth.at(pos), :-1] / (2 * ne)
        values = np.stack([d.times[d.at(pos), :-1] for d in draws]) / (2 * ne)
        if 'pair_tmrca_rmse' in selected:
            accum[:, 0] += np.sum(weights * (values.mean(axis=0) - actual)**2, axis=0)
        if intervals:
            tail = (1 - level) / 2
            low, high = np.quantile(values, [tail, 1-tail], axis=0, method='linear')
            accum[:, 1] += np.sum(weights * ((low <= actual) & (actual <= high)), axis=0)
            accum[:, 2] += np.sum(weights * (high-low), axis=0)
        if progress and (start == 0 or stop == len(spans) or start // chunk_size % 100 == 0):
            print(f'  TMRCA: {stop}/{len(spans)} exact intervals', flush=True)
    summary = {}
    if 'pair_tmrca_rmse' in selected:
        summary['pair_tmrca_rmse'] = float(np.sqrt(accum[:, 0].mean()))
    if 'tmrca_coverage' in selected:
        summary['tmrca_coverage'] = float(accum[:, 1].mean())
    if 'tmrca_interval_width' in selected:
        summary['tmrca_interval_width'] = float(accum[:, 2].mean())
    return summary, accum, len(spans)


def topology_metrics(truth, draws, n):
    """Sweep topology changes; unary nodes and the full/root clade are excluded."""
    require(bool(draws) and n >= 2, 'Topology metrics require draws and at least two samples')
    events = defaultdict(Counter)
    for draw in draws:
        previous = set()
        for left, signature in zip(draw.boundaries[:-1], draw.topologies):
            current = set(signature)
            if current != previous:
                events[float(left)].update(current - previous)
                events[float(left)].subtract(previous - current)
            previous = current
    boundaries = np.unique(np.concatenate([truth.boundaries, list(events)]))
    counts = Counter()
    rf_total = brier_sum = union_score = 0.
    universe_size = 2**n - n - 2
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        counts.update(events.get(float(left), {}))
        # Zero-count clades must not enlarge the observed-union denominator.
        counts = +counts
        actual = set(truth.topologies[int(truth.at(left))])
        probabilities = {c: count / len(draws) for c, count in counts.items()}
        union = counts.keys() | actual
        error = sum((probabilities.get(c, 0.) - int(c in actual))**2 for c in union)
        rf = sum(probabilities.values()) + len(actual) - 2 * sum(probabilities.get(c, 0.) for c in actual)
        weight = (right-left) / truth.boundaries[-1]
        rf_total += weight * rf
        brier_sum += weight * error
        union_score += weight * (error / len(union) if union else 0.)
    return dict(rooted_rf=float(max(0., rf_total)),
                clade_brier_observed_union=float(union_score),
                clade_brier_fixed_universe=float(brier_sum / universe_size if universe_size else 0.),
                clade_brier_sum=float(brier_sum), clade_universe_size=universe_size)


def load_draws(name, dataset_dir, observations, source, config):
    directory = resolve(source['directory'])
    kind = source['format']
    provenance = dict(source=source)
    expected = source.get('expected_samples', config['expected_samples'])
    burnin = source.get('burnin_samples', 0)
    require(isinstance(burnin, int) and burnin >= 0, 'Invalid burnin_samples')
    rows, observed = None, None
    if kind == 'argflow':
        require(burnin == 0, 'Independent ARGFlow draws do not use burn-in')
        manifest_path = directory / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        require(manifest['status'] == 'complete', 'Sampling manifest is incomplete')
        require(manifest['dataset'] == name, 'Manifest dataset mismatch')
        require(manifest['haplotype_ids'] == list(observations.haplotype_ids), 'Haplotype order mismatch')
        require(manifest['sequence_length'] == observations.sequence_length, 'Manifest length mismatch')
        rows = manifest['samples']
        filenames = [r['trees_file'] for r in rows]
        require(len(set(filenames)) == len(rows), 'Duplicate manifest filenames')
        require(set(filenames) == {p.name for p in directory.glob('*.trees')}, 'Manifest/file inventory mismatch')
        require(sha256(resolve(manifest['checkpoint'])) == manifest['checkpoint_sha256'], 'Checkpoint hash mismatch')
        provenance.update(sampling_manifest=dict(path=str(manifest_path), sha256=sha256(manifest_path)),
                          checkpoint_sha256=manifest['checkpoint_sha256'])
        files = [(i, directory / row['trees_file']) for i, row in enumerate(rows)]
    else:
        # Both baselines received the VCF's integer coordinates, not the original
        # continuous simulation positions. Verify their data in those coordinates.
        input_dir = resolve(source.get('input_dir', str(dataset_dir.parent / 'arginfer_inputs')))
        _, _, _, observed = validate_inputs(dataset_dir, input_dir)
        provenance['baseline_input_manifest'] = dict(path=str(input_dir/'manifest.json'),
                                                     sha256=sha256(input_dir/'manifest.json'))
        if kind == 'arginfer':
            require(__debug__, 'ARGInfer validation requires Python without -O')
            files, available = inventory(directory, burnin, source.get('expected_thin', 1000))
            provenance['available_samples'] = available
        elif kind == 'singer':
            files = []
            for path in directory.glob('*.trees'):
                match = re.fullmatch(r'trees_(\d+)\.trees', path.name)
                require(match is not None, f'Unexpected SINGER filename: {path}')
                files.append((int(match[1]), path))
            files.sort()
            require(all(b[0]-a[0] == 1 for a, b in zip(files, files[1:])), 'SINGER index gap')
            provenance['available_samples'] = len(files)
            files = files[burnin:]
        else:
            raise ValueError(f'Unknown format: {kind}')
    require(len(files) == expected, f'Expected {expected} draws, found {len(files)} in {directory}')
    provenance['first_index'], provenance['last_index'] = files[0][0], files[-1][0]
    draws, hashes = [], []
    for i, (index, path) in enumerate(files):
        digest = sha256(path)
        if kind == 'arginfer':
            # These are the user's trusted local inference outputs.
            with path.open('rb') as handle:
                arg = pickle.load(handle)
            arg.verify()
            ts, raw, mapping = arg_to_ts(arg, observations.num_haplotypes, observations.sequence_length)
            check_mutations(arg, raw, mapping, observed, observations.num_haplotypes)
        else:
            ts = tskit.load(path)
            if kind == 'argflow':
                row = rows[i]
                require(row['source'] == 'policy' and row['status'] == 'complete' and row['temperature'] == 1,
                        'Expected completed, untempered policy draws')
                if config['verify_sample_hashes']:
                    require(digest == row['trees_sha256'], f'Sample hash mismatch: {path}')
                for j, pos in enumerate(observations.positions):
                    tree = ts.at(pos)
                    derived = set(np.flatnonzero(observations.genotypes[:, j]))
                    require(any(set(tree.samples(node)) == derived for node in tree.nodes()),
                            f'Incompatible observed SNP in {path} at {pos}')
            else:
                np.testing.assert_array_equal(ts.tables.sites.position, sorted(observed))
                for variant in ts.variants():
                    actual = {j for j, g in enumerate(variant.genotypes)
                              if variant.alleles[g] != variant.site.ancestral_state}
                    require(actual == observed[variant.site.position], f'SINGER genotype mismatch: {path}')
                if ts.time_units == 'unknown':
                    require(source.get('unknown_time_units') == 'generations',
                            'SINGER has unknown time units; specify its documented units in config')
                    tables = ts.dump_tables()
                    tables.time_units = 'generations'
                    ts = tables.tree_sequence()
        require(ts.num_samples == observations.num_haplotypes, 'Sample count mismatch')
        require(ts.sequence_length == observations.sequence_length, 'Sample length mismatch')
        np.testing.assert_array_equal(ts.samples(), np.arange(observations.num_haplotypes))
        draws.append(extract_features(ts))
        hashes.append(dict(path=str(path), index=index, sha256=digest))
        if (i+1) % 250 == 0 or i+1 == len(files):
            print(f'  Loaded and checked {i+1}/{len(files)} draws', flush=True)
    provenance['files'] = hashes
    return draws, provenance


def evaluate_dataset(name, settings, source, config, output):
    started = time.monotonic()
    dataset_dir = resolve(settings['dataset_dir'])
    metadata_path = dataset_dir / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    observations = load_snp_dataset(dataset_dir)
    require(metadata['dataset_name'] == name, 'Dataset name mismatch')
    ne = float(metadata['parameters']['population_size'])
    truth_path = dataset_dir / metadata['files']['ground_truth_trees']
    truth_ts = tskit.load(truth_path)
    truth_order = metadata['sample_nodes_in_haplotype_order']
    require(truth_ts.sequence_length == observations.sequence_length, 'Truth length mismatch')
    np.testing.assert_array_equal(truth_ts.tables.sites.position, observations.positions)
    for j, variant in enumerate(truth_ts.variants(samples=truth_order)):
        np.testing.assert_array_equal(
            [variant.alleles[g] != variant.site.ancestral_state for g in variant.genotypes],
            observations.genotypes[:, j])
    truth = extract_features(truth_ts, truth_order)
    draws, provenance = load_draws(name, dataset_dir, observations, source, config)
    selected = set(config['metrics'])
    result = dict(dataset=name, method=source['method'],
                  checkpoint=Path(source['directory']).name, samples=len(draws))
    details = {}
    if selected & {'pair_tmrca_rmse', 'tmrca_coverage', 'tmrca_interval_width'}:
        summary, per_pair, aligned = time_metrics(truth, draws, ne, selected,
            config['credible_level'], config['chunk_size'], progress=True)
        result.update(summary)
        details['aligned_intervals'] = aligned
        with (output / 'tmrca_by_pair.csv').open('w', newline='') as handle:
            fields = ['haplotype_a', 'haplotype_b'] + [m for m in METRICS if m in summary]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for pair, values in zip(itertools.combinations(observations.haplotype_ids, 2), per_pair):
                entry = dict(haplotype_a=pair[0], haplotype_b=pair[1])
                candidates = dict(pair_tmrca_rmse=float(np.sqrt(values[0])),
                                  tmrca_coverage=float(values[1]), tmrca_interval_width=float(values[2]))
                entry.update({k: v for k, v in candidates.items() if k in summary})
                writer.writerow(entry)
    if selected & {'rooted_rf', 'clade_brier'}:
        topology = topology_metrics(truth, draws, observations.num_haplotypes)
        if 'rooted_rf' in selected:
            result['rooted_rf'] = topology['rooted_rf']
        if 'clade_brier' in selected:
            result['clade_brier'] = topology['clade_brier_' + config['brier_normalization']]
            details.update({k: v for k, v in topology.items() if k != 'rooted_rf'})
    details.update(result=result, population_size=ne, time_divisor=2*ne, time_units='2Ne generations',
                   credible_level=config['credible_level'], quantile_method='linear',
                   interval_endpoints='inclusive', brier_normalization=config['brier_normalization'],
                   sample_weighting='equal; no importance reweighting',
                   genomic_weighting='exact spans; equal weight across unordered haplotype pairs',
                   truth=dict(path=str(truth_path), sha256=sha256(truth_path)),
                   metadata=dict(path=str(metadata_path), sha256=sha256(metadata_path)),
                   provenance=provenance,
                   elapsed_seconds=time.monotonic()-started)
    write_json(output / 'results.json', details)
    return result


def write_table(output, rows, config):
    fields = ['dataset', 'method', 'checkpoint', 'samples'] + config['metrics']
    with (output / 'section_5_1.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    level = f'{100*config["credible_level"]:g}%'
    labels = dict(dataset='Dataset', method='Method', checkpoint='Checkpoint', samples='Draws',
                  pair_tmrca_rmse='TMRCA RMSE (2Ne)', rooted_rf='Rooted RF', clade_brier='Clade Brier',
                  tmrca_coverage=f'{level} coverage', tmrca_interval_width=f'{level} width (2Ne)')
    lines = ['| ' + ' | '.join(labels[f] for f in fields) + ' |',
             '| ' + ' | '.join('---' for _ in fields) + ' |']
    for row in rows:
        cells = []
        for field in fields:
            value = row[field]
            cells.append(f'{100*value:.2f}%' if field == 'tmrca_coverage' else
                         f'{value:.6f}' if isinstance(value, float) else str(value))
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.extend(['', 'All draws have equal weight. Genomic spans are integrated exactly.',
                  f'Clade Brier normalization: `{config["brier_normalization"]}`; both definitions are in per-checkpoint results.json.',
                  'Coverage is descriptive for one simulated dataset per regime; positions and pairs are dependent.',
                  'Rows describe separate checkpoints; no checkpoint selection based on truth was performed.', ''])
    (output / 'section_5_1.md').write_text('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    args, config = load_config(parser, argv)
    require(bool(config['metrics']) and set(config['metrics']) <= set(METRICS), 'Unknown/empty metrics')
    require(len(set(config['metrics'])) == len(config['metrics']), 'Duplicate metrics')
    require(config['brier_normalization'] in ('observed_union', 'fixed_universe'), 'Unknown Brier normalization')
    require(0 < config['credible_level'] < 1, 'Invalid credible_level')
    require(isinstance(config['chunk_size'], int) and config['chunk_size'] > 0, 'Invalid chunk_size')
    require(isinstance(config['expected_samples'], int) and config['expected_samples'] > 0, 'Invalid expected_samples')
    jobs = [(name, settings, source) for name, settings in config['datasets'].items()
            if settings.get('enabled', True) for source in settings['sources'] if source.get('enabled', True)]
    require(bool(jobs), 'No enabled datasets/sample directories')
    require(len({(name, source['method'], Path(source['directory']).name) for name, _, source in jobs}) == len(jobs),
            'Duplicate output row names')
    output = resolve(config['output_dir'])
    for _, settings, source in jobs:
        for protected in (resolve(source['directory']), resolve(settings['dataset_dir'])):
            require(not (output == protected or protected in output.parents), 'Output must be separate from inputs')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.used.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(output / 'provenance.json', dict(
        python=platform.python_version(), numpy=np.__version__, tskit=tskit.__version__, pyyaml=yaml.__version__,
        config=dict(path=str(args.config.resolve()), sha256=sha256(args.config)),
        sources={str(p.relative_to(ROOT)): sha256(p) for p in
                 (Path(__file__), ROOT/'validation/scripts/evaluate_arginfer.py', ROOT/'env/snp_data.py')}))
    rows = []
    for name, settings, source in jobs:
        print(f'Evaluating {name}/{source["method"]}/{Path(source["directory"]).name}', flush=True)
        directory = output / name / source['method'] / Path(source['directory']).name
        directory.mkdir(parents=True, exist_ok=True)
        rows.append(evaluate_dataset(name, settings, source, config, directory))
        print(json.dumps(rows[-1], indent=2), flush=True)
    write_table(output, rows, config)
    print(f'Complete: {output / "section_5_1.md"}', flush=True)


if __name__ == '__main__':
    main()
