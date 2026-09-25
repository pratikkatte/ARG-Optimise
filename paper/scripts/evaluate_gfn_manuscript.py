"""Evaluate 256 unweighted GFN draws with the manuscript's r1 truth metrics."""
import argparse
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import tskit
from env.snp_data import load_snp_dataset
from eval.ess import importance_stats
from validation.scripts.evaluate_arginfer import (
    extract_features, require, sha256, summarize_posterior, write_json)
from paper.scripts.compare_r1_arg_summaries_256 import clade_scores, comparison_plots


def run(manifest_path, dataset, output):
    require(not output.exists(), f'Output exists: {output}')
    manifest = json.loads(manifest_path.read_text())
    metadata = json.loads((dataset/'metadata.json').read_text())
    observations = load_snp_dataset(dataset)
    require(manifest['sampling_distribution'] == 'learned_policy', 'Expected policy draws')
    require(manifest['haplotype_ids'] == list(observations.haplotype_ids), 'Sample identity mismatch')
    require(manifest['sequence_length'] == observations.sequence_length, 'Sequence length mismatch')
    records = manifest['samples']
    require(len(records) == 256 and [r['index'] for r in records] == list(range(256)),
            'Expected exactly 256 ordered draws')
    truth_path = dataset/metadata['files']['ground_truth_trees']
    truth_ts = tskit.load(truth_path)
    sample_order = metadata['sample_nodes_in_haplotype_order']
    variants = list(truth_ts.variants(samples=sample_order))
    require(len(variants) == observations.num_variants, 'Truth SNP count mismatch')
    for j, variant in enumerate(variants):
        require(variant.site.position == observations.positions[j], 'Truth SNP coordinate mismatch')
        np.testing.assert_array_equal(variant.genotypes != 0, observations.genotypes[:, j])
    truth = extract_features(truth_ts, sample_order)
    features, hashes = [], {}
    n = observations.num_haplotypes
    for row in records:
        require(row['source'] == 'policy' and row['status'] == 'complete' and row['temperature'] == 1.,
                'Expected complete T=1 policy draw')
        path = manifest_path.parent/row['trees_file']
        ts = tskit.load(path)
        require(ts.num_samples == n and ts.sequence_length == observations.sequence_length,
                'Draw dimensions mismatch')
        for j, position in enumerate(observations.positions):
            tree = ts.at(position)
            samples = list(ts.samples())
            derived = {samples[k] for k in np.flatnonzero(observations.genotypes[:, j])}
            require(any(set(tree.samples(node)) == derived for node in tree.nodes()), 'Incompatible SNP')
        features.append(extract_features(ts))
        hashes[str(path)] = sha256(path)
    importance = importance_stats([r['log_importance_weight'] for r in records])
    require(np.isclose(importance['ess'], manifest['summary']['importance_ess']), 'ESS mismatch')
    output.mkdir(parents=True)
    summary = summarize_posterior(truth, features, list(itertools.combinations(range(n), 2)), output)
    summary['topology']['rf_normalized'] = summary['topology']['rf']/(2*(n-2))
    summary['topology'].update(clade_scores(truth, features, n, output))
    summary['importance'] = importance
    summary['protocol'] = dict(draws=256, seed=manifest['seed'], weights='equal; unweighted',
        time_units='generations', genomic_weighting='exact spans; equal haplotype pairs',
        source_manifest=str(manifest_path.resolve()), arginfer_comparison='Unavailable: reference files missing',
        limitation='Single dataset; linked sites and pairs are not independent calibration replicates.')
    write_json(output/'summary.json', summary)
    write_json(output/'provenance.json', dict(files=hashes, truth_sha256=sha256(truth_path),
        manifest_sha256=sha256(manifest_path), code={str(p): sha256(p) for p in
        [Path(__file__), ROOT/'validation/scripts/evaluate_arginfer.py',
         ROOT/'paper/scripts/compare_r1_arg_summaries_256.py']}))
    comparison_plots({'GFN': summary}, output)
    lines = ['# GFN manuscript metrics against r1 truth', '',
             '256 equally weighted T=1 policy draws. ARGinfer comparison unavailable.', '',
             '| Quantity | Metric | Value |', '|---|---|---:|']
    for quantity in ('pair_tmrca', 'root_time', 'topology', 'importance'):
        for name, value in summary[quantity].items():
            if isinstance(value, (int, float)):
                lines.append(f'| {quantity} | {name} | {value:.8g} |')
    for quantity in ('pair_tmrca', 'root_time'):
        lines.append(f'| {quantity} | rank KL | {summary["ranks"][quantity]["kl_from_uniform"]:.8g} |')
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    (output/'SUCCESS').write_text('Completed GFN-only truth metrics; no ARGinfer comparison.\n')
    print((output/'report.md').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.dataset_dir, args.output_dir)
