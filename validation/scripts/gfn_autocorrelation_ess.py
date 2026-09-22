#!/usr/bin/env python3
"""Apply ARGinfer-style autocorrelation ESS to ordered, fresh GFN inference draws.

This diagnoses serial dependence under the learned policy, not posterior fit.
Keep importance ESS alongside it. Input is the manifest produced by infer.py.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from validation.scripts.evaluate_arginfer import (
    GLOBAL_TRACES, diagnose, extract_features, require, sha256, write_json)
from eval.ess import importance_stats, normalized_weights
import numpy as np
import pandas as pd
import tskit


def run(manifest_path, output, checkpoint=None, reference_report=None):
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get('sampling_distribution') == 'learned_policy',
            'Requires fresh learned-policy draws from infer.py')
    records = manifest['samples']
    require(len(records) >= 8, 'At least eight saved draws required')
    require([r['index'] for r in records] == list(range(len(records))),
            'Manifest must preserve the original complete draw order')
    require(not output.exists(), 'Output directory already exists')
    length = manifest['sequence_length']
    positions = (np.arange(5)+.5)*length/5
    pairs = list(itertools.combinations(range(len(manifest['haplotype_ids'])), 2))
    rows, hashes = [], {}
    for record in records:
        require(record['status'] == 'complete' and record['source'] == 'policy'
                and record['temperature'] == 1., 'Requires complete untempered policy draws')
        path = manifest_path.parent/record['trees_file']
        ts = tskit.load(path)
        require(ts.sequence_length == length and ts.num_samples == len(manifest['haplotype_ids']),
                'Manifest/tree dimensions differ')
        features = extract_features(ts)
        spans = np.diff(features.boundaries)
        average = lambda x: float(np.dot(spans, x)/length)
        row = dict(draw=record['index'], log_likelihood=record['log_likelihood'],
                   log_prior=record['log_prior'],
                   log_posterior=record['log_likelihood']+record['log_prior'],
                   recombinations=record['recombinations'], marginal_trees=len(features.branch),
                   mean_pair_tmrca=average(features.times[:, :-1].mean(axis=1)),
                   mean_root_time=average(features.times[:, -1]),
                   mean_total_branch_length=average(features.branch),
                   log_importance_weight=record['log_importance_weight'])
        require(np.isclose(record['log_importance_weight'],
                           record['log_reward']-record['log_policy_density'], rtol=0, atol=1e-8),
                'Stored importance weight does not match reward minus policy density')
        for j, values in enumerate(features.times[features.at(positions)]):
            for k, (a, b) in enumerate(pairs):
                row[f'pos{j}_pair_{a}_{b}'] = values[k]
            row[f'pos{j}_root_time'] = values[-1]
        rows.append(row)
        hashes[record['trees_file']] = sha256(path)
    frame = pd.DataFrame(rows).set_index('draw')
    require(np.isfinite(frame.to_numpy()).all(), 'Nonfinite input values')
    importance = importance_stats(frame.log_importance_weight.to_numpy())
    require(np.isclose(importance['ess'], manifest['summary']['importance_ess']),
            'Recomputed importance ESS differs from inference manifest')
    diagnostics = diagnose(frame)
    diagnostics['mcmc_threshold_status_for_reference_only'] = diagnostics['status']
    diagnostics['status'] = np.where(diagnostics.status == 'constant_uninformative',
                                    'constant_uninformative', 'policy_dependence_diagnostic_only')
    diagnostics['mcse_distribution'] = 'learned_policy_unweighted_not_target_posterior'
    output.mkdir(parents=True)
    frame.to_csv(output/'traces.csv.gz')
    diagnostics.to_csv(output/'diagnostics.csv')
    selected = diagnostics.loc[list(GLOBAL_TRACES)]
    weights = normalized_weights(frame.log_importance_weight.to_numpy())
    summary = dict(draws=len(records), seed=manifest['seed'],
                   source_manifest=str(manifest_path.resolve()), source_manifest_sha256=sha256(manifest_path),
                   checkpoint=str(checkpoint.resolve()) if checkpoint else None,
                   checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
                   time_units='generations', diagnostic_positions=positions,
                   global_autocorrelation_diagnostics=selected.reset_index().to_dict('records'),
                   importance=importance,
                   importance_weighted_global_means={name: float(weights @ frame[name].to_numpy())
                                                     for name in GLOBAL_TRACES},
                   likelihood_max_abs_error=manifest['summary']['max_likelihood_error'],
                   interpretation=[
                       'Fresh draws from one fixed GFN checkpoint, in original generation order.',
                       'Autocorrelation ESS/MCSE describe unweighted learned-policy draws, not target-posterior accuracy.',
                       'High autocorrelation ESS can coexist with low importance ESS and biased target estimates.',
                       'No R-hat or posterior-convergence claim; the generic MCMC threshold status is reference-only.',
                       'Bulk ESS may exceed draw count due to negative estimated autocorrelation; it is not capped.',
                       'Weighted mean precision is not given by the unweighted MCSE columns.'],
                   file_hashes=hashes,
                   code_hashes={str(p): sha256(p) for p in [Path(__file__),
                       ROOT/'validation/scripts/evaluate_arginfer.py', ROOT/'eval/ess.py']})
    if reference_report:
        reference = pd.read_csv(reference_report/'diagnostics.csv').set_index('metric')
        comparison = selected[['draws', 'ess_bulk', 'ess_tail', 'ess_fraction']].join(
            reference.loc[list(GLOBAL_TRACES), ['draws', 'ess_bulk', 'ess_tail', 'ess_fraction']],
            lsuffix='_gfn_policy', rsuffix='_arginfer_mcmc')
        comparison.to_csv(output/'arginfer_comparison.csv')
        summary['reference_report'] = str(reference_report.resolve())
    write_json(output/'summary.json', summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), constrained_layout=True)
    for ax, name in zip(axes.flat, GLOBAL_TRACES):
        ax.plot(frame.index, frame[name], linewidth=.65)
        ax.set(title=name, xlabel='Fresh draw index')
    fig.savefig(output/'traces.png', dpi=150)
    plt.close(fig)
    lines = ['# GFN autocorrelation ESS audit', '',
             f'Checkpoint: `{checkpoint}`. Fresh draws: {len(records)}; seed: {manifest["seed"]}.', '',
             f'Importance ESS: **{importance["ess"]:.4f}/{len(records)}**.', '',
             '| Quantity | Autocorrelation bulk ESS | Tail ESS |', '|---|---:|---:|']
    for name, row in selected.iterrows():
        lines.append(f'| {name} | {row.ess_bulk:.2f} | {row.ess_tail:.2f} |')
    lines += ['', *[f'- {note}' for note in summary['interpretation']]]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    (output/'SUCCESS').write_text('Completed diagnostic calculation.\n')
    print(selected[['ess_bulk', 'ess_tail', 'mcse_mean']].to_string())
    print('Importance ESS:', importance['ess'], '/', len(records))
    print('Report:', output/'report.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--reference-report', type=Path)
    args = parser.parse_args()
    run(args.manifest, args.output_dir, args.checkpoint, args.reference_report)
