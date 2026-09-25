#!/usr/bin/env python3
"""Reproduce the requested r1 comparison of 256 retained ARGinfer and raw GFN draws."""
import argparse
from collections import Counter
import gzip
import itertools
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from validation.scripts.evaluate_arginfer import (
    LEVELS, GLOBAL_TRACES, arg_to_ts, check_mutations, diagnose, extract_features,
    inventory, require, sha256, summarize_posterior, validate_inputs, write_json)
from env.snp_data import load_snp_dataset
from eval.ess import importance_stats
import numpy as np
import pandas as pd
import tskit


def clade_scores(truth, draws, n, output):
    """Fixed-universe Brier: both methods score the same 2^n-n-2 clade events.

    Unlike averaging over each method's observed-clade union, this denominator
    cannot reward a method merely for producing more distinct false clades.
    """
    universe = [mask for mask in range(1, (1 << n)-1) if mask.bit_count() > 1]
    accumulated = {mask: np.zeros(3) for mask in universe}
    boundaries = np.unique(np.concatenate([truth.boundaries, *[d.boundaries for d in draws]]))
    positions = (boundaries[:-1]+boundaries[1:])/2
    indices = [d.at(positions) for d in draws]
    truth_indices = truth.at(positions)
    length = boundaries[-1]
    sum_brier = 0.
    with gzip.open(output/'clade_probabilities.jsonl.gz', 'wt') as handle:
        for j, position in enumerate(positions):
            counts = Counter()
            for d, idx in zip(draws, indices):
                counts.update(d.topologies[idx[j]])
            true = set(truth.topologies[truth_indices[j]])
            support = {c: count/len(draws) for c, count in counts.items()}
            span_fraction = (boundaries[j+1]-boundaries[j])/length
            for mask in counts.keys() | true:
                probability = support.get(mask, 0.)
                expected = int(mask in true)
                error = (probability-expected)**2
                accumulated[mask] += span_fraction*np.array([probability, expected, error])
                sum_brier += span_fraction*error
            handle.write(json.dumps(dict(left=float(boundaries[j]), right=float(boundaries[j+1]),
                true_clades=sorted(true), probabilities=support), separators=(',', ':'))+'\n')
    rows = [dict(clade_mask=mask, haplotype_indices=','.join(str(i) for i in range(n) if mask & (1 << i)),
                 mean_probability=values[0], truth_genome_fraction=values[1],
                 span_weighted_brier=values[2]) for mask, values in accumulated.items()]
    pd.DataFrame(rows).to_csv(output/'clade_scores.csv', index=False)
    require(np.isclose(sum_brier, sum(row['span_weighted_brier'] for row in rows)), 'Brier sum check failed')
    return dict(clade_universe_size=len(universe), clade_brier_sum=sum_brier,
                clade_brier_fixed_universe=sum_brier/len(universe),
                definition='Genomic mean of sum_c (P(c)-1[c in truth])^2 over all nontrivial rooted clades; fixed-universe score divides by universe size.')


def comparison_plots(summaries, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, quantity in zip(axes, ('pair_tmrca', 'root_time')):
        for offset, (name, summary) in zip((-.2, .2), summaries.items()):
            rank = summary['ranks'][quantity]
            ax.bar(np.arange(len(rank['probabilities']))+offset, rank['probabilities'], width=.4, label=name)
        ax.plot(rank['uniform_probabilities'], 'k--', label='Uniform rank reference')
        ax.set(title=quantity, xlabel='Truth rank bin (20 bins)', ylabel='Genomic/pair probability mass')
        ax.legend()
    for suffix in ('png', 'pdf'):
        fig.savefig(output/f'rank_histograms.{suffix}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for j, quantity in enumerate(('pair_tmrca', 'root_time')):
        for name, summary in summaries.items():
            axes[0, j].plot(LEVELS, [summary[quantity][f'coverage_{round(l*100)}'] for l in LEVELS], 'o-', label=name)
            axes[1, j].plot(LEVELS, [summary[quantity][f'width_{round(l*100)}'] for l in LEVELS], 'o-', label=name)
        axes[0, j].plot(LEVELS, LEVELS, 'k--', label='Nominal')
        axes[0, j].set(title=quantity, ylabel='Truth coverage', xlabel='Nominal probability', ylim=(0, 1))
        axes[1, j].set(ylabel='Mean interval width (generations)', xlabel='Nominal probability')
        for ax in axes[:, j]:
            ax.legend()
    for suffix in ('png', 'pdf'):
        fig.savefig(output/f'interval_coverage_width.{suffix}', dpi=180)
    plt.close(fig)


def run(output):
    require(not output.exists(), f'Output exists: {output}')
    base = ROOT/'paper/datasets'
    comparison_root = ROOT/'validation/reports/gfn_arginfer_style/r1_best5900'
    metadata, inputs, truth_ts, data = validate_inputs(base/'r1/rep0', base/'r1/arginfer_inputs')
    observations = load_snp_dataset(base/'r1/rep0')
    manifest_path = comparison_root/'draws/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    require(manifest['haplotype_ids'] == list(observations.haplotype_ids), 'GFN sample order differs from truth/input')
    records = manifest['samples']
    require(len(records) == 256 and [r['index'] for r in records] == list(range(256)), 'Expected original 256 GFN draws')
    arg_files, _ = inventory(ROOT/'paper/outputs/ARGInfer/r1')
    arg_files = arg_files[:256]
    n, length = inputs['num_haplotypes'], inputs['sequence_length']
    pairs = list(itertools.combinations(range(n), 2))
    truth = extract_features(truth_ts, metadata['sample_nodes_in_haplotype_order'])
    ensembles, hashes = {'ARGinfer': [], 'GFN': []}, {}
    for i, (iteration, path) in enumerate(arg_files):
        with path.open('rb') as handle:
            arg = pickle.load(handle)
        arg.verify()
        ts, raw, mapping = arg_to_ts(arg, n, length)
        check_mutations(arg, raw, mapping, data, n)
        ensembles['ARGinfer'].append(extract_features(ts))
        hashes[str(path)] = sha256(path)
    for row in records:
        require(row['source'] == 'policy' and row['status'] == 'complete' and row['temperature'] == 1., 'Invalid GFN sampling protocol')
        path = manifest_path.parent/row['trees_file']
        ts = tskit.load(path)
        require(ts.num_samples == n and ts.sequence_length == length, 'GFN tree dimensions differ')
        # Check GFN mutations against its exact input coordinates, not rounded ARGinfer positions.
        for j, position in enumerate(observations.positions):
            tree = ts.at(position)
            samples = list(ts.samples())
            derived = {samples[k] for k in np.flatnonzero(observations.genotypes[:, j])}
            require(any(set(tree.samples(node)) == derived for node in tree.nodes()), 'Incompatible GFN SNP')
        ensembles['GFN'].append(extract_features(ts))
        hashes[str(path)] = sha256(path)
    output.mkdir(parents=True)
    summaries = {}
    for name, features in ensembles.items():
        print(f'{name}: evaluating 256 equally weighted draws', flush=True)
        destination = output/name.lower()
        destination.mkdir()
        summaries[name] = summarize_posterior(truth, features, pairs, destination)
        summaries[name]['topology']['rf_normalized'] = summaries[name]['topology']['rf']/(2*(n-2))
        summaries[name]['topology'].update(clade_scores(truth, features, n, destination))
        write_json(destination/'summary.json', summaries[name])
    # Recompute ARGinfer chain diagnostics on the exact selected iteration IDs.
    at = pd.read_csv(ROOT/'validation/reports/arginfer/r1/traces.csv.gz', index_col='iteration')
    at = at.loc[[iteration for iteration, _ in arg_files], list(GLOBAL_TRACES)]
    ad = diagnose(at)
    ad.to_csv(output/'arginfer_ess.csv')
    importance = importance_stats([r['log_importance_weight'] for r in records])
    require(np.isclose(importance['ess'], manifest['summary']['importance_ess']), 'Importance ESS mismatch')
    integer_positions = np.loadtxt(base/'r1/arginfer_inputs/positions.txt')
    protocol = dict(dataset='r1/rep0', draws_per_method=256, weights='equal within each method; GFN unweighted',
                    arginfer_iterations=[arg_files[0][0], arg_files[-1][0]], arginfer_spacing=1000,
                    gfn_checkpoint_step=5900, gfn_seed=manifest['seed'],
                    sample_order=manifest['haplotype_ids'], time_units='generations',
                    genealogy_normalization='Simplify both methods to sampled genealogies, dropping unary nodes and ancestry above the local MRCA.',
                    weighting='Exact genomic spans and equal haplotype pairs',
                    rmse='RMSE of posterior-mean TMRCA against truth, not mean per-draw RMSE',
                    rf='Expected per-draw rooted RF against truth, exact genomic weighting; normalized by 2(n-2)',
                    ranks='20 equal-rank-count bins over ranks 0..256; deterministic uniform tie breaking',
                    topology_sets='Empirical highest-frequency sets including all cutoff ties; not unseen-mass-corrected',
                    clade_brier='Common full nontrivial-clade universe, avoiding method-dependent denominators',
                    differing_snp_coordinates=int(np.count_nonzero(integer_positions != observations.positions)),
                    max_coordinate_difference_bp=float(np.max(abs(integer_positions-observations.positions))),
                    limitations=['One dataset; linked genomic positions and pairs are not independent replicates.',
                                 'One ARGinfer chain; R-hat unavailable, convergence not established.',
                                 'Different SNP coordinate conventions persist; effect not quantified.',
                                 '256 draws may miss substantial topology probability mass.'])
    summary = dict(protocol=protocol, methods=summaries,
                   numerical_reliability=dict(arginfer=ad.reset_index().to_dict('records'),
                                              arginfer_rhat=None, gfn_importance=importance))
    write_json(output/'summary.json', summary)
    write_json(output/'provenance.json', dict(files=hashes,
        truth_sha256=sha256(base/'r1/rep0/r1.trees'), gfn_manifest_sha256=sha256(manifest_path),
        code={str(p): sha256(p) for p in [Path(__file__), ROOT/'validation/scripts/evaluate_arginfer.py']}))
    rows = []
    def add(question, label, function):
        rows.append(dict(question=question, metric=label, **{name: function(s) for name, s in summaries.items()}))
    for quantity in ('pair_tmrca', 'root_time'):
        add('Are estimated genealogies accurate?', f'{quantity} RMSE (generations)', lambda s, q=quantity: s[q]['mean_rmse'])
    add('Are estimated genealogies accurate?', 'Rooted RF (raw)', lambda s: s['topology']['rf'])
    add('Are estimated genealogies accurate?', 'Rooted RF (normalized)', lambda s: s['topology']['rf_normalized'])
    for quantity in ('pair_tmrca', 'root_time'):
        for level in LEVELS:
            k = round(level*100)
            for metric in ('coverage', 'width'):
                add('Are uncertainty estimates useful?', f'{quantity} {k}% {metric}', lambda s, q=quantity, k=k, m=metric: s[q][f'{m}_{k}'])
        add('Are uncertainty estimates useful?', f'{quantity} rank KL (nats)', lambda s, q=quantity: s['ranks'][q]['kl_from_uniform'])
    for metric in ('true_clade_support', 'clade_brier_sum', 'clade_brier_fixed_universe', 'truth_probability', 'unique_topologies'):
        add('Is topology uncertainty represented?', metric, lambda s, m=metric: s['topology'][m])
    for level in LEVELS:
        k = round(level*100)
        for metric in ('truth_covered', 'mass', 'size'):
            add('Is topology uncertainty represented?', f'Topology set {k}% {metric}', lambda s, k=k, m=metric: s['topology'][f'credible_{k}_{m}'])
    pd.DataFrame(rows).to_csv(output/'comparison.csv', index=False)
    comparison_plots(summaries, output)
    lines = ['# r1: 256-draw ARG-summary comparison', '',
             'ARGinfer iterations 201000–456000; GFN best checkpoint 5900, seed 20260922. All draws have equal weight. Both representations are simplified to sampled genealogies; ancestry above the local MRCA is excluded.', '',
             '| Question | Metric | ARGinfer | GFN (unweighted) |', '|---|---|---:|---:|']
    for row in rows:
        lines.append(f'| {row["question"]} | {row["metric"]} | {row["ARGinfer"]:.6g} | {row["GFN"]:.6g} |')
    for name in GLOBAL_TRACES:
        lines.append(f'| Are estimates numerically reliable? | {name} MCMC bulk/tail ESS | {ad.loc[name, "ess_bulk"]:.2f} / {ad.loc[name, "ess_tail"]:.2f} | Not the GFN target-fit diagnostic |')
    lines += [f'| Are estimates numerically reliable? | Importance ESS | Not applicable | {importance["ess"]:.4f}/256 |',
              '| Are estimates numerically reliable? | R-hat | Unavailable: one chain | Not applicable to posterior convergence of the learned policy |', '',
              'Coverage and probabilities are fractions in this table; widths are generations. Lower RMSE/RF/Brier is better. Coverage should be considered jointly with width.', '',
              'Clade Brier scores in the main comparison use the same full universe of 1012 clades for both methods. Per-method observed-union Brier is retained only in individual JSON summaries.', '',
              *[f'- {note}' for note in protocol['limitations']]]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    (output/'SUCCESS').write_text('Completed equal-count, unweighted ARG-summary comparison.\n')
    print('Completed:', output/'report.md', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args().output_dir)
