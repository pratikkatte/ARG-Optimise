"""Build exact span-weighted pairwise TMRCA panels from existing saved samples.

Run with the phylogfn_orig environment. Does not run inference or change samples.
"""
from pathlib import Path
import csv
import gzip
import hashlib
import itertools
import json
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import tskit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from validation.scripts.evaluate_arginfer import arg_to_ts, inventory, validate_inputs, check_mutations

BASE = ROOT / 'validation/datasets/paper_datasets'
OUT = ROOT / 'validation/paper_datasets/report/tmrca_reconstruction'
METHODS = ['ARGFlow', 'ARGInfer', 'SINGER']
DATASETS = ['r1', 'r2', 'r4']


def track(ts, samples=None):
    samples = list(ts.samples()) if samples is None else samples
    assert len(samples) == 10 and ts.sequence_length == 25000
    pairs = list(itertools.combinations(samples, 2))
    values = []
    for tree in ts.trees():
        assert tree.num_roots == 1
        values.append([tree.tmrca(a, b) / 20000 for a, b in pairs])
    return np.asarray(list(ts.breakpoints())), np.asarray(values)


def exact_mean(truth, tracks):
    boundaries = np.unique(np.concatenate([truth[0]] + [b for b, _ in tracks]))
    # Sum piecewise-constant changes, avoiding a draw x interval x pair array.
    delta = np.zeros((len(boundaries), 45))
    for b, values in tracks:
        changes = np.vstack([values[0], np.diff(values, axis=0), -values[-1]])
        delta[np.searchsorted(boundaries, b)] += changes / len(tracks)
    mean = np.cumsum(delta, axis=0)[:-1]
    midpoint = (boundaries[:-1] + boundaries[1:]) / 2
    expected = truth[1][np.searchsorted(truth[0][1:], midpoint, side='right')]
    return expected, mean, np.diff(boundaries)


def save_panel(dataset, method, arrays, sources, expected_rmse=None, draws=None):
    x, y, weights = [], [], []
    repeat_rmse = []
    for truth, mean, spans in arrays:
        assert truth.shape == mean.shape and truth.shape[1] == 45
        assert np.isclose(spans.sum(), 25000)
        w = np.broadcast_to(spans[:, None] / (25000 * 45), truth.shape).copy()
        assert np.isfinite(mean).all() and np.isfinite(truth).all()
        assert mean.min() > -1e-9 and truth.min() >= 0
        repeat_rmse.append(float(np.sqrt(np.sum(w * (mean - truth)**2))))
        x.append(truth.ravel()); y.append(np.maximum(mean, 0).ravel())
        weights.append(w.ravel() / len(arrays))
    x, y, weights = map(np.concatenate, (x, y, weights))
    assert np.isclose(weights.sum(), 1)
    rmse = float(np.mean(repeat_rmse))
    if expected_rmse is not None:
        assert np.isclose(rmse, expected_rmse, rtol=1e-8, atol=1e-10), (dataset, method, rmse, expected_rmse)
    np.savez_compressed(OUT / f'{dataset}_{method}.npz', truth=x, posterior_mean=y, weight=weights)
    result = dict(dataset=dataset, method=method, rmse_2Ne=rmse,
                  repeat_rmse_2Ne=repeat_rmse, draws=draws,
                  observations=len(x), maximum=float(max(x.max(), y.max())),
                  sources=[str(p.relative_to(ROOT)) for p in sources])
    print(json.dumps({k: v for k, v in result.items() if k != 'sources'}), flush=True)
    return result


def prepare():
    results = []
    for dataset in DATASETS:
        metadata, manifest, truth_ts, observed = validate_inputs(BASE / dataset / 'rep0', BASE / 'arginfer_inputs' / dataset)
        assert manifest['population_size'] == 10000
        truth = track(truth_ts, metadata['sample_nodes_in_haplotype_order'])
        for method in METHODS:
            print(f'Preparing {dataset} {method}', flush=True)
            cache = OUT / f'{dataset}_{method}.json'
            if cache.exists():
                results.append(json.loads(cache.read_text())); continue
            if method == 'ARGFlow' and dataset != 'r1':
                run, step = ('pilot_20260921_a', 200) if dataset == 'r2' else ('pilot_20260921_b_extended_r4', 500)
                directory = ROOT / 'runs/paper_datasets_stable' / run / dataset / 'evaluation'
                arrays, sources, expected = [], [], []
                for repeat in range(3):
                    p = directory / f'step_{step:06d}_repeat_{repeat:02d}.json.gz'
                    with gzip.open(p, 'rt') as f: report = json.load(f)
                    exact = report['details']['truth']['pair_tmrca_exact']
                    assert exact['time_divisor'] == 20000
                    arrays.append((np.asarray(exact['truth']), np.asarray(exact['mean']), np.diff(exact['boundaries'])))
                    expected.append(report['metrics']['eval_truth_pair_tmrca_rmse'])
                    sources.append(p)
                    del report
                result = save_panel(dataset, method, arrays, sources, np.mean(expected), '3 x 256; repeat densities equally weighted')
            else:
                tracks, sources, expected = [], [], None
                if method == 'ARGInfer':
                    files, _ = inventory(BASE / 'output/arginfer' / dataset / 'job_38078901')
                    if dataset == 'r1': files = files[:256]
                    for i, (_, p) in enumerate(files):
                        with p.open('rb') as f: arg = pickle.load(f)
                        ts, raw, mapping = arg_to_ts(arg, 10, 25000)
                        check_mutations(arg, raw, mapping, observed, 10)
                        tracks.append(track(ts)); sources.append(p)
                        if (i + 1) % 300 == 0: print(f'  {i+1}/{len(files)} ARG samples', flush=True)
                    if dataset == 'r1': expected = 8369.505471988607 / 20000
                    if dataset == 'r4': expected = 6826.829845441732 / 20000
                elif method == 'SINGER':
                    directory = BASE / 'output/singer' / dataset
                    files = sorted((directory / 'trees').glob('trees_*.trees'), key=lambda p: int(p.stem.split('_')[-1]))
                    assert [int(p.stem.split('_')[-1]) for p in files] == list(range(100, 1100))
                    for p in files:
                        ts = tskit.load(p)
                        assert ts.time_units in ('unknown', 'generations')
                        assert np.array_equal(ts.samples(), np.arange(10))
                        # Check integer SNP/sample alignment against ARGinfer's validated VCF input.
                        assert np.array_equal(ts.tables.sites.position, sorted(observed))
                        variants = list(ts.variants())
                        decoded = np.array([[int(v.alleles[g]) for g in v.genotypes] for v in variants])
                        target = np.array([[int(i in observed[pos]) for i in range(10)] for pos in sorted(observed)])
                        assert np.array_equal(decoded, target)
                        tracks.append(track(ts)); sources.append(p)
                    p = directory / 'trees/point_accuracy/singercommonMetrics.tsv'
                    values = {r['metric']: r['value'] for r in csv.DictReader(p.open(), delimiter='\t')}
                    expected = float(values['weighted_rmse'])
                else:
                    directory = ROOT / 'validation/reports/gfn_arginfer_style/r1_best5900/draws'
                    manifest = json.loads((directory / 'manifest.json').read_text())
                    for row in manifest['samples']:
                        p = directory / row['trees_file']
                        tracks.append(track(tskit.load(p))); sources.append(p)
                    assert len(tracks) == 256
                    expected = 12573.107959479088 / 20000
                result = save_panel(dataset, method, [exact_mean(truth, tracks)], sources, expected, len(tracks))
            cache.write_text(json.dumps(result, indent=2) + '\n')
            results.append(result)
    return results


def plot(results):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    bin_width = .05
    maximum = max(r['maximum'] for r in results)
    limit = np.ceil(maximum * 1.025 * 2) / 2
    bins = np.arange(0, limit + bin_width / 2, bin_width)
    histograms = {}
    for row in results:
        key = row['dataset'], row['method']
        data = np.load(OUT / f'{key[0]}_{key[1]}.npz')
        h, _, _ = np.histogram2d(data['truth'], data['posterior_mean'], bins=[bins, bins], weights=data['weight'] * 100)
        assert np.isclose(h.sum(), 100)
        histograms[key] = h
    vmax = max(h.max() for h in histograms.values())
    norm = LogNorm(vmin=.001, vmax=max(10, np.ceil(vmax)))
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('white')
    fig, axes = plt.subplots(3, 3, figsize=(11.1, 10.2), sharex='row', sharey='row')
    fig.subplots_adjust(left=.12, right=.87, bottom=.105, top=.885, hspace=.16, wspace=.13)
    for i, dataset in enumerate(DATASETS):
        row_limit = np.ceil(max(r['maximum'] for r in results if r['dataset'] == dataset) * 1.025 * 2) / 2
        for j, method in enumerate(METHODS):
            ax = axes[i, j]; h = histograms[dataset, method]
            mesh = ax.pcolormesh(bins, bins, np.ma.masked_where(h.T == 0, h.T), cmap=cmap, norm=norm, rasterized=True)
            ax.plot([0, limit], [0, limit], '--', color='#d64b45', linewidth=1.05, alpha=.9)
            ax.set(xlim=(0, row_limit), ylim=(0, row_limit), aspect='equal')
            ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.tick_params(length=3, colors='#465363', labelsize=9)
            for spine in ax.spines.values(): spine.set_color('#b6c0ca')
            row = next(r for r in results if r['dataset'] == dataset and r['method'] == method)
            label = 'Mean RMSE' if len(row['repeat_rmse_2Ne']) > 1 else 'RMSE'
            ax.text(.04, .96, f'{label} = {row["rmse_2Ne"]:.3f}', transform=ax.transAxes, va='top', fontsize=9, bbox=dict(facecolor='white', edgecolor='none', alpha=.88, pad=2))
            if i == 0: ax.set_title(method, fontsize=15, fontweight='semibold', pad=13, color='#172b42')
            if j == 0: ax.text(-.22, .5, dataset, transform=ax.transAxes, va='center', ha='center', fontsize=14, fontweight='semibold', color='#172b42')
    cax = fig.add_axes([.90, .28, .017, .40])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label('Genomic-span-weighted mass per bin (%)', labelpad=10)
    fig.supxlabel(r'True pairwise TMRCA ($2N_e$ units)', y=.058, fontsize=12)
    fig.supylabel(r'Posterior mean pairwise TMRCA ($2N_e$ units)', x=.025, fontsize=12)
    fig.suptitle('Pairwise TMRCA reconstruction', fontsize=18, fontweight='semibold', y=.976, color='#172b42')
    fig.text(.49, .943, 'Simulated truth versus posterior mean  •  dashed line: exact agreement', ha='center', fontsize=10, color='#526171')
    fig.text(.49, .022, r'45 haplotype pairs  •  axes shared within each row  •  $2N_e = 20{,}000$ generations', ha='center', fontsize=9, color='#526171')
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'pairwise_tmrca_reconstruction.{extension}', dpi=220, facecolor='white')
    plt.close(fig)
    return {'axis_limits_2Ne': {ds: float(np.ceil(max(r['maximum'] for r in results if r['dataset'] == ds) * 1.025 * 2) / 2) for ds in DATASETS}, 'bin_width_2Ne': bin_width, 'color_scale': 'logarithmic probability mass percent per bin; normalized separately per panel', 'density_floor_percent': .001}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = prepare()
    settings = plot(results)
    (OUT / 'provenance.json').write_text(json.dumps({'panels': results, 'plot': settings, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + '\n')
    caption = ('Pairwise TMRCA reconstruction against simulated truth. Posterior mean pairwise TMRCA is plotted against the generating TMRCA for ARGFlow, ARGInfer, and SINGER (columns) across r1, r2, and r4 (rows). The dashed diagonal denotes exact agreement. Color shows genomic-span-weighted probability mass per 0.05-by-0.05 bin, normalized within each panel, with equal weight for each of the 45 haplotype pairs and a shared logarithmic color scale. Both axes are in 2Ne units (Ne = 10,000).')
    notes = ('r1 ARGFlow and ARGInfer use the corrected 256-draw comparison (GFN step 5,900). r2/r4 ARGFlow show equally weighted mixtures of the three repeat-specific posterior-mean reconstructions at steps 200/500, respectively; their annotations report the mean of repeat RMSEs, matching the table. These are not posterior means pooled across repeats. SINGER uses 1,000 saved draws per dataset. ARGInfer uses 1,800 saved draws for r2/r4, with no additional burn-in. Draw counts and model targets differ across methods. r2 ARGInfer was newly summarized from existing raw ARG samples; no inference was run. Its absence from the earlier table meant no saved metric report had been found. All eight previously available RMSE values were reproduced to numerical tolerance. SINGER generation units follow the existing converter/evaluation convention despite unknown time-unit metadata; SNP coordinates and sample-genotype ordering were checked for every sample.')
    (OUT / 'README.md').write_text('# Pairwise TMRCA reconstruction\n\n' + caption + '\n\n' + notes + '\n\nReproduce with:\n\n```bash\n/private/home/pkatte/anaconda3/envs/phylogfn_orig/bin/python validation/scripts/plot_paper_tmrca_reconstruction.py\n```\n\nPanel NPZ files contain flattened truth, posterior mean, and normalized span/pair weights. Per-panel JSON and provenance.json record the exact sources.\n')
    print('Figure complete:', OUT, flush=True)


if __name__ == '__main__':
    main()
