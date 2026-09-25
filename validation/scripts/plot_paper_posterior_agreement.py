#!/usr/bin/env python3
"""Figure 3: empirical posterior-summary agreement, using saved ARG draws.

TMRCA mass is uniform over draws/pairs and uniform in genomic position.
Clade RMSE uses exact spans and a shared local union across all three methods.
No truth values enter either agreement metric. See the generated README.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import hashlib
import json
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import tskit
from scipy.stats import wasserstein_distance
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from validation.scripts.evaluate_arginfer import (
    arg_to_ts, inventory, validate_inputs, check_mutations, extract_features, sha256)
from env.snp_data import load_snp_dataset

BASE = ROOT / 'validation/datasets/paper_datasets'
DEFAULT_OUT = ROOT / 'validation/paper_datasets/report/posterior_agreement'
METHODS = ('ARGFlow', 'ARGInfer', 'SINGER')
DATASETS = ('r1', 'r2', 'r4')
COLORS = {'ARGFlow': '#0072B2', 'ARGInfer': '#D55E00', 'SINGER': '#009E73'}
SCALE = 20000.
LENGTH = 25000.
N = 10


def sources(dataset, method, out):
    if method == 'ARGFlow':
        directory = (ROOT / 'validation/reports/gfn_arginfer_style/r1_best5900/draws'
                     if dataset == 'r1' else out / 'draws' / dataset)
        manifest = json.loads((directory / 'manifest.json').read_text())
        observations = load_snp_dataset(BASE / dataset / 'rep0')
        assert manifest['haplotype_ids'] == list(observations.haplotype_ids)
        assert len(manifest['samples']) == 256
        assert all(r['source'] == 'policy' and r['status'] == 'complete'
                   and r['temperature'] == 1 for r in manifest['samples'])
        paths = [directory / row['trees_file'] for row in manifest['samples']]
        extra = dict(manifest=str(directory / 'manifest.json'),
                     manifest_sha256=sha256(directory / 'manifest.json'),
                     seed=manifest['seed'], checkpoint_step={'r1': 5900, 'r2': 200, 'r4': 500}[dataset],
                     fresh_for_figure3=dataset != 'r1', importance_weighted=False)
    elif method == 'ARGInfer':
        files, _ = inventory(BASE / 'output/arginfer' / dataset / 'job_38078901')
        if dataset == 'r1':
            files = files[:256]
        paths = [p for _, p in files]
        extra = dict(iteration_first=files[0][0], iteration_last=files[-1][0],
                     additional_burnin=0, saved_iteration_spacing=1000)
    else:
        paths = sorted((BASE / 'output/singer' / dataset / 'trees').glob('trees_*.trees'),
                       key=lambda p: int(p.stem.split('_')[-1]))
        assert [int(p.stem.split('_')[-1]) for p in paths] == list(range(100, 1100))
        extra = dict(time_units='Generation units follow the existing SINGER converter/evaluation convention; input metadata may be unknown.')
    return paths, extra


def summarize_trees(trees):
    """Compress a mixture of trees into TMRCA masses and clade count-change tracks."""
    masses = defaultdict(float)
    events = defaultdict(lambda: defaultdict(int))
    count = 0
    for ts in trees:
        assert ts.num_samples == N and ts.sequence_length == LENGTH
        features = extract_features(ts)
        times = features.times[:, :-1] / SCALE
        weights = np.broadcast_to(np.diff(features.boundaries)[:, None] / (LENGTH * 45), times.shape)
        values, inverse = np.unique(times, return_inverse=True)
        for value, mass in zip(values, np.bincount(inverse.ravel(), weights=weights.ravel())):
            masses[float(value)] += float(mass)
        for left, right, clades in zip(features.boundaries[:-1], features.boundaries[1:], features.topologies):
            for clade in clades:
                events[clade][float(left)] += 1
                events[clade][float(right)] -= 1
        count += 1
        if count % 250 == 0:
            print(f'  summarized {count} draws', flush=True)
    values = np.array(sorted(masses))
    weights = np.array([masses[t] / count for t in values])
    assert np.isclose(weights.sum(), 1) and np.isfinite(values).all() and np.all(values >= 0)
    masks, positions, changes = [], [], []
    for mask, track in sorted(events.items()):
        for position, change in sorted(track.items()):
            if change:
                masks.append(mask); positions.append(position); changes.append(change)
    return dict(tmrca=values, weight=weights, mask=np.array(masks, dtype=np.int16),
                position=np.array(positions), change=np.array(changes, dtype=np.int32), draws=count)


def prepare(dataset, method, out):
    paths, extra = sources(dataset, method, out)
    # Fingerprint every input: cached summaries cannot silently outlive edited samples.
    files = [{'path': str(p.relative_to(ROOT)), 'sha256': sha256(p)} for p in paths]
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    cache = out / f'{dataset}_{method}.npz'
    provenance = out / f'{dataset}_{method}.json'
    if cache.exists() and provenance.exists():
        saved = json.loads(provenance.read_text())
        if saved['input_fingerprint'] == fingerprint:
            return dict(np.load(cache)), saved
    _, manifest, _, observed = validate_inputs(BASE / dataset / 'rep0', BASE / 'arginfer_inputs' / dataset)
    assert manifest['population_size'] == SCALE / 2
    observations = load_snp_dataset(BASE / dataset / 'rep0')
    def trees():
        for path in paths:
            if method == 'ARGInfer':
                with path.open('rb') as handle:
                    arg = pickle.load(handle)
                ts, raw, mapping = arg_to_ts(arg, N, LENGTH)
                check_mutations(arg, raw, mapping, observed, N)
            else:
                ts = tskit.load(path)
                assert np.array_equal(ts.samples(), np.arange(N))
                if method == 'SINGER':
                    assert ts.time_units in ('unknown', 'generations')
                    assert np.array_equal(ts.tables.sites.position, sorted(observed))
                    decoded = np.array([[int(v.alleles[g]) for g in v.genotypes] for v in ts.variants()])
                    target = np.array([[int(i in observed[pos]) for i in range(N)] for pos in sorted(observed)])
                    assert np.array_equal(decoded, target)
                    tables = ts.dump_tables(); tables.time_units = 'generations'
                    ts = tables.tree_sequence()
                else:
                    assert ts.time_units == 'generations'
                    for j, position in enumerate(observations.positions):
                        tree = ts.at(position)
                        derived = set(np.flatnonzero(observations.genotypes[:, j]))
                        assert any(set(tree.samples(node)) == derived for node in tree.nodes())
            yield ts
    print(f'Preparing {dataset} {method}: {len(paths)} draws', flush=True)
    data = summarize_trees(trees())
    np.savez_compressed(cache, **data)
    meta = dict(dataset=dataset, method=method, draws=int(data['draws']), files=files,
                input_fingerprint=fingerprint, protocol=extra)
    provenance.write_text(json.dumps(meta, indent=2) + '\n')
    return data, meta


def clade_tracks(data):
    tracks = {}
    for mask in np.unique(data['mask']):
        selected = data['mask'] == mask
        positions = data['position'][selected]
        counts = np.cumsum(data['change'][selected])
        assert counts[-1] == 0 and counts.min() >= 0 and counts.max() <= data['draws']
        tracks[int(mask)] = (positions, counts)
    return tracks


def compare_clades(ensembles, length=LENGTH, n=N):
    """Exact span integration over the shared local union, without dense genome grids."""
    tracks = [clade_tracks(d) for d in ensembles]
    draws = np.array([int(d['draws']) for d in ensembles])
    universe = set().union(*(t.keys() for t in tracks))
    accumulated = [defaultdict(float), defaultdict(float)]
    squared = np.zeros(2)
    union_mass = 0.
    events = 0
    for clade in sorted(universe):
        boundaries = np.unique(np.concatenate(([0., length], *[t[clade][0] for t in tracks if clade in t])))
        counts = np.zeros((len(boundaries)-1, 3), dtype=np.int32)
        for j, track in enumerate(tracks):
            if clade in track:
                positions, values = track[clade]
                indices = np.searchsorted(positions, boundaries[:-1], side='right') - 1
                valid = indices >= 0
                counts[valid, j] = values[indices[valid]]
        active = np.any(counts > 0, axis=1)
        spans = np.diff(boundaries)[active] / length
        counts = counts[active]
        probabilities = counts / draws
        union_mass += spans.sum()
        events += len(spans)
        for j in range(2):
            squared[j] += np.dot(spans, (probabilities[:, 0] - probabilities[:, j+1])**2)
            # Integer count pairs merge coincident scatter points exactly.
            codes = counts[:, 0].astype(np.int64) * (draws[j+1]+1) + counts[:, j+1]
            unique, inverse = np.unique(codes, return_inverse=True)
            for code, mass in zip(unique, np.bincount(inverse, weights=spans)):
                accumulated[j][int(code)] += float(mass)
    results = []
    for j in range(2):
        codes = np.array(sorted(accumulated[j]))
        mass = np.array([accumulated[j][int(k)] for k in codes])
        assert np.isclose(mass.sum(), union_mass)
        result = dict(reference=METHODS[j+1],
                      rmse=float(np.sqrt(squared[j] / union_mass)),
                      full_universe_rmse=float(np.sqrt(squared[j] / (2**n-n-2))),
                      local_union_mean_size=float(union_mass), interval_clade_events=events)
        results.append((dict(reference_probability=(codes % (draws[j+1]+1)) / draws[j+1],
                             argflow_probability=(codes // (draws[j+1]+1)) / draws[0],
                             weight=mass / union_mass), result))
    return results


def plot(out, summaries, comparisons):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(11.6, 6.7))
    grid = fig.add_gridspec(2, 3, left=.085, right=.98, top=.84, bottom=.115,
                           hspace=.73, wspace=.26, height_ratios=(1, .84))
    maximum = max(float(d['tmrca'].max()) for d in summaries.values())
    bin_width = .025
    bins = np.arange(0, maximum + 2*bin_width, bin_width)
    centers = (bins[:-1]+bins[1:])/2
    # Same bandwidth for all panels/methods; reflection respects TMRCA >= 0.
    bandwidth = .10
    clade_bins = np.linspace(0, 1, 26)
    clade_norm = LogNorm(vmin=1e-5, vmax=100)
    ymax = 0
    top_axes = []
    for col, dataset in enumerate(DATASETS):
        ax = fig.add_subplot(grid[0, col]); top_axes.append(ax)
        for method in METHODS:
            data = summaries[dataset, method]
            hist, _ = np.histogram(data['tmrca'], bins=bins, weights=data['weight'])
            assert np.isclose(hist.sum(), 1)
            density = gaussian_filter1d(hist / bin_width, bandwidth / bin_width, mode='reflect')
            ymax = max(ymax, density.max())
            ax.plot(centers, density, color=COLORS[method], lw=1.8, label=method)
        ax.set_title(dataset, fontsize=14, fontweight='semibold', pad=10)
        ax.set_xscale('symlog', linthresh=2, linscale=2, base=2)
        ax.set_xlabel(r'Pairwise TMRCA ($2N_e$ units)')
        ax.set_xlim(0, np.ceil(maximum))
        ticks = [x for x in [0, 1, 2, 4, 8, 16, 32, 64, 128] if x <= maximum]
        ax.set_xticks(ticks, labels=[str(x) for x in ticks])
        ax.axvline(2, color='#d6dce2', lw=.6, zorder=0)
        ax.text(.98, .97, '\n'.join(f'GFN–{ref}: $W_1$ = {comparisons[dataset,ref]["wasserstein_2Ne"]:.3f}'
                  for ref in METHODS[1:]), transform=ax.transAxes, ha='right', va='top', fontsize=8,
                  bbox=dict(facecolor='white', edgecolor='none', alpha=.88, pad=2))
        if col == 0:
            ax.set_ylabel('Posterior density')
        subgrid = grid[1, col].subgridspec(1, 2, wspace=.21)
        for j, reference in enumerate(METHODS[1:]):
            sub = fig.add_subplot(subgrid[0, j])
            data = np.load(out / f'{dataset}_clades_{reference}.npz')
            sub.plot([0,1], [0,1], color='#66717e', ls='--', lw=.85, zorder=3)
            mass, _, _ = np.histogram2d(
                data['reference_probability'], data['argflow_probability'],
                bins=(clade_bins, clade_bins), weights=100 * data['weight'])
            assert np.isclose(mass.sum(), 100)
            np.savez_compressed(out / f'{dataset}_clade_density_{reference}.npz',
                                edges=clade_bins, mass_percent=mass)
            mesh = sub.pcolormesh(clade_bins, clade_bins,
                                 np.ma.masked_equal(mass.T, 0), cmap='Blues',
                                 norm=clade_norm, rasterized=True, zorder=2)
            sub.set(xlim=(-.025,1.025), ylim=(-.025,1.025), aspect='equal',
                    xticks=[0,.5,1], yticks=[0,.5,1])
            sub.set_xticklabels(['0','.5','1']); sub.set_yticklabels(['0','.5','1'])
            sub.set_title(reference, color=COLORS[reference], fontsize=9, pad=6)
            sub.text(.04, .97, f'RMSE\n{comparisons[dataset,reference]["rmse"]:.3f}',
                     transform=sub.transAxes, va='top', fontsize=7,
                     bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1))
            if j:
                sub.set_yticklabels([])
            elif col == 0:
                sub.set_ylabel('ARGFlow clade probability', labelpad=5)
    for ax in top_axes:
        ax.set_ylim(0, ymax * 1.23)
    fig.text(.085,.913,'(a) Pairwise TMRCA posterior distributions',fontsize=10)
    fig.legend(*top_axes[0].get_legend_handles_labels(),loc='upper right',bbox_to_anchor=(.985,.952),
               ncol=3, frameon=False, fontsize=10)
    fig.text(.085,.435,'(b) Local-clade probability agreement',fontsize=10)
    color_ax = fig.add_axes([.685, .426, .285, .012])
    colorbar = fig.colorbar(mesh, cax=color_ax, orientation='horizontal',
                           ticks=[1e-4, 1e-2, 1, 100])
    colorbar.set_ticklabels(['0.0001', '0.01', '1', '100'])
    colorbar.ax.tick_params(labelsize=7, length=2)
    colorbar.ax.xaxis.set_label_position('top')
    colorbar.set_label('Genomic-span-weighted mass per bin (%)', fontsize=7, labelpad=4)
    fig.text(.54,.067,'Reference clade probability',ha='center',fontsize=10)
    fig.text(.54,.023,r'TMRCA axis: linear to 2, logarithmic above 2  ·  $2N_e = 20{,}000$ generations',
             ha='center',fontsize=8,color='#526171')
    for suffix in ('png','pdf','svg'):
        fig.savefig(out / f'figure3_posterior_agreement.{suffix}', dpi=250, facecolor='white')
    plt.close(fig)
    return dict(bin_width_2Ne=bin_width, gaussian_bandwidth_2Ne=bandwidth,
                density='Reflected Gaussian smoothing of weighted histogram; metrics use unsmoothed empirical masses.',
                axis_maximum_2Ne=float(np.ceil(maximum)), tail_truncation=False,
                xscale='symlog: linear from 0 to 2; base-2 logarithmic above 2; linscale=2',
                clade_rendering='25 x 25 density bins; exact genomic-span-weighted mass; empty bins white',
                clade_color_scale='Shared logarithmic scale, 0.00001 to 100 percent per bin')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--prepare-reference-only', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    summaries, metadata, comparisons = {}, [], {}
    for dataset in DATASETS:
        for method in METHODS:
            if args.prepare_reference_only and method == 'ARGFlow':
                continue
            data, meta = prepare(dataset, method, out)
            summaries[dataset, method] = data; metadata.append(meta)
        if args.prepare_reference_only:
            continue
        clades = compare_clades([summaries[dataset, method] for method in METHODS])
        for reference, (points, stats) in zip(METHODS[1:], clades):
            a, b = summaries[dataset,'ARGFlow'], summaries[dataset,reference]
            stats.update(dataset=dataset, wasserstein_2Ne=float(wasserstein_distance(
                a['tmrca'], b['tmrca'], a['weight'], b['weight'])))
            comparisons[dataset,reference] = stats
            np.savez_compressed(out / f'{dataset}_clades_{reference}.npz', **points)
            print(json.dumps(stats), flush=True)
    if args.prepare_reference_only:
        return
    settings = plot(out, summaries, comparisons)
    rows = list(comparisons.values())
    with (out / 'metrics.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    caption = ('**Figure 3: Posterior-summary agreement across inference methods.** '
        '**(a)** Pooled marginal pairwise TMRCA distributions from ARGFlow, ARGInfer, and SINGER '
        'for r1, r2, and r4. Each draw and each of the 45 haplotype pairs receives equal weight, '
        'with genomic positions weighted by exact interval length. Curves use a common Gaussian '
        'smoothing bandwidth. To display the full tail, the TMRCA axis is linear up to 2 and '
        'logarithmic above 2 (the faint vertical line marks the transition); densities remain '
        'per unit TMRCA. Annotations give the one-dimensional Wasserstein distance between '
        'the unsmoothed ARGFlow (GFN) and reference empirical distributions, calculated on '
        'original linear TMRCA values in 2Ne units (Ne = 10,000). '
        '**(b)** Local-clade posterior probabilities for ARGFlow versus each reference sampler; '
        'the dashed diagonal denotes equality. Both comparisons use the shared local union of '
        'nontrivial rooted clades observed in any of the three methods, with missing clades '
        'assigned probability zero. Color shows genomic-span-weighted mass in 0.04-by-0.04 '
        'probability bins on a shared logarithmic color scale; empty bins are white. '
        'RMSE uses unbinned probabilities and weights each clade-interval event by its genomic span, '
        'normalized over that shared union. These are marginal-summary comparisons, not a test '
        'of equality of the full ARG posterior.')
    notes = ('ARGFlow uses 256 unweighted policy draws per dataset at the Figure 2 checkpoints '
        '(r1: 5,900; r2: 200; r4: 500). r1 reuses the existing draw set; r2/r4 use fresh draws '
        '(seed 20260923, batch size 16) from the frozen source used for training. Figure 2 r2/r4 '
        'instead combined three evaluation repeats of 256 draws, so its saved posterior means '
        'are not reused as samples here. ARGInfer uses the same selection as Figure 2: the '
        'first 256 retained draws for r1 and all 1,800 retained draws for r2/r4, without '
        'additional burn-in. SINGER uses saved draws 100–1099 (1,000 draws per dataset). '
        'Draw counts are not effective sample sizes. Input sample order and SNP compatibility '
        'are checked; SINGER times follow the existing converter/evaluation convention despite '
        'unknown input time-unit metadata. SNP coordinate conventions and model targets differ '
        'across methods. Convergence is not established by this figure. No truth TMRCA or '
        'truth clade probabilities enter the agreement metrics.\n\n'
        'For each nontrivial clade, the union of posterior breakpoints partitions the genome '
        'into intervals with constant probabilities. The shared local-union RMSE is '
        '`sqrt(sum_c integral 1[c in U(x)] (p_ARGFlow-p_ref)^2 dx / integral |U(x)| dx)`, '
        'where U(x) includes clades seen in any of the three ensembles. Clades absent from '
        'a method have probability zero. Thus reference-only clades are included; the same '
        'denominator is used for both reference comparisons within a dataset. Linked clade '
        'events are not treated as independent replicates. The CSV also reports '
        'RMSE over all 1,012 possible nontrivial clades (including all-zero events); that '
        'alternative denominator is not used in panel annotations.\n\n'
        'Density smoothing: reflected Gaussian smoothing of a 0.025-wide histogram with '
        'bandwidth 0.10 in 2Ne units, identical for all methods and datasets. The full TMRCA '
        'range is shown on an axis linear up to 2 and base-2 logarithmic above 2. Density is '
        'per unit TMRCA, so visual area on this nonlinear axis is not probability mass. '
        'Wasserstein distances use exact empirical masses, not smoothed curves.')
    (out / 'caption.md').write_text(caption + '\n')
    (out / 'README.md').write_text('# Figure 3: posterior agreement\n\n' + caption + '\n\n' + notes +
        '\n\nReproduce after generating the documented fresh draws:\n\n```bash\n'
        '/private/home/pkatte/anaconda3/envs/phylogfn_orig/bin/python validation/scripts/plot_paper_posterior_agreement.py\n```\n\n'
        'Per-method NPZs preserve empirical TMRCA masses and exact clade count-change tracks. '
        'Comparison NPZs preserve unbinned probability coordinates and normalized span weights; '
        'clade_density NPZs preserve the displayed bin edges and mass percentages. '
        'Per-method JSON files hash every source draw. See sampling_provenance.json for '
        'fresh-draw commands, checkpoint hashes, and frozen source hashes.\n')
    (out / 'provenance.json').write_text(json.dumps(dict(methods=metadata, metrics=rows, plot=settings,
        script_sha256=sha256(Path(__file__))), indent=2) + '\n')
    print(f'Figure complete: {out}', flush=True)


if __name__ == '__main__':
    main()
