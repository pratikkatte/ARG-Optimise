"""Reproduce manuscript Figure 2 from the configured saved posterior draws."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
import tskit

from validation.scripts.paper_datasets.evaluate import (
    load_draws, resolve, require, sha256, write_json)
from validation.scripts.evaluate_arginfer import extract_features
from env.snp_data import load_snp_dataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, LogFormatterMathtext


def exact_mean(truth, draws, ne):
    """Integrate changes in each draw, avoiding a draw x position x pair tensor."""
    require(bool(draws) and ne > 0, 'Draws and positive Ne are required')
    boundaries = np.unique(np.concatenate([truth.boundaries, *[d.boundaries for d in draws]]))
    pairs = truth.times.shape[1] - 1
    changes = np.zeros((len(boundaries), pairs))
    for draw in draws:
        values = draw.times[:, :-1] / (2*ne)
        delta = np.vstack([values[0], np.diff(values, axis=0), -values[-1]])
        changes[np.searchsorted(boundaries, draw.boundaries)] += delta / len(draws)
    mean = np.cumsum(changes, axis=0)[:-1]
    require(np.isfinite(mean).all() and mean.min() >= -1e-10, 'Invalid posterior mean')
    mean = np.maximum(mean, 0.)
    actual = truth.times[truth.at(boundaries[:-1]), :-1] / (2*ne)
    spans = np.diff(boundaries)
    weights = np.broadcast_to(spans[:, None] / (spans.sum()*pairs), actual.shape)
    return actual.ravel(), mean.ravel(), weights.ravel()


def panel_statistics(x, y, weights, bins):
    require(np.isfinite(x).all() and np.isfinite(y).all(), 'Nonfinite TMRCA')
    require(np.isclose(weights.sum(), 1) and np.all(weights >= 0), 'Invalid weights')
    require(x.min() >= bins[0] and x.max() <= bins[-1], 'Truth outside configured axis range')
    require(y.min() >= 0 and y.max() <= bins[-1], 'Inferred upper tail outside configured axis range')
    mx, my = np.sum(weights*x), np.sum(weights*y)
    variance_product = np.sum(weights*(x-mx)**2)*np.sum(weights*(y-my)**2)
    correlation = (float(np.sum(weights*(x-mx)*(y-my))/np.sqrt(variance_product))
                   if variance_product > 0 else None)
    histogram, _, _ = np.histogram2d(x, y, bins=[bins, bins], weights=weights*100)
    below = y < bins[0]
    underflow, _ = np.histogram(x[below], bins=bins, weights=weights[below]*100)
    require(np.isclose(histogram.sum()+underflow.sum(), 100), 'Plot lost probability mass')
    return dict(rmse_2Ne=float(np.sqrt(np.sum(weights*(x-y)**2))),
                pearson_r=correlation, below_range_percent=float(underflow.sum())), histogram, underflow


def prepare_panel(dataset, settings, source, config, output, bins):
    directory = resolve(settings['dataset_dir'])
    metadata_path = directory/'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    require(metadata['dataset_name'] == dataset, 'Dataset metadata mismatch')
    truth_path = directory/metadata['files']['ground_truth_trees']
    observations = load_snp_dataset(directory)
    truth_ts = tskit.load(truth_path)
    order = metadata['sample_nodes_in_haplotype_order']
    np.testing.assert_array_equal(truth_ts.tables.sites.position, observations.positions)
    for j, variant in enumerate(truth_ts.variants(samples=order)):
        np.testing.assert_array_equal(
            [variant.alleles[g] != variant.site.ancestral_state for g in variant.genotypes],
            observations.genotypes[:, j])
    truth = extract_features(truth_ts, order)
    draws, provenance = load_draws(dataset, directory, observations, source, config)
    ne = float(metadata['parameters']['population_size'])
    x, y, weights = exact_mean(truth, draws, ne)
    stats, histogram, underflow = panel_statistics(x, y, weights, bins)
    checkpoint = Path(source['directory']).name
    key = f'{dataset}_{source["method"]}_{checkpoint}'
    reference_path = resolve(config['output_dir'])/dataset/source['method']/checkpoint/'results.json'
    reference_check = None
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        require(reference['truth']['sha256'] == sha256(truth_path), 'Table truth differs from figure truth')
        require(reference['provenance']['files'] == provenance['files'], 'Table samples differ from figure samples')
        if 'pair_tmrca_rmse' in reference['result']:
            delta = abs(reference['result']['pair_tmrca_rmse']-stats['rmse_2Ne'])
            require(delta < 1e-10, f'Figure/table RMSE disagreement: {delta}')
            reference_check = dict(path=str(reference_path), sha256=sha256(reference_path),
                                   rmse_absolute_difference=delta)
    npz = output/f'{key}.npz'
    np.savez_compressed(npz, truth=x, posterior_mean=y, weight=weights,
                        histogram_percent=histogram, underflow_percent=underflow, bins_2Ne=bins)
    row = dict(dataset=dataset, method=source['method'], checkpoint=checkpoint, draws=len(draws),
               pairs=observations.num_haplotypes*(observations.num_haplotypes-1)//2,
               population_size=ne, time_units='2Ne generations', **stats,
               plot_data=dict(path=str(npz), sha256=sha256(npz)),
               truth=dict(path=str(truth_path), sha256=sha256(truth_path)),
               metadata=dict(path=str(metadata_path), sha256=sha256(metadata_path)),
               source_provenance=provenance, table_crosscheck=reference_check)
    write_json(output/f'{key}.json', row)
    print(json.dumps({k: row[k] for k in ('dataset','method','checkpoint','draws',
                                        'rmse_2Ne','pearson_r','below_range_percent')}), flush=True)
    return row


def plot(panels, datasets, methods, options, output, filename):
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
                         'axes.linewidth':.6, 'pdf.fonttype':42, 'svg.fonttype':'none'})
    low, high = options['axis_limits_2Ne']
    norm = LogNorm(*options['color_limits_percent'])
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('black')
    fig, axes = plt.subplots(len(datasets), len(methods), figsize=(9.2, 3*len(datasets)), squeeze=False)
    fig.subplots_adjust(left=.11, right=.88, bottom=.11, top=.945, wspace=.25, hspace=.40)
    for i, dataset in enumerate(datasets):
        for j, method in enumerate(methods):
            ax = axes[i,j]
            row = panels[dataset,method]
            with np.load(row['plot_data']['path']) as data:
                h, underflow, bins = data['histogram_percent'], data['underflow_percent'], data['bins_2Ne']
            ax.set_facecolor('black')
            ax.set(xscale='log', yscale='log', xlim=(low,high), ylim=(low,high), aspect='equal')
            ax.pcolormesh(bins, bins, np.ma.masked_where(h.T==0,h.T), cmap=cmap,
                          norm=norm, rasterized=True, zorder=1)
            ax.plot([low,high],[low,high],color='#eeeeee',linestyle='--',lw=.8,zorder=3)
            mask = underflow > 0
            if mask.any():
                ax.scatter(np.sqrt(bins[:-1]*bins[1:])[mask], np.full(mask.sum(),low*1.07),
                           c=underflow[mask], cmap=cmap, norm=norm, marker='v', s=20,
                           linewidths=.5, edgecolors='white', zorder=5)
                ax.text(.97,.045,f'Below range: {row["below_range_percent"]:.2f}%',
                        transform=ax.transAxes,color='white',fontsize=7.5,ha='right',
                        bbox=dict(facecolor='black',edgecolor='none',alpha=.8,pad=1))
            corr = f'{row["pearson_r"]:.2f}' if row['pearson_r'] is not None else 'undefined'
            ax.text(.055,.95,r'RMSE$_{2N_e}$' + f' = {row["rmse_2Ne"]:.3f}\n' + r'$r$' + f' = {corr}',
                    transform=ax.transAxes,va='top',fontsize=9.2,color='#eeeeee',zorder=6)
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(LogLocator(base=10,numticks=5))
                axis.set_major_formatter(LogFormatterMathtext())
                axis.set_minor_locator(LogLocator(base=10,subs=np.arange(2,10),numticks=100))
            ax.grid(which='both',color='#747474',alpha=.18,linewidth=.35,zorder=2)
            ax.tick_params(which='major',length=3,width=.6,labelsize=8)
            ax.tick_params(which='minor',length=0)
            if j != 0:
                ax.tick_params(labelleft=False)
            for spine in ax.spines.values():
                spine.set_color('#777777')
            if i == 0:
                ax.set_title(method,fontsize=12,pad=9)
            if j == 0:
                ax.text(-.30,1.06,chr(ord('a')+i),transform=ax.transAxes,fontsize=14,fontweight='bold')
                ax.text(0,1.055,dataset,transform=ax.transAxes,fontsize=10)
    cax = fig.add_axes([.915,.26,.020,.49])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax)
    cb.ax.tick_params(labelsize=8,width=.6,length=3)
    cb.ax.set_title('Weighted\nmass (%)',fontsize=9,pad=8)
    fig.supxlabel(r'Simulated pairwise TMRCA ($2N_e$ units)',y=.064,fontsize=11)
    fig.supylabel(r'Posterior mean pairwise TMRCA ($2N_e$ units)',x=.022,fontsize=11)
    pair_counts = {row['pairs'] for row in panels.values()}
    pair_text = f'{next(iter(pair_counts))} haplotype pairs · ' if len(pair_counts)==1 else ''
    fig.text(.49,.024,'Posterior means · '+pair_text+'genomic-span weights · shared log scales',
             fontsize=8,ha='center',color='#555555')
    for extension in ('png','pdf','svg'):
        fig.savefig(output/f'{filename}.{extension}',dpi=options['dpi'],facecolor='white')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'))
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    options = config['figure2']
    output = resolve(options['output_dir'])
    low, high = options['axis_limits_2Ne']
    require(0 < low < high and options['bins'] > 0, 'Invalid plot range/bins')
    require(0 < options['color_limits_percent'][0] < options['color_limits_percent'][1], 'Invalid color range')
    methods = options['methods']
    require(len(methods)==len(set(methods)) and methods, 'Methods must be distinct and nonempty')
    datasets = [name for name,settings in config['datasets'].items() if settings.get('enabled',True)]
    require(bool(datasets), 'No enabled datasets')
    jobs = [(name,config['datasets'][name],source) for name in datasets
            for source in config['datasets'][name]['sources']
            if source.get('enabled',True) and source['method'] in methods]
    for name,settings,source in jobs:
        for protected in (resolve(settings['dataset_dir']),resolve(source['directory']),resolve(config['output_dir'])):
            require(output != protected and protected not in output.parents, 'Figure output must be separate from inputs/reports')
    output.mkdir(parents=True,exist_ok=True)
    (output/'config.used.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    bins = np.geomspace(low,high,options['bins']+1)
    rows = []
    for dataset,settings,source in jobs:
        print(f'Preparing {dataset}/{source["method"]}/{Path(source["directory"]).name}',flush=True)
        rows.append(prepare_panel(dataset,settings,source,config,output,bins))
    main_panels = {}
    for dataset in datasets:
        for method in methods:
            matches = [r for r in rows if r['dataset']==dataset and r['method']==method]
            if method=='ARGFlows':
                selected = options['main_argflows_checkpoints'][dataset]
                matches = [r for r in matches if r['checkpoint']==selected]
            require(len(matches)==1,f'Expected one selected panel for {dataset}/{method}')
            main_panels[dataset,method]=matches[0]
    variants = {'figure_2':main_panels}
    if options.get('save_alternatives',True):
        for row in rows:
            key = row['dataset'],row['method']
            if row['method']=='ARGFlows' and row is not main_panels[key]:
                variants[f'figure_2_{row["dataset"]}_{row["checkpoint"]}']={**main_panels,key:row}
    for filename,panels in variants.items():
        plot(panels,datasets,methods,options,output,filename)
    write_json(output/'provenance.json',dict(
        config_sha256=sha256(args.config), options=options,
        software=dict(python=platform.python_version(),numpy=np.__version__,tskit=tskit.__version__,matplotlib=matplotlib.__version__),
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in
                      (Path(__file__),Path(__file__).with_name('evaluate.py'),
                       ROOT/'validation/scripts/evaluate_arginfer.py',ROOT/'env/snp_data.py')},
        panels=[{k:v for k,v in r.items() if k!='source_provenance'} for r in rows],
        figures={name:[dict(dataset=r['dataset'],method=r['method'],checkpoint=r['checkpoint'])
                       for r in panels.values()] for name,panels in variants.items()}))
    caption = ('Figure 2. Pairwise TMRCA reconstruction against simulated truth. '
        'Columns show ARGFlows, ARGInfer and SINGER; rows show r1, r2 and r4. '
        'Posterior means are compared with generating pairwise TMRCAs on shared logarithmic axes, '
        'in units of 2Ne generations. The dashed diagonal denotes exact agreement. '
        'Color shows probability mass per logarithmic bin, weighted by exact genomic span and '
        'equally across haplotype pairs, with a shared logarithmic color scale. '
        'Annotations report RMSE and span-weighted Pearson correlation computed on the original '
        'untransformed time values. Downward triangles indicate inferred values below the plotting '
        'range; their mass is reported in each affected panel and included in both statistics.')
    (output/'caption.txt').write_text(caption+'\n')
    print(f'Figure complete: {output/"figure_2.pdf"}',flush=True)


if __name__=='__main__':
    main()
