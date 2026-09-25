"""Figure 3: exact posterior-summary agreement from the final saved draws."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import itertools
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wasserstein_distance
import yaml

from validation.scripts.paper_datasets.evaluate import load_draws, resolve, require, sha256, write_json
from validation.scripts.plot_paper_posterior_agreement import compare_clades
from env.snp_data import load_snp_dataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

METHODS = ('ARGFlows', 'ARGInfer', 'SINGER')
COLORS = dict(ARGFlows='#0072B2', ARGInfer='#D55E00', SINGER='#009E73')


def summarize_features(draws, ne, length):
    """Equal-draw/equal-pair/span mixture and exact clade count-change tracks."""
    require(bool(draws) and ne > 0 and length > 0, 'Invalid posterior dimensions')
    values, weights = [], []
    events = defaultdict(lambda: defaultdict(int))
    pairs = draws[0].times.shape[1]-1
    for draw in draws:
        times = draw.times[:, :-1]/(2*ne)
        values.append(times.ravel())
        weights.append(np.broadcast_to(np.diff(draw.boundaries)[:, None]/(length*pairs*len(draws)), times.shape).ravel())
        for left, right, clades in zip(draw.boundaries[:-1], draw.boundaries[1:], draw.topologies):
            for clade in clades:
                events[clade][float(left)] += 1
                events[clade][float(right)] -= 1
    values, inverse = np.unique(np.concatenate(values), return_inverse=True)
    mass = np.bincount(inverse, weights=np.concatenate(weights))
    require(np.isclose(mass.sum(), 1), 'TMRCA mixture lost mass')
    masks, positions, changes = [], [], []
    for mask, track in sorted(events.items()):
        for position, change in sorted(track.items()):
            if change:
                masks.append(mask); positions.append(position); changes.append(change)
    return dict(tmrca=values, weight=mass, mask=np.array(masks, dtype=np.int64),
                position=np.array(positions), change=np.array(changes, dtype=np.int32), draws=len(draws))


def empirical_wasserstein_sorted(a, b):
    """Exact integral of |Q_a(u)-Q_b(u)| for unequal uniform empirical samples.

    Leading axes are arbitrary batches. The last axis holds sorted draws.
    Integer quantile breakpoints avoid interpolation or resampling.
    """
    n, m = a.shape[-1], b.shape[-1]
    require(n > 0 and m > 0 and a.shape[:-1] == b.shape[:-1], 'Invalid empirical arrays')
    if n == m:
        return np.mean(np.abs(a-b), axis=-1)
    denominator = math.lcm(n, m)
    step_a, step_b = denominator//n, denominator//m
    edges = np.union1d(np.arange(n+1, dtype=np.int64)*step_a,
                       np.arange(m+1, dtype=np.int64)*step_b)
    weights = np.diff(edges)/denominator
    return np.sum(np.abs(a[..., edges[:-1]//step_a]-b[..., edges[:-1]//step_b])*weights, axis=-1)


def local_wasserstein(ensembles, ne, length, chunk_size=16, progress=False):
    """Mean pair/position W1 on the exact joint breakpoint partition."""
    require(len(ensembles)==3 and all(ensembles), 'Expected three nonempty ensembles')
    require(ne > 0 and length > 0 and chunk_size > 0, 'Invalid local W1 settings')
    boundaries = np.unique(np.concatenate([d.boundaries for group in ensembles for d in group]))
    require(boundaries[0]==0 and boundaries[-1]==length, 'Inconsistent sequence span')
    pairs = ensembles[0][0].times.shape[1]-1
    sums = np.zeros((2, pairs))
    for start in range(0, len(boundaries)-1, chunk_size):
        stop = min(start+chunk_size, len(boundaries)-1)
        left = boundaries[start:stop]
        span = np.diff(boundaries[start:stop+1])/length
        sorted_times = []
        for group in ensembles:
            times = np.stack([d.times[d.at(left), :-1] for d in group], axis=-1)/(2*ne)
            times.sort(axis=-1)
            sorted_times.append(times)
        for j in range(2):
            distances = empirical_wasserstein_sorted(sorted_times[0], sorted_times[j+1])
            sums[j] += np.sum(span[:, None]*distances, axis=0)
        if progress and (start==0 or stop==len(boundaries)-1 or start//chunk_size % 200==0):
            print(f'  Local W1: {stop}/{len(boundaries)-1} exact intervals', flush=True)
    return sums.mean(axis=1), sums, len(boundaries)-1


def clade_agreement(summaries, length, n, normalization):
    require(normalization in ('fixed_universe', 'shared_union'), 'Unknown clade normalization')
    universe_size = 2**n-n-2
    require(universe_size > 0, 'Clade comparison needs at least three haplotypes')
    comparisons = compare_clades(summaries, length=length, n=n)
    result = []
    for points, stats in comparisons:
        union = stats['local_union_mean_size']
        require(union <= universe_size+1e-9, 'Clade union exceeds full universe')
        # Include (0,0) events for clades absent from every method when displaying
        # the fixed universe, so the plotted weights and RMSE share a denominator.
        if normalization=='fixed_universe':
            points = {k: np.array(v, copy=True) for k,v in points.items()}
            points['weight'] *= union/universe_size
            points['reference_probability'] = np.r_[points['reference_probability'], 0.]
            points['argflow_probability'] = np.r_[points['argflow_probability'], 0.]
            points['weight'] = np.r_[points['weight'], max(0., 1-union/universe_size)]
        error = points['argflow_probability']-points['reference_probability']
        selected = float(np.sqrt(np.dot(points['weight'], error**2)))
        expected = stats['full_universe_rmse'] if normalization=='fixed_universe' else stats['rmse']
        require(np.isclose(selected, expected, atol=1e-12), 'Clade plot/metric weighting mismatch')
        require(np.isclose(points['weight'].sum(), 1), 'Clade weights lost mass')
        result.append((points, dict(clade_rmse=selected,
            clade_rmse_fixed_universe=stats['full_universe_rmse'],
            clade_rmse_shared_union=stats['rmse'], clade_universe_size=universe_size,
            shared_union_mean_size=union, clade_normalization=normalization)))
    return result


def plot(output, datasets, summaries, comparisons, options):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                         'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig = plt.figure(figsize=(11.6, 7.1))
    grid = fig.add_gridspec(2, len(datasets), left=.085, right=.98, top=.84, bottom=.115,
                           hspace=.73, wspace=.26, height_ratios=(1,.84))
    maximum = max(float(s['tmrca'].max()) for s in summaries.values())
    width, bandwidth = options['density_bin_width_2Ne'], options['density_bandwidth_2Ne']
    bins = np.arange(0, maximum+2*width, width)
    centers = (bins[:-1]+bins[1:])/2
    clade_bins = np.linspace(0,1,options['clade_bins']+1)
    norm = LogNorm(*options['clade_color_limits_percent'])
    axes, ymax = [], 0.
    annotation = options['wasserstein_annotation']
    metric_key = 'local_mean_wasserstein_2Ne' if annotation=='local_mean' else 'pooled_wasserstein_2Ne'
    for col,dataset in enumerate(datasets):
        ax = fig.add_subplot(grid[0,col]); axes.append(ax)
        for method in METHODS:
            data = summaries[dataset,method]
            hist,_ = np.histogram(data['tmrca'], bins=bins, weights=data['weight'])
            require(np.isclose(hist.sum(),1), 'Density plot truncated TMRCA mass')
            density = gaussian_filter1d(hist/width, bandwidth/width, mode='reflect')
            ymax = max(ymax,float(density.max()))
            ax.plot(centers,density,color=COLORS[method],lw=1.8,label=method)
        ax.set_title(dataset,fontsize=14,fontweight='semibold',pad=10)
        ax.set_xscale('symlog',linthresh=2,linscale=2,base=2)
        ax.set_xlim(0,max(2,np.ceil(maximum)))
        ax.set_xlabel(r'Pairwise TMRCA ($2N_e$ units)')
        ticks=[x for x in [0,1,2,4,8,16,32,64,128,256,512] if x <= max(2,np.ceil(maximum))]
        ax.set_xticks(ticks,labels=[str(x) for x in ticks])
        ax.axvline(2,color='#d6dce2',lw=.6,zorder=0)
        label = 'Mean local $W_1$' if annotation=='local_mean' else 'Pooled $W_1$'
        text = label+'\n'+'\n'.join(f'vs {ref}: {comparisons[dataset,ref][metric_key]:.3f}' for ref in METHODS[1:])
        ax.text(.98,.97,text,transform=ax.transAxes,ha='right',va='top',fontsize=8,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=2))
        if col==0:
            ax.set_ylabel('Pooled posterior density')
        subgrid = grid[1,col].subgridspec(1,2,wspace=.21)
        for j,reference in enumerate(METHODS[1:]):
            sub = fig.add_subplot(subgrid[0,j])
            with np.load(output/f'{dataset}_clades_{reference}.npz') as points:
                mass,_,_ = np.histogram2d(points['reference_probability'],points['argflow_probability'],
                                         bins=(clade_bins,clade_bins),weights=100*points['weight'])
            require(np.isclose(mass.sum(),100), 'Clade plot lost mass')
            np.savez_compressed(output/f'{dataset}_clade_density_{reference}.npz',
                                edges=clade_bins,mass_percent=mass)
            mesh = sub.pcolormesh(clade_bins,clade_bins,np.ma.masked_equal(mass.T,0),
                                 cmap='Blues',norm=norm,rasterized=True,zorder=2)
            sub.plot([0,1],[0,1],color='#66717e',ls='--',lw=.85,zorder=3)
            sub.set(xlim=(-.025,1.025),ylim=(-.025,1.025),aspect='equal',xticks=[0,.5,1],yticks=[0,.5,1])
            sub.set_xticklabels(['0','.5','1']); sub.set_yticklabels(['0','.5','1'])
            sub.set_title(reference,color=COLORS[reference],fontsize=9,pad=6)
            sub.text(.04,.97,f'RMSE\n{comparisons[dataset,reference]["clade_rmse"]:.4f}',
                     transform=sub.transAxes,va='top',fontsize=7,
                     bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
            if j:
                sub.set_yticklabels([])
            elif col==0:
                sub.set_ylabel('ARGFlows clade probability',labelpad=5)
    for ax in axes:
        ax.set_ylim(0,ymax*1.23)
    fig.text(.085,.913,'(a) Pairwise TMRCA posterior distributions',fontsize=10)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper right',bbox_to_anchor=(.985,.952),
               ncol=3,frameon=False,fontsize=10)
    fig.text(.085,.435,'(b) Local-clade probability agreement',fontsize=10)
    color_ax = fig.add_axes([.685,.426,.285,.012])
    cb = fig.colorbar(mesh,cax=color_ax,orientation='horizontal',ticks=[1e-4,1e-2,1,100])
    cb.set_ticklabels(['0.0001','0.01','1','100'])
    cb.ax.tick_params(labelsize=7,length=2)
    cb.ax.xaxis.set_label_position('top')
    cb.set_label('Genomic-span-weighted mass per bin (%)',fontsize=7,labelpad=4)
    fig.text(.54,.067,'Reference clade probability',ha='center',fontsize=10)
    fig.text(.54,.023,r'TMRCA axis: linear to 2, logarithmic above 2 · times in $2N_e$ generations',
             ha='center',fontsize=8,color='#526171')
    for suffix in ('png','pdf','svg'):
        fig.savefig(output/f'figure_3.{suffix}',dpi=options['dpi'],facecolor='white')
    plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'))
    args=parser.parse_args(argv)
    config=yaml.safe_load(args.config.read_text()); options=config['figure3']
    require(tuple(options['methods'])==METHODS,'Expected ARGFlows, ARGInfer, SINGER method order')
    require(options['wasserstein_annotation'] in ('local_mean','pooled'),'Unknown Wasserstein annotation')
    require(options['clade_normalization'] in ('fixed_universe','shared_union'),'Unknown clade normalization')
    for key in ('chunk_size','density_bin_width_2Ne','density_bandwidth_2Ne','clade_bins','dpi'):
        require(options[key]>0,f'{key} must be positive')
    output=resolve(options['output_dir'])
    datasets=[d for d,s in config['datasets'].items() if s.get('enabled',True)]
    require(bool(datasets),'No enabled datasets')
    selected={}
    for dataset in datasets:
        settings=config['datasets'][dataset]
        for method in METHODS:
            sources=[s for s in settings['sources'] if s.get('enabled',True) and s['method']==method]
            if method=='ARGFlows':
                sources=[s for s in sources if Path(s['directory']).name==options['main_argflows_checkpoints'][dataset]]
            require(len(sources)==1,f'Expected one selected source: {dataset}/{method}')
            selected[dataset,method]=sources[0]
            for protected in (resolve(sources[0]['directory']),resolve(settings['dataset_dir']),resolve(config['output_dir'])):
                require(output!=protected and protected not in output.parents,'Figure output must be separate from inputs')
    output.mkdir(parents=True,exist_ok=True)
    (output/'config.used.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    summaries,comparisons,source_records={},{},[]
    for dataset in datasets:
        directory=resolve(config['datasets'][dataset]['dataset_dir'])
        observations=load_snp_dataset(directory)
        metadata=json.loads((directory/'metadata.json').read_text())
        require(metadata['dataset_name']==dataset,'Dataset mismatch')
        ne=float(metadata['parameters']['population_size'])
        length=observations.sequence_length
        groups=[]
        for method in METHODS:
            source=selected[dataset,method]
            print(f'Preparing {dataset}/{method}/{Path(source["directory"]).name}',flush=True)
            draws,provenance=load_draws(dataset,directory,observations,source,config)
            groups.append(draws)
            summary=summarize_features(draws,ne,length)
            summaries[dataset,method]=summary
            np.savez_compressed(output/f'{dataset}_{method}.npz',**summary)
            record=dict(dataset=dataset,method=method,draws=len(draws),population_size=ne,
                        metadata_sha256=sha256(directory/'metadata.json'),source_provenance=provenance)
            write_json(output/f'{dataset}_{method}.json',record)
            source_records.append(record)
        print(f'Comparing {dataset} posterior distributions',flush=True)
        local,per_pair,intervals=local_wasserstein(groups,ne,length,options['chunk_size'],progress=True)
        with (output/f'{dataset}_wasserstein_by_pair.csv').open('w',newline='') as handle:
            writer=csv.writer(handle)
            writer.writerow(['haplotype_a','haplotype_b',*['local_mean_W1_vs_'+m for m in METHODS[1:]]])
            for pair,values in zip(itertools.combinations(observations.haplotype_ids,2),per_pair.T):
                writer.writerow([*pair,*values])
        clades=clade_agreement([summaries[dataset,m] for m in METHODS],length,
                               observations.num_haplotypes,options['clade_normalization'])
        for j,reference in enumerate(METHODS[1:]):
            points,stats=clades[j]
            a,b=summaries[dataset,'ARGFlows'],summaries[dataset,reference]
            pooled=float(wasserstein_distance(a['tmrca'],b['tmrca'],a['weight'],b['weight']))
            require(pooled <= local[j]+1e-9,'Pooled W1 cannot exceed mean local W1')
            row=dict(dataset=dataset,reference=reference,
                     argflows_checkpoint=options['main_argflows_checkpoints'][dataset],
                     argflows_draws=len(groups[0]),reference_draws=len(groups[j+1]),
                     local_mean_wasserstein_2Ne=float(local[j]),pooled_wasserstein_2Ne=pooled,
                     aligned_intervals=intervals,**stats)
            comparisons[dataset,reference]=row
            np.savez_compressed(output/f'{dataset}_clades_{reference}.npz',**points)
            print(json.dumps(row),flush=True)
        del groups,draws
    plot(output,datasets,summaries,comparisons,options)
    rows=list(comparisons.values())
    with (output/'metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    write_json(output/'metrics.json',rows)
    write_json(output/'provenance.json',dict(options=options,config_sha256=sha256(args.config),
        software=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (Path(__file__),
            Path(__file__).with_name('evaluate.py'),ROOT/'validation/scripts/plot_paper_posterior_agreement.py',
            ROOT/'validation/scripts/evaluate_arginfer.py',ROOT/'env/snp_data.py')},
        methods=source_records,metrics=rows))
    annotation=('the mean one-dimensional Wasserstein distance computed separately at each '
                'haplotype pair and genomic position, then averaged equally over pairs and '
                'weighted by exact genomic spans' if options['wasserstein_annotation']=='local_mean'
                else 'the Wasserstein distance between the pooled empirical distributions')
    normalization=('all 1,012 possible nontrivial rooted clades for these 10-haplotype datasets, '
                   'including clades absent from every method at a position'
                   if options['clade_normalization']=='fixed_universe'
                   else 'the shared local union of clades observed in any of the three methods')
    caption=('Figure 3. Posterior-summary agreement across inference methods. '
        '(a) Pooled marginal pairwise TMRCA distributions for ARGFlows, ARGInfer and SINGER '
        'across r1, r2 and r4. Draws and haplotype pairs have equal weight and genomic positions '
        'are weighted by exact span. Curves use a common Gaussian smoothing bandwidth; '
        'annotations report '+annotation+'. Distances use unsmoothed empirical samples in '
        '2Ne generations. The axis is linear up to 2 and logarithmic above 2; density is per '
        'unit linear TMRCA, so visual area on the transformed axis is not probability mass. '
        '(b) Local-clade probability agreement, with the diagonal indicating equality. '
        'RMSE and probability-bin mass use a common denominator over '+normalization+'. '
        'Clade RMSE compares posterior probabilities between methods and is distinct from '
        'the truth-based clade Brier score in Table 2. No ground-truth TMRCA or clade membership '
        'enters either agreement metric. These summaries do not establish equality of the full ARG posterior.')
    (output/'caption.txt').write_text(caption+'\n')
    print(f'Figure complete: {output/"figure_3.pdf"}',flush=True)


if __name__=='__main__':
    main()
