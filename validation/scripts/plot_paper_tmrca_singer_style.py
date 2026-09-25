"""Restyle saved exact TMRCA reconstructions after SINGER Figure 2a/b."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, LogFormatterMathtext

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'validation/paper_datasets/report/tmrca_reconstruction'
DATASETS = ['r1', 'r2', 'r4']
METHODS = ['ARGFlow', 'ARGInfer', 'SINGER']
LOW, HIGH = .01, 10.
BINS = np.geomspace(LOW, HIGH, 81)


def read_panels():
    panels = {}
    for dataset in DATASETS:
        for method in METHODS:
            path = OUT / f'{dataset}_{method}.npz'
            if not path.exists():
                panels[dataset, method] = None
                continue
            d = np.load(path)
            x, y, w = d['truth'], d['posterior_mean'], d['weight']
            assert np.isfinite(x).all() and np.isfinite(y).all() and np.isclose(w.sum(), 1)
            assert x.min() >= LOW and max(x.max(), y.max()) <= HIGH
            mx, my = np.sum(w*x), np.sum(w*y)
            correlation = np.sum(w*(x-mx)*(y-my)) / np.sqrt(np.sum(w*(x-mx)**2)*np.sum(w*(y-my)**2))
            density, _, _ = np.histogram2d(x, y, bins=[BINS, BINS], weights=w*100)
            underflow = np.histogram(x[y < LOW], bins=BINS, weights=w[y < LOW]*100)[0]
            assert np.isclose(density.sum()+underflow.sum(), 100)
            meta = json.loads(path.with_suffix('.json').read_text())
            panels[dataset, method] = dict(density=density, underflow=underflow,
                underflow_percent=float(underflow.sum()), correlation=float(correlation),
                rmse_2Ne=meta['rmse_2Ne'], repeat_average=len(meta['repeat_rmse_2Ne'])>1,
                source=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return panels


def plot(panels, units):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.linewidth': .6, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    scale = 1 if units == '2Ne' else 20000
    norm = LogNorm(vmin=.001, vmax=100)
    cmap = plt.get_cmap('magma').copy(); cmap.set_bad('black')
    fig, axes = plt.subplots(3, 3, figsize=(9.2, 9.0))
    fig.subplots_adjust(left=.11, right=.88, bottom=.11, top=.945, wspace=.25, hspace=.40)
    for i, dataset in enumerate(DATASETS):
        for j, method in enumerate(METHODS):
            ax = axes[i, j]
            ax.set_facecolor('black')
            row = panels[dataset, method]
            ax.set(xscale='log', yscale='log', xlim=(LOW*scale, HIGH*scale), ylim=(LOW*scale, HIGH*scale), aspect='equal')
            if row is None:
                ax.text(.5, .52, 'SYN', transform=ax.transAxes, color='white', fontsize=34, alpha=.5, rotation=25, ha='center', va='center')
                ax.text(.5, .25, 'Output unavailable', transform=ax.transAxes, color='white', ha='center', fontsize=9)
            else:
                h = row['density']
                mesh = ax.pcolormesh(BINS*scale, BINS*scale, np.ma.masked_where(h.T == 0, h.T), cmap=cmap, norm=norm, rasterized=True, zorder=1)
                ax.plot([LOW*scale, HIGH*scale], [LOW*scale, HIGH*scale], color='#e6e6e6', lw=.7, zorder=3)
                # Underflow observations are marked at the boundary, not silently discarded
                # or treated as observations with TMRCA equal to the axis minimum.
                mask = row['underflow'] > 0
                if mask.any():
                    centres = np.sqrt(BINS[:-1]*BINS[1:])[mask]*scale
                    ax.scatter(centres, np.full(len(centres), LOW*scale*1.065), c=row['underflow'][mask], norm=norm, cmap=cmap, marker='v', s=18, linewidths=.45, edgecolors='white', zorder=5)
                    ax.text(.97, .045, f'Below range: {row["underflow_percent"]:.2f}%', transform=ax.transAxes, color='white', fontsize=7.2, ha='right', bbox=dict(facecolor='black', edgecolor='none', alpha=.75, pad=1))
                rmse_label = r'mean RMSE$_{2N_e}$' if row['repeat_average'] else r'RMSE$_{2N_e}$'
                ax.text(.055, .95, f'{rmse_label} = {row["rmse_2Ne"]:.3f}\n' + r'$r$' + f' = {row["correlation"]:.2f}', transform=ax.transAxes, ha='left', va='top', fontsize=9.2, color='#eeeeee', zorder=6)
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
            ax.xaxis.set_major_formatter(LogFormatterMathtext())
            ax.yaxis.set_major_formatter(LogFormatterMathtext())
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2,10), numticks=100))
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2,10), numticks=100))
            ax.grid(which='both', color='#747474', alpha=.18, linewidth=.35, zorder=2)
            ax.tick_params(which='major', direction='out', length=3, width=.6, labelsize=8)
            ax.tick_params(which='minor', length=0)
            if j != 0: ax.tick_params(labelleft=False)
            for spine in ax.spines.values(): spine.set_color('#777777')
            if i == 0: ax.set_title(method, fontsize=12, pad=9)
            if j == 0:
                ax.text(-.30, 1.06, chr(ord('a')+i), transform=ax.transAxes, fontsize=14, fontweight='bold')
                ax.text(0, 1.055, dataset, transform=ax.transAxes, fontsize=10)
    cax = fig.add_axes([.915, .26, .020, .49])
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(mappable, cax=cax, ticks=[.001,.01,.1,1,10,100])
    cb.ax.tick_params(labelsize=8, width=.6, length=3)
    cb.ax.set_title('Weighted\nmass (%)', fontsize=9, pad=8)
    unit_label = r'$2N_e$ units' if units == '2Ne' else 'generations'
    fig.supxlabel(f'Simulated pairwise TMRCA ({unit_label})', y=.064, fontsize=11)
    fig.supylabel(f'Inferred pairwise TMRCA ({unit_label})', x=.022, fontsize=11)
    fig.text(.49, .024, 'Posterior means · 45 haplotype pairs · genomic-span weights · shared log scales', fontsize=8, ha='center', color='#555555')
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'pairwise_tmrca_singer_style_{units}.{extension}', dpi=300, facecolor='white')
    plt.close(fig)


def main():
    panels = read_panels()
    for units in ['2Ne', 'generations']:
        plot(panels, units)
    stats = [{**{k:v for k,v in panel.items() if k not in ['density','underflow']}, 'dataset':ds, 'method':method}
             if panel is not None else dict(dataset=ds, method=method, missing=True)
             for (ds, method), panel in panels.items()]
    (OUT/'singer_style_statistics.json').write_text(json.dumps(stats, indent=2)+'\n')
    text = '''# SINGER-style pairwise TMRCA reconstruction

Visual reference: Deng, Nielsen and Song (2025), Figure 2a–b:
https://www.nature.com/articles/s41588-025-02317-9/figures/2

ARGinfer uses a different display for TMRCA: genomic-position traces of the local sample MRCA, with a black dashed truth, red posterior mean and shaded 50% credible interval (Mahmoudi et al. 2022, Figure 5):
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009960#pcbi-1009960-g005

## Caption

Pairwise TMRCA reconstruction against simulated truth. Columns show ARGFlow, ARGInfer and SINGER; rows show r1, r2 and r4. Posterior means are plotted against generating pairwise TMRCAs on logarithmic axes. The white diagonal indicates exact agreement. Color indicates probability mass per logarithmic bin, weighted by genomic span and equally across the 45 haplotype pairs. The axes and logarithmic color scale are shared across panels. Annotations give RMSE in 2Ne units and genomic-span-weighted Pearson correlation on the original, untransformed time values. Ne = 10,000. Downward triangles denote observations below the lower plotting bound; their weighted mass is reported in the affected panel. Statistics include all observations, including underflow observations.

## Adaptation and interpretation

- This adopts SINGER's dark log–log heatmaps, magma palette, light agreement line, small method headings and compact statistic labels. It uses this project's saved data; it does not reuse the paper's plotted results.
- Unlike the paper's MSE annotation, RMSE is retained for consistency with the user's comparison table. The paper's raw site-count color scale is replaced by normalized genomic-span-weighted probability mass on a shared logarithmic scale. Bins have identical widths in log time; no density smoothing is applied.
- Display range is 0.01–10 in 2Ne units (200–200,000 generations). All true values and all upper tails lie within this range. SINGER has some inferred values below the lower bound. These appear as boundary triangles, not as literal estimates at the boundary, and are not excluded from error/correlation calculations. The 2Ne and generations versions are coordinate rescalings of the same data; RMSE annotations remain in 2Ne units in both versions.
- r2/r4 ARGFlow combine three repeat-specific reconstructions with equal repeat weight; the RMSE label is the mean of the three repeat RMSEs, while Pearson r summarizes the combined weighted observations. These are not pooled-sample posterior means.
- There is one simulated 10-haplotype dataset per regime here. The SINGER paper's benchmark uses many simulated datasets and more haplotypes, so its density clouds look smoother. No extra observations were synthesized to imitate that appearance.
- All nine panels have real outputs. A missing panel would receive a SYN placeholder, but none was needed.
- See README.md and provenance.json for sample selection and original data provenance. singer_style_statistics.json records the plotted statistics, underflow mass and SHA-256 hashes of all input NPZ files.

Reproduce with:

```bash
/private/home/pkatte/anaconda3/envs/phylogfn_orig/bin/python validation/scripts/plot_paper_tmrca_singer_style.py
```
'''
    (OUT/'SINGER_STYLE.md').write_text(text)
    print(json.dumps(stats, indent=2))
    print('Saved SINGER-style PNG/PDF/SVG in', OUT)


if __name__ == '__main__':
    main()
