#!/usr/bin/env python3
"""Produce the final supplement table and figure from audited fixed-budget runs."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import numpy as np
import yaml


def summarize(config_path, include_control=False):
    config=yaml.safe_load(Path(config_path).read_text())
    output=ROOT/config['output_dir']
    specifications=[(f'main_gamma_mixture_seed{s}','Gamma mixture',s) for s in config['training']['seeds']]
    if include_control:
        specifications += [('single_gamma_gamma_seed7','Single Gamma',7)]
    runs=[]
    for name,label,seed in specifications:
        directory=output/'runs'/name
        result=json.loads((directory/'result.json').read_text())
        audit=json.loads((directory/'audit.json').read_text())
        if not audit['artifact_checks_passed'] or result['step']!=config['training']['steps']:
            raise ValueError(f'{name} has not passed its fixed-budget artifact audit')
        if result['training_seed']!=seed or not result['final']:
            raise ValueError(f'{name} has inconsistent result provenance')
        runs.append((directory,label,result,audit))
    mixture=[r for _,label,r,_ in runs if label=='Gamma mixture']
    values=np.array([r['full_history_tv']['estimate'] for r in mixture])
    prior=json.loads((output/'reference/prior_metrics.json').read_text())
    evaluation_devices={str(result['training_seed']):
        json.loads((directory/'evaluation_execution.json').read_text())['device']
        if (directory/'evaluation_execution.json').exists() else
        json.loads((directory/'provenance.json').read_text())['device']
        for directory,label,result,_ in runs if label=='Gamma mixture'}
    simultaneous=[]
    for result in mixture:
        radius=math.sqrt(math.log(2*len(mixture)/.05)/8*(1/result['reference_samples']+1/result['policy_samples']))
        simultaneous.append(min(1.,result['full_history_tv']['estimate']+radius))
    report=dict(dataset=config['dataset'],training_updates=config['training']['steps'],
        mixture_training_seeds=config['training']['seeds'],full_history_tv_by_seed=values.tolist(),
        mean_tv=float(values.mean()),training_seed_standard_deviation=float(values.std(ddof=1)),
        simultaneous_95_tv_upper_bounds=simultaneous,
        all_simultaneous_bounds_below_target=all(x<config['evaluation']['tv_target'] for x in simultaneous),
        all_mixture_runs_meet_target=all(r['meets_tv_target'] for r in mixture),
        tv_target=config['evaluation']['tv_target'],prior_tv=prior['full_history_tv'],
        evaluation_devices_by_seed=evaluation_devices,
        single_gamma_control='completed_one_seed' if include_control else 'optional_not_included',
        interpretation='TV estimates measure complete timed ARG histories. Training-seed SD is separate from fixed-policy Monte Carlo intervals.',
        run_directories=[str(d.relative_to(output)) for d,_,_,_ in runs])
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    fields=['head','seed','updates','tv','tv_mc_lower95','tv_mc_upper95','squared_hellinger',
            'forward_kl_nats','ess_fraction','learned_log_evidence','reference_log_evidence']
    rows=[]
    for _,label,r,_ in runs:
        tv=r['full_history_tv']
        rows.append([label,r['training_seed'],r['step'],tv['estimate'],*tv['ci95'],
            r['squared_hellinger']['estimate'],r['forward_kl_nats']['estimate'],
            r['importance_ess_fraction'],r['learned_log_evidence'],r['reference_log_evidence']])
    with (output/'supplement_table.csv').open('w') as handle:
        writer=csv.writer(handle);writer.writerow(fields);writer.writerows(rows)
    latex=[r'\begin{tabular}{lrrrr}',r'\hline',
           r'Time head & Seed & Full-history TV (95\% MC CI) & $H^2$ & ESS/$N$ \\',r'\hline']
    for row in rows:
        latex.append(f'{row[0]} & {row[1]} & {row[3]:.5f} [{row[4]:.5f}, {row[5]:.5f}] & {row[6]:.6f} & {row[8]:.4f} '+r'\\')
    latex += [r'\hline',r'\end{tabular}']
    (output/'supplement_table.tex').write_text('\n'.join(latex)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
    colors=plt.get_cmap('tab10').colors
    for i,(directory,label,result,_) in enumerate(runs):
        curve=sorted([json.loads(p.read_text()) for p in directory.glob('metrics_*.json')],key=lambda r:r['step'])
        x=np.array([r['step'] for r in curve]);y=np.array([r['full_history_tv']['estimate'] for r in curve])
        se=np.array([1.96*r['full_history_tv']['standard_error'] for r in curve])
        name=f'{label}, seed {result["training_seed"]}'
        axes[0].plot(x,y,marker='o',markersize=3,label=name,color=colors[i])
        axes[0].fill_between(x,np.maximum(0,y-se),y+se,color=colors[i],alpha=.12)
        tv=result['full_history_tv']
        axes[1].errorbar(i,tv['estimate'],yerr=1.96*tv['standard_error'],fmt='o',capsize=4,color=colors[i])
    axes[0].axhline(prior['full_history_tv']['estimate'],color='grey',linestyle=':',label='Physical prior')
    for ax in axes:
        ax.axhline(config['evaluation']['tv_target'],color='black',linestyle='--',linewidth=.8)
        ax.set_ylabel('Full-history total variation')
        ax.set_ylim(bottom=0)
    axes[0].set_xlabel('Training updates');axes[0].legend(fontsize=7)
    axes[1].set_xticks(range(len(runs)),[f'{label}\nseed {r["training_seed"]}' for _,label,r,_ in runs],fontsize=8)
    axes[1].set_title('Fixed final checkpoint; 95% MC intervals')
    for extension in ('png','pdf'):
        fig.savefig(output/f'supplement_figure.{extension}',dpi=200)
    plt.close(fig)
    text=['# Posterior POC: final audited results','',
          f"All {len(mixture)} mixture runs used {config['training']['steps']:,} SubTB updates on the same fixed two-haplotype, two-base dataset.",
          f"Mean full-history TV across training seeds: **{values.mean():.5f}**; training-seed standard deviation: **{values.std(ddof=1):.5f}**.",
          f"All mixture runs meet the configured TV target: **{report['all_mixture_runs_meet_target']}**.",'',
          '| Time head | Seed | Full-history TV (95% MC interval) | Hellinger² | ESS/N |',
          '|---|---:|---|---:|---:|']
    for row in rows:
        text.append(f'| {row[0]} | {row[1]} | {row[3]:.5f} [{row[4]:.5f}, {row[5]:.5f}] | {row[6]:.6f} | {row[8]:.4f} |')
    text += ['', 'Conservative simultaneous 95% upper bounds on true TV (Hoeffding plus a union bound across the three fixed policies): **'+', '.join(f'{x:.5f}' for x in simultaneous)+'**.',
             'These bounds account for the two independent sampling strata and remain valid when the same reference bank is reused across training seeds. They assume exact reference draws, correct density evaluation, and policies independent of the evaluation bank.',
             '', '[Supplement figure (PDF)](supplement_figure.pdf) · [CSV table](supplement_table.csv) · [LaTeX table](supplement_table.tex)',
             '', 'TV is evaluated from full chronological history densities with an independently normalized exact reference. The samples are unweighted policy draws. Intervals quantify Monte Carlo error for a fixed trained policy; they are not confidence intervals across training seeds.',
             '', 'Final evaluation devices by seed: '+', '.join(f'{seed}: {device}' for seed,device in evaluation_devices.items())+'. All three training runs used the A100; the last saved checkpoint was evaluated on CPU after GPU work stopped.',
             '', ('The optional single-Gamma comparison uses one seed and is descriptive.' if include_control else
                  'The single-Gamma comparison was left optional at the user\'s request because of the allocation time limit. No numerical comparison with that head is claimed. The independent analytic first-wait derivation in the reproduction protocol explains why a single Gamma is not exact for this example.'),
             '', 'This example includes recombination-history and continuous-time uncertainty; two tips have no local-topology uncertainty, and the two-base region has one possible breakpoint. Small measured TV supports recovery to a numerical tolerance, not a proof of exact equality.',
             '', 'The original interrupted launches and the completed 200-update pilot are retained. See [all completed runs](RESULTS.md), the per-run `audit.json` files, and the [reproduction protocol](README.md).','']
    (output/'FINAL_REPORT.md').write_text('\n'.join(text))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'validation/config/poc.yaml')
    p.add_argument('--include-control',action='store_true',help='Also require and report the optional audited single-Gamma seed-7 run')
    args=p.parse_args()
    print(json.dumps(summarize(args.config,args.include_control),indent=2))
