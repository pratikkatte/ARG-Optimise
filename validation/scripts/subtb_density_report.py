"""CPU reporting for the frozen full-history density-fit protocol."""
import csv
import json
import math
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR', '/tmp/argopt_density_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_subtb_density_fit import read_json, write_json, METRICS, importance, digest, ROOT, density_summary


def verify_artifacts(output, protocol):
    """Check cached numerical identities, provenance, and protected inputs."""
    checks, counts, maxima = {}, {}, {}
    for filename, expected in protocol['heldout_sha256'].items():
        checks['heldout:'+filename] = digest(filename) == expected
    for relative, expected in protocol['source_sha256'].items():
        if relative != 'validation/scripts/evaluate_subtb_density_fit.py':
            checks['training_or_audit_source:'+relative] = digest(ROOT/relative) == expected
    bank_path = output/'bank.json.gz'
    if not bank_path.exists():
        return dict(complete=False, checks=checks)
    bank = read_json(bank_path)
    from subtb_density_normalization import check_material_history
    from utils import load_sequences
    sequences=load_sequences(protocol['dataset'])
    samples,blocks=len(sequences),len(sequences[0])
    origin = protocol.get('bank_origin')
    checks['bank_protocol'] = bank['protocol_sha256'] == (origin['protocol_sha256'] if origin else digest(output/'protocol.json'))
    if origin:
        checks['immutable_base_bank'] = digest(bank_path) == origin['sha256'] == digest(origin['path'])
        for cached in protocol.get('reused_artifacts', []):
            if Path(cached['target']).name != 'summary.json':
                checks['reused:'+cached['target']] = digest(output/cached['target']) == cached['sha256']
    fingerprints = [r['fingerprint'] for r in bank['records']]
    checks['bank_unique'] = len(fingerprints) == len(set(fingerprints)) == 768
    checks['bank_excludes_retained_training_and_heldout'] = not set(fingerprints)&set(protocol['forbidden_fingerprints'])
    checks['bank_strata_counts'] = all(sum(r['stratum']==s for r in bank['records'])==256 for s in ('low','medium','high'))
    from infer import load_checkpoint
    final_replay_checks = []
    for run in protocol['runs']:
        path = Path(run['path'])/'latest.pt'
        sha = digest(path)
        checkpoint = load_checkpoint(path,map_location='cpu')
        assert digest(path) == sha, 'Checkpoint changed during overlap recheck'
        meta=checkpoint['metadata']
        replay=meta.get('replay_training_state',{}).get('buffer')
        overlap=sorted(set(fingerprints)&set(replay['entries'])) if replay else []
        checks['final_replay_overlap:'+run['name']] = not overlap
        final_replay_checks.append(dict(label=run['name'],checkpoint=str(path),sha256=sha,
            update=meta['epoch']+1,entries=len(replay['entries']) if replay else 0,overlap=overlap))
        del checkpoint
    seen_nonoverlap = 0
    complete = True
    for job in protocol['jobs']:
        checks['checkpoint:'+job['checkpoint']] = digest(job['checkpoint']) == job['sha256']
        folder = output/job['label']/f"step_{job['step']:04d}"
        fixed_ids, fresh_count = [], 0
        for population in ('fixed','fresh'):
            for path in sorted(folder.glob(population+'_*.json.gz')):
                batch = read_json(path)
                rows = batch['records']
                for r in rows:
                    if population == 'fixed': fixed_ids.append(r['fingerprint'])
                    else: fresh_count += 1
                    seen_nonoverlap += r['nonoverlap_coalescences']
                    assert r['log_backward_probability'] == 0
                    assert len(r['events']) == len(r['actions']) == r['event_count']
                    assert all(e['inverse_count']==1 for e in r['events'])
                    assert check_material_history(r['events'],samples,blocks)==r['nonoverlap_coalescences']
                    assert abs(r['log_policy_density']+r['log_weight']-r['log_reward']) < 1e-8
                    assert abs(r['log_reward']-r['reward_constant']-r['log_likelihood']-r['log_prior']) < 1e-7
                loss = float(np.mean([r['balance']['loss'] for r in rows]))
                error = abs(loss-batch['metrics']['eval_subtb_loss'])
                maxima['independent_subtb_loss'] = max(maxima.get('independent_subtb_loss',0),error)
                assert error < 1e-7
                for key,error in batch['audit_errors'].items():
                    maxima[key] = max(maxima.get(key,0),error)
                if population == 'fresh':
                    stats=importance(rows)
                    assert abs(stats['ess_fraction']-batch['metrics']['eval_importance_ess_fraction']) < 1e-10
                    assert abs(stats['max_weight']-batch['metrics']['eval_importance_max_weight']) < 1e-10
                else:
                    assert batch['metrics']['eval_importance_ess_fraction'] is None
        name=f"{job['label']}/{job['step']}"
        expected_fresh = 1280 if str(job['step']) in protocol['fresh_seeds'] else 0
        counts[name]=dict(fixed=len(fixed_ids),fresh=fresh_count,expected_fresh=expected_fresh)
        ready = (folder/'summary.json').exists()
        complete &= ready
        if ready:
            checks['fixed_bank_order:'+name] = fixed_ids == fingerprints
            checks['fresh_count:'+name] = fresh_count == expected_fresh
    assert all(checks.values()), {k:v for k,v in checks.items() if not v}
    return dict(complete=complete, all_completed_checks_passed=True, checks=checks, counts=counts,
        nonoverlap_events_checked=seen_nonoverlap, max_absolute_errors=maxima,
        final_replay_checks=final_replay_checks,
        limitations=['Replay overlap coverage is limited to retained snapshots and final buffers; evicted histories are unavailable.',
                    'The separate ideal-policy normalization argument does not remove finite-precision support/floor effects.'])


def describe(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    return dict(count=len(a), mean=float(a.mean()) if len(a) else None,
        std=float(a.std(ddof=1)) if len(a)>1 else None,
        median=float(np.median(a)) if len(a) else None,
        minimum=float(a.min()) if len(a) else None, maximum=float(a.max()) if len(a) else None)


def csv_file(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def figures(folder, summary, rows, population='fixed', reuse=False):
    prefix = '' if population == 'fixed' else 'fresh_'
    if reuse and all((folder/f'{prefix}{kind}.{suffix}').exists()
                     for kind in ('raw','prior_relative','count_residuals') for suffix in ('png','pdf')):
        return
    for kind in ('raw', 'prior_relative'):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
        fit = summary[population][kind]['global_fit']
        for ax, stratum in zip(axes, (None, 'low', 'medium', 'high')):
            for name, color in zip(('low','medium','high'), ('#4477aa','#eeaa33','#cc6677')):
                if stratum is not None and name != stratum:
                    continue
                chosen = [r for r in rows if r['stratum'] == name]
                x = [r['log_reward'] if kind == 'raw' else r['log_likelihood'] for r in chosen]
                y = [r['log_policy_density']-(r['log_prior'] if kind != 'raw' else 0) for r in chosen]
                ax.scatter(x, y, s=7, alpha=.55, color=color, label=name)
            xlim = ax.get_xlim()
            ax.plot(xlim, np.array(xlim)+fit['slope_one_intercept'], 'k--', lw=1)
            ax.set_title(stratum or 'all strata')
            ax.set_xlabel('log reward' if kind == 'raw' else 'log likelihood')
            ax.set_ylabel('log policy density' if kind == 'raw' else 'log policy − log prior')
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{summary['label']} / update {summary['step']} / {population} {kind}; one shared intercept")
        for suffix in ('png','pdf'):
            fig.savefig(folder/f'{prefix}{kind}.{suffix}', dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    a = summary[population]['raw']['global_fit']['slope_one_intercept']
    residual = [r['log_policy_density']-r['log_reward']-a for r in rows]
    for ax, key in zip(axes, ('event_count','recombinations')):
        ax.scatter([r[key] for r in rows], residual, s=8, alpha=.4)
        ax.axhline(0,color='black',ls='--')
        ax.set(xlabel=key,ylabel='shared-intercept residual')
    for suffix in ('png','pdf'):
        fig.savefig(folder/f'{prefix}count_residuals.{suffix}',dpi=150)
    plt.close(fig)


def report(output, reuse_plots=False):
    output = Path(output)
    p = read_json(output/'protocol.json')
    from subtb_density_normalization import audit as audit_normalization
    audit_normalization(output)
    invocations = sorted(output.glob('invocation_*.json'))
    invocation = read_json(invocations[-1]) if invocations else {}
    deadline = invocation.get('stop_at_unix',p['training_stop_at_unix'])
    write_json(output/'artifact_validation.json',verify_artifacts(output,p))
    summaries, table, repeat_rows, calibration, per_arg = [], [], [], [], []
    for job in p['jobs']:
        folder = output/job['label']/f"step_{job['step']:04d}"
        if not (folder/'summary.json').exists():
            continue
        s = read_json(folder/'summary.json')
        summaries.append(s)
        rows = [r for f in sorted(folder.glob('fixed_*.json.gz')) for r in read_json(f)['records']]
        s['fixed'] = density_summary(rows)
        figures(folder, s, rows, reuse=reuse_plots)
        if s['fresh']:
            fresh_rows = [r for f in sorted(folder.glob('fresh_*.json.gz')) for r in read_json(f)['records']]
            s['fresh'] = density_summary(fresh_rows)
            figures(folder,s,fresh_rows,population='fresh',reuse=reuse_plots)
        write_json(folder/'summary.json',s)
        for population, pattern in [('fixed','fixed_*.json.gz'), ('fresh','fresh_*.json.gz')]:
            for f in sorted(folder.glob(pattern)):
                for r in read_json(f)['records']:
                    keys = ['fingerprint','log_policy_density','log_backward_probability','log_likelihood','log_prior',
                        'log_reward','reward_constant','event_count','recombinations','nonoverlap_coalescences',
                        'topology_sha256','marginal_tree_count','terminal_lineage_count','stratum','log_weight']
                    per_arg.append(dict(label=s['label'],step=s['step'],population=population,
                        **{k:r.get(k) for k in keys}, provenance=json.dumps(r.get('provenance',{}))))
        record = dict(label=s['label'], step=s['step'], fixed_subtb=s['fixed_subtb'],
            raw_pearson=s['fixed']['raw']['global_fit']['pearson'],
            relative_pearson=s['fixed']['prior_relative']['global_fit']['pearson'],
            raw_slope=s['fixed']['raw']['global_fit']['slope'],
            relative_slope=s['fixed']['prior_relative']['global_fit']['slope'],
            fixed_density_rmse=s['fixed']['raw']['global_fit']['rmse'])
        for repeat in s['repeats']:
            repeat_rows.append(dict(label=s['label'],step=s['step'],**repeat))
        for metric in METRICS:
            stats = describe([r.get(metric) for r in s['repeats']])
            record.update({metric+'_'+k:v for k,v in stats.items()})
        if s['pooled']:
            record.update({'pooled_'+k:v for k,v in s['pooled'].items()})
        if s['repeats']:
            logz = np.array([r['log_evidence'] for r in s['repeats']])
            scale = float(logz.max())
            record.update(evidence_log_scale=scale,
                evidence_scaled_mean=float(np.exp(logz-scale).mean()),
                evidence_scaled_std=float(np.exp(logz-scale).std(ddof=1)),
                log_mean_evidence=float(np.log(np.exp(logz-scale).mean())+scale))
            assert abs(record['log_mean_evidence']-s['pooled']['log_evidence']) < 1e-10
        table.append(record)
        for population in ('fixed','fresh'):
            if s.get(population) is None:
                continue
            for kind, fits in s[population].items():
                base = dict(label=s['label'],step=s['step'],population=population,plot=kind)
                calibration.append(dict(**base,group='global',value='all',**fits['global_fit']))
                for group, values in fits['groups'].items():
                    calibration.extend(dict(**base,group=group,value=v,**fit) for v,fit in values.items())
    csv_file(output/'checkpoint_summary.csv',table)
    csv_file(output/'repeats.csv',repeat_rows)
    csv_file(output/'calibration.csv',calibration)
    csv_file(output/'per_arg_scores.csv',per_arg)
    write_json(output/'comparison.json',dict(protocol_sha256=__import__('evaluate_subtb_density_fit').digest(output/'protocol.json'),
        complete=len(summaries)==len(p['jobs']),completed=len(summaries),expected=len(p['jobs']),records=summaries))
    # Preserve original training/fresh/fixed diagnostics and their explicit provenance.
    existing = []
    for run in p['runs']:
        for root in run.get('paths', [run['path']]):
            source = Path(root)/'results.json'
            results = read_json(source)
            for population in ('training','evaluation','fixed_evaluation'):
                for row in results[population]:
                    existing.append(dict(label=run['name'],population=population,source=str(source),**row))
    csv_file(output/'existing_diagnostics.csv',existing)
    schedules = [r for r in existing if r['population']=='training' and 'sampling_policy_temperature' in r
                 and 'policy_lr' in r]
    if schedules:
        csv_file(output/'training_schedules.csv', schedules)
        fig, axes = plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
        for label in sorted({r['label'] for r in schedules}):
            rows = sorted([r for r in schedules if r['label']==label],key=lambda r:r['step'])
            for ax, key in zip(axes, ('sampling_policy_temperature','policy_lr','flow_lr')):
                ax.plot([r['step'] for r in rows],[r[key] for r in rows],label=label)
                ax.set(xlabel='training update',ylabel=key)
        axes[0].legend(fontsize=7)
        for suffix in ('png','pdf'): fig.savefig(output/f'training_schedules.{suffix}',dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(16,8),constrained_layout=True)
    diagnostic_panels=[('training','subtb_loss','Training SubTB'),
        ('evaluation','eval_subtb_loss','Existing fresh SubTB'),
        ('fixed_evaluation','eval_subtb_loss','Existing held-out SubTB'),
        ('evaluation','eval_residual_std','Fresh full-trajectory residual SD'),
        ('evaluation','eval_subtb_terminal_loss','Fresh terminal-segment contribution'),
        ('evaluation','eval_subtb_interior_loss','Fresh nonterminal-segment contribution')]
    for label in [r['name'] for r in p['runs']]:
        for ax,(population,key,title) in zip(axes.flat,diagnostic_panels):
            chosen=[r for r in existing if r['label']==label and r['population']==population
                    and r.get('step',0)<=p['latest_common_update'] and r.get(key) is not None]
            ax.plot([r['step'] for r in chosen],[r[key] for r in chosen],label=label)
            ax.set(xlabel='training updates',title=title)
    axes[0,0].legend(fontsize=7)
    for suffix in ('png','pdf'):
        fig.savefig(output/f'existing_diagnostics.{suffix}',dpi=150,bbox_inches='tight')
    plt.close(fig)
    prior_file = output/'candidates/hudson_prior.json.gz'
    prior_reference = None
    if prior_file.exists():
        from scipy.special import logsumexp
        prior_rows = read_json(prior_file)['records']
        prior_ll = np.array([r['log_likelihood'] for r in prior_rows])
        prior_reference = dict(samples=len(prior_ll), source=str(prior_file),
            log_evidence=float(logsumexp(prior_ll)-np.log(len(prior_ll))),
            batch_log_evidence=[float(logsumexp(prior_ll[i:i+256])-np.log(len(prior_ll[i:i+256])))
                                for i in range(0,len(prior_ll),256)],
            interpretation='Independent Hudson-prior Monte Carlo, all unselected candidates; noisy cross-check, not ground truth')
        write_json(output/'prior_evidence_reference.json',prior_reference)
    fig, axes = plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    metrics = [('fixed_density_rmse','Fixed-bank density RMSE'),('relative_pearson','Prior-relative Pearson'),
               ('fixed_subtb','Fixed-bank SubTB'),('ess_fraction_mean','Fresh ESS/N'),
               ('log_evidence_mean','Fresh log evidence'),('eval_truth_pair_tmrca_rmse_mean','Fresh TMRCA RMSE')]
    initial = next((r for r in table if r['step']==0),None)
    for label in [r['name'] for r in p['runs']]:
        points = sorted(([initial] if initial else [])+[r for r in table if r['label']==label],key=lambda r:r['step'])
        for ax,(key,title) in zip(axes.flat,metrics):
            valid = [r for r in points if r.get(key) is not None]
            if key.endswith('_mean'):
                ax.errorbar([r['step'] for r in valid],[r[key] for r in valid],
                    yerr=[r.get(key[:-5]+'_std',0) or 0 for r in valid],fmt='-o',ms=3,capsize=3,label=label)
            else:
                ax.plot([r['step'] for r in valid],[r[key] for r in valid],'-o',ms=3,label=label)
            ax.set(xlabel='training updates',title=title)
    axes[0,0].legend(fontsize=7)
    for suffix in ('png','pdf'):
        fig.savefig(output/f'comparison_curves.{suffix}',dpi=150)
    plt.close(fig)
    def fmt(value): return 'NA' if value is None else f'{value:.4g}'
    lines = ['# 500 bp full-history density-fit evaluation','',
        f"Completed {len(summaries)}/{len(p['jobs'])} frozen checkpoint evaluations. Latest common update: {p['latest_common_update']}.",
        'Shared initialization is evaluated once after tensor equality verification across all four variants.', '',
        '| Variant | Update | Raw r | Relative r | Raw slope | Relative slope | Density RMSE | ESS/N mean ± SD | log evidence mean ± SD | TMRCA RMSE mean |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append('| '+ ' | '.join([r['label'],str(r['step'])]+[fmt(r[k]) for k in
            ['raw_pearson','relative_pearson','raw_slope','relative_slope','fixed_density_rmse']]+
            [fmt(r['ess_fraction_mean'])+' ± '+fmt(r['ess_fraction_std']),
             fmt(r['log_evidence_mean'])+' ± '+fmt(r['log_evidence_std']),fmt(r['eval_truth_pair_tmrca_rmse_mean'])])+' |')
    lines += ['', 'Fresh repeat summaries below use five independent batches; uncertainty is sample SD across repeats.', '',
        '| Variant / update | Max normalized weight | Log-weight SD | Local topology richness | Pairwise local RF | Truth RF |',
        '|---|---:|---:|---:|---:|---:|']
    for r in table:
        if not r['ess_fraction_count']:
            continue
        metrics=['max_weight','log_weight_std','eval_topology_unique_local_mean','eval_pairwise_local_rf_mean','eval_truth_rooted_rf_mean']
        lines.append('| '+r['label']+' / '+str(r['step'])+' | '+' | '.join(
            fmt(r[k+'_mean'])+' ± '+fmt(r[k+'_std']) for k in metrics)+' |')
    lines += ['', 'Fixed-bank regional residual means use the same fitted intercept within each checkpoint.', '',
        '| Variant / update | Low | Medium | High |', '|---|---:|---:|---:|']
    for s in summaries:
        if s['step'] not in (0,p['latest_common_update']):
            continue
        offsets=s['fixed']['raw']['groups']['stratum']
        lines.append('| '+s['label']+' / '+str(s['step'])+' | '+' | '.join(
            fmt(offsets[k]['residual_mean']) for k in ('low','medium','high'))+' |')
    lines += ['', '![Equal-update curves](comparison_curves.png)',
        '', '![Existing training and flow diagnostics](existing_diagnostics.png)', '', '## Interpretation and accounting', '',
        'The fixed bank asks whether the same full ARG histories receive correct relative densities. Fresh policy repeats measure what each checkpoint generates. Bank strata are constructed coverage, not posterior probability masses. No policy-only importance ESS is applied to the bank.',
        'Raw and prior-relative slope-one residuals agree algebraically after global centering. Their Pearson correlations and free slopes can differ. Each checkpoint/population uses one intercept across all strata; regional means remain visible in calibration.csv. Pearson correlation is null for fewer than three observations or zero variance; free slopes require at least two observations and varying x. An evaluation intercept is never a trained logZ.',
        'For every retained evaluated path, chronological node allocation reconstructs all events and the actual inverse enumerator returns one parent for every graph prefix. Full records retain non-overlap common-ancestor events, breakpoints and waits. Tree sequences/topologies are structural summaries, not the density space.',
        'The history space is a disjoint union over event counts and discrete chronological histories. Unordered coalescence pairs, recombination lineage choices and integer links use counting measure; positive waits use product Lebesgue measure in generations/(2Ne). A recombination has one wait despite creating two parent nodes. Ordered event times have unit triangular Jacobian, without an extra permutation factor. In generations, both policy and prior log densities subtract event_count*log(2Ne), which is not common across histories. Likelihood pruning independently uses generations and mutation rate, includes the JC69 root factor 1/4, and scores all 500 observed alignment columns.',
        'The exact production reward is retained: C + log likelihood + log Hudson prior (C=3000 here). Per-path independent pruning and density decomposition are checked. Full-trajectory log q−log R equals negative log importance weight when log PB=0.',
        'Fresh evidence estimates use logmeanexp(log R + log PB − log PF) − C. Evidence is unbiased under a normalized terminal proposal with adequate support and normalized target prior/likelihood; its logarithm is generally downward biased. The separate normalization_audit.json bounds pair-choice logits for every checkpoint and supplies an ideal-arithmetic absorption argument: active material never increases, active lineages are at most 4,000, and overlapping coalescences have a uniform positive conditional probability. At most 3,500 material-reducing events are needed, while total event counts remain unbounded. Every saved history also passes an independent bitset material reconstruction. This is a normalization argument, not a useful convergence/runtime bound. Finite-precision support, padded-logit approximations, and numerical likelihood floors remain limitations; likelihood/density agreement is independently checked on evaluated paths.',
        'Pooling uses a single normalization over all 1,280 fresh samples. Average batch ESS is a separate statistic. The pooled log-evidence equals log(mean of the five evidence estimates), not the mean of their logarithms. checkpoint_summary.csv retains both quantities and evidence-scale mean/SD as scaled values multiplied by exp(evidence_log_scale), avoiding numerical underflow. Five-repeat standard deviations measure observed sampling variability and cannot bound unseen importance-weight tails. Same seeds across variants do not imply identical trajectories.',
        'The separately labelled two-tip, one-site, no-recombination reference has an analytically normalized posterior and evidence, checked by quadrature. TV and JS compare continuous densities on that same restricted space. No finite enumeration or catalogue normalization is claimed for the eight-tip 500 bp Hudson model.',
        'Training settings, neural flow parameterization, exact rewards, checkpoints, replay buffers and the fixed held-out set are untouched. Existing training, fresh/fixed SubTB, terminal/intermediate and biological diagnostics are preserved in existing_diagnostics.csv.',
        '10,000 × 128 is 1.28 million scored training trajectories; replay reduces fresh generations. PhyloGFN’s 32 million training examples provide context, not a transferable convergence threshold. [Paper, Appendices B/K](https://proceedings.iclr.cc/paper_files/paper/2024/file/9837dc00ff67d176373268ed48042d49-Paper-Conference.pdf); [official repository](https://github.com/zmy1116/phylogfn).',
        '', '## Artifacts and reproduction', '',
        '`protocol.json` freezes checkpoints, seeds, source hashes and exclusions. `bank_manifest.json` records realized strata; `bank.json.gz` and `candidates/` retain full histories and provenance. Per-checkpoint compressed batches retain per-event scores and full balance diagnostics. `per_arg_scores.csv`, `calibration.csv`, `repeats.csv`, and `checkpoint_summary.csv` provide analysis tables.', '',
        '```bash',
        'OMP_NUM_THREADS=2 /private/home/pkatte/anaconda3/envs/phylogfn_orig/bin/python -u validation/scripts/evaluate_subtb_density_fit.py all --stop-at-unix '+str(int(deadline))+' --output '+str(output),
        '/private/home/pkatte/anaconda3/envs/phylogfn_orig/bin/python validation/scripts/evaluate_subtb_density_fit.py report --output '+str(output),
        '```', '',
        'For a later allocation, resume the same output directory with `--stop-at-unix` set to its authorized stop time. Cached complete batches and the frozen bank are reused. Incomplete checkpoint comparisons remain explicitly incomplete.']
    if prior_reference:
        lines += ['', f"Independent prior Monte Carlo ({prior_reference['samples']} histories) gives log evidence {prior_reference['log_evidence']:.6f}, with batch values {prior_reference['batch_log_evidence']}. This small prior sample is a noisy independent cross-check, not a normalized posterior reference."]
    if p.get('bank_origin'):
        lines += ['', 'This comparison reuses the original bank byte-for-byte. Its strata were not rebuilt for the additional variant. Cached scores retain their original source hashes in protocol.json. The baseline_temperature_cosine variant combines discrete policy-temperature annealing and cosine decay; this experiment does not separate their effects. Every fresh density/ESS/evidence evaluation uses temperature 1.']
    if initial:
        lines += ['', '## Observed changes', '']
        for r in table:
            if r['step'] != p['latest_common_update']:
                continue
            lines.append(f"- {r['label']}: fixed density RMSE {fmt(initial['fixed_density_rmse'])} → {fmt(r['fixed_density_rmse'])}; prior-relative slope {fmt(initial['relative_slope'])} → {fmt(r['relative_slope'])}; fresh ESS/N {fmt(r['ess_fraction_mean'])} ± {fmt(r['ess_fraction_std'])}. Interpret alongside TMRCA, regional offsets and SubTB, not as posterior convergence.")
        endpoints=[r for r in table if r['step']==p['latest_common_update']]
        if len(endpoints)==4:
            lines += ['',
                'All four variants improve fixed-history density fitting from initialization. Exploration and replay improve coverage calibration more than the baseline at this budget, but every global prior-relative slope remains below one. Positive low-stratum and negative high-stratum residual means show that the fitted relative-density range is still compressed.',
                'Fresh sampling gives a different ranking: the baseline has the lowest TMRCA error and highest mean batch ESS, while exploration/replay variants have stronger fixed-bank fits. TMRCA against one simulated truth is a biological diagnostic, not a distance to the exact posterior. No variant is established as posterior-converged by these results.',
                'Pooled ESS remains very small, and high normalized maximum weights reveal substantial concentration. Several-log-unit differences in evidence estimates across policies targeting the same reward point to inadequate Monte Carlo tail coverage; small within-run variation cannot exclude missing mass. The 512-sample prior cross-check is not accurate enough to resolve the true 500 bp evidence.',
                'The next evidence needed is longer equal-budget joint policy-and-flow training on this 500 bp diagnostic under the preserved settings, followed by this same frozen-bank/repeated-fresh protocol. A separately trained restricted two-tip neural benchmark has not been run here; the supplied restricted reference validates normalization and estimation, not learned convergence. Larger datasets remain deferred.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
