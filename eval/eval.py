#!/usr/bin/env python3
"""Evaluate a saved ARG GFlowNet using a training YAML and optional overrides."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import glob
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

# Put the package parent first, including when invoked as `python eval/eval.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import tskit
import yaml

from eval.density_fit import density_summary, select_bank
from eval.ess import importance_stats, log_importance_weights
from eval.posterior_summary import (TerminalSamplingEvaluator, compare_ensembles,
    ensemble_truth_metrics, summarize_ensemble, topology_signature, validate_tree_sequence)
from flow_training import preserve_sampling
from utils import action_as_dict, action_from_dict
from infer import environment_from_metadata, load_checkpoint, resolve_device, validate_metadata
from rollout_worker_arg import RolloutWorker
from tb_gfn import TBGFlowNetGenerator
from training_exploration import sample_prior_trajectories
from trajectory_buffer import action_fingerprint, environment_fingerprint
from utils import load_sequences

METRICS = ('density_fit', 'ess', 'posterior_summary')
OPTION_KEYS = {'checkpoint', 'metrics', 'output_dir', 'num_samples', 'batch_size', 'repeats',
               'seed', 'device', 'grid_size', 'density_bank', 'bank_per_stratum',
               'bank_candidates', 'baselines', 'rank_bins'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    path = Path(path)
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    opener = gzip.open(temporary, 'wt') if path.suffix == '.gz' else temporary.open('w')
    with opener as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    temporary.replace(path)


def repository_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def load_config(path, overrides=None):
    path = Path(path).expanduser()
    if not path.is_file():
        candidates = [ROOT/path, ROOT/'config'/path]
        path = next((p for p in candidates if p.is_file()), path)
    try:
        settings = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f'Cannot read evaluation config {path}: {exc}') from exc
    if not isinstance(settings, dict) or not isinstance(settings.get('evaluation', {}), dict):
        raise ValueError('YAML must be a mapping; evaluation must be a mapping when supplied')
    for key in ('dataset_path', 'output_path'):
        if not isinstance(settings.get(key), str) or not settings[key]:
            raise ValueError(f'Config requires {key}')
    custom = dict(settings.get('evaluation', {}))
    if set(custom)-OPTION_KEYS:
        raise ValueError(f'Unknown evaluation settings: {sorted(set(custom)-OPTION_KEYS)}')
    custom.update({k: v for k, v in (overrides or {}).items() if v is not None})
    options = dict(checkpoint='best_eval', metrics=list(METRICS),
        num_samples=settings.get('eval_episodes', 256), batch_size=32,
        repeats=settings.get('terminal_eval_repeats', 3), seed=int(settings.get('seed', 7))+100000,
        device='auto', grid_size=settings.get('terminal_eval_grid_size', 100),
        density_bank=None, bank_per_stratum=64, bank_candidates=768, baselines={}, output_dir=None, rank_bins=20)
    options.update(custom)
    for key in ('num_samples', 'batch_size', 'repeats', 'grid_size', 'bank_per_stratum', 'bank_candidates', 'rank_bins'):
        if isinstance(options[key], bool) or not isinstance(options[key], int) or options[key] < 1:
            raise ValueError(f'evaluation.{key} must be a positive integer')
    if not isinstance(options['seed'], int) or isinstance(options['seed'], bool) or not 0 <= options['seed'] < 2**32:
        raise ValueError('evaluation.seed must be an integer in [0, 2**32)')
    if (not isinstance(options['metrics'], list) or not options['metrics']
            or any(x not in METRICS for x in options['metrics'])
            or len(set(options['metrics'])) != len(options['metrics'])):
        raise ValueError(f'evaluation.metrics must be a nonempty unique list from {METRICS}')
    if not isinstance(options['baselines'], dict):
        raise ValueError('evaluation.baselines must be a mapping')
    options.update(config=str(path.resolve()), dataset_path=str(repository_path(settings['dataset_path'])),
                   run_dir=str(repository_path(settings['output_path'])),
                   loss_type=settings.get('loss_type', 'tb'))
    options['output_dir'] = str(repository_path(options['output_dir']) if options['output_dir']
                                else Path(options['run_dir'])/'evaluation')
    if options['density_bank']:
        options['density_bank'] = str(repository_path(options['density_bank']))
    return options


def select_checkpoint(options):
    selection = options['checkpoint']
    if selection != 'best_eval':
        if not isinstance(selection, str) or not selection:
            raise ValueError('evaluation.checkpoint must be best_eval or a checkpoint path')
        path = repository_path(selection)
        if not path.is_file():
            raise ValueError(f'Explicit checkpoint does not exist: {path}')
        return path
    name = 'best_eval_subtb_loss.pt' if options['loss_type'] == 'subtb' else 'best_eval_loss.pt'
    candidates = [Path(options['run_dir'])/'checkpoints'/name,
                  Path(options['run_dir'])/'best_on_policy.pt']
    for path in candidates:
        if path.is_file():
            return path
    raise ValueError('Best evaluation checkpoint not found. Checked '+', '.join(map(str, candidates))+
                     '. Set evaluation.checkpoint or --checkpoint to an existing file.')


@contextmanager
def preserve_global_rng():
    """Also isolate RNG consumed while constructing a model, before it has an env."""
    py, np_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)


def load_model(path, options):
    checkpoint = load_checkpoint(str(path), map_location='cpu')
    metadata = checkpoint.get('metadata', {})
    validate_metadata(metadata)
    fasta = Path(options['dataset_path'])
    names = [s[1:].split()[0] for s in fasta.read_text().splitlines() if s.startswith('>')]
    if len(names) != len(set(names)) or len(names) != metadata['num_sequences']:
        raise ValueError('FASTA sample names must be unique and match the checkpoint sample count')
    sequences = load_sequences(str(fasta))
    if sequences != list(metadata['sequences']) or any(len(s) != metadata['sequence_length'] for s in sequences):
        raise ValueError('Dataset sequences/order or dimensions differ from the checkpoint')
    if 'sample_names' in metadata and names != list(metadata['sample_names']):
        raise ValueError('FASTA sample names/order differ from the checkpoint')
    env = environment_from_metadata(metadata, options['seed'], resolve_device(options['device']))
    model = TBGFlowNetGenerator(env, init_z_sample_count=metadata['init_z_sample_count'],
        device=env.device, verbose=False, initialize_z_from_policy=False,
        model_kwargs=dict(metadata.get('model', {})), loss_type=metadata.get('loss_type', 'tb'),
        subtb_lambda=metadata.get('subtb_lambda', .9), flow_head_version=metadata.get('flow_head_version', 1),
        flow_lr=metadata.get('flow_lr', metadata.get('policy_lr', .001)))
    model.load(checkpoint, load_optimizer=False, map_location=env.device)
    return model, metadata, names


def natural_key(path):
    return [int(p) if p.isdigit() else p for p in re.split(r'(\d+)', str(path))]


def load_baselines(config, names, length):
    results = {}
    for method, spec in config.items():
        if method not in ('singer', 'tsinfer_tsdate', 'relate'):
            raise ValueError(f'Unknown baseline {method}; use singer, tsinfer_tsdate, or relate')
        if not isinstance(spec, dict) or set(spec)-{'trees', 'sample_order', 'burnin', 'stride', 'kind'}:
            raise ValueError(f'Invalid baseline settings for {method}')
        patterns = spec.get('trees')
        patterns = [patterns] if isinstance(patterns, str) else patterns
        if not isinstance(patterns, list) or not patterns or not all(isinstance(x, str) for x in patterns):
            raise ValueError(f'{method}.trees requires .trees paths or globs')
        paths = []
        for pattern in patterns:
            matches = sorted(glob.glob(str(repository_path(pattern))), key=natural_key)
            if not matches:
                raise ValueError(f'No baseline files matched {pattern}')
            paths.extend(matches)
        if len(paths) != len(set(paths)) or any(Path(p).suffix != '.trees' for p in paths):
            raise ValueError(f'{method} inputs must be distinct .trees files')
        order = spec.get('sample_order')
        if (not isinstance(order, list) or len(order) != len(names)
                or not all(isinstance(x, str) for x in order) or set(order) != set(names)):
            raise ValueError(f'{method}.sample_order must list FASTA names in ts.samples() order')
        burnin, stride = spec.get('burnin', 0), spec.get('stride', 1)
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (burnin, stride)) or burnin < 0 or stride < 1:
            raise ValueError('Baseline burnin must be nonnegative and stride positive')
        paths = paths[burnin::stride]
        if not paths:
            raise ValueError(f'{method} has no samples after burn-in/stride')
        trees, maps = [], []
        for path in paths:
            ts = tskit.load(path)
            if ts.num_samples != len(order):
                raise ValueError(f'{path}: sample count differs from the declared mapping')
            mapping = [int(ts.samples()[order.index(name)]) for name in names]
            validate_tree_sequence(ts, mapping, length, len(names))
            trees.append(ts); maps.append(mapping)
        kind = ('point' if len(trees) == 1 or method == 'tsinfer_tsdate'
                else 'conditional_times' if method == 'relate' else 'posterior')
        if spec.get('kind', kind) != kind or (kind == 'point' and len(trees) != 1):
            raise ValueError(f'{method}: expected {kind}; single-file outputs are point estimates, '
                             'tsinfer+tsdate needs one file, and Relate ensembles have conditional time uncertainty')
        results[method] = dict(trees=trees, maps=maps, kind=kind,
            provenance=dict(files=[dict(path=p, sha256=digest(p)) for p in paths],
                            sample_order=order, burnin=burnin, stride=stride, kind=kind))
    return results


def discover_truth(options, env):
    metadata = Path(options['dataset_path']).parent/'metadata.json'
    if not metadata.is_file() or not read_json(metadata).get('files', {}).get('ground_truth_trees'):
        return None
    return TerminalSamplingEvaluator.from_dataset(options['dataset_path'], env,
                                                  options['grid_size'], tmrca_method='point_accuracy')


def collect(model, count, batch_size, seed, *, fixed=None, prior=False, grid_size=100):
    """Generate once or rescore histories, retaining their exact terminal scores."""
    worker = RolloutWorker(model.env)
    records, trees = [], []
    if fixed is not None and len(fixed) != count:
        raise ValueError('Fixed history count differs from requested count')
    with preserve_sampling(model, seed), torch.no_grad():
        for start in range(0, count, batch_size):
            n = min(batch_size, count-start)
            paths = (fixed[start:start+n] if fixed is not None else
                     sample_prior_trajectories(model.env, n) if prior else None)
            if paths is None:
                outputs, trajectories = worker.rollout(model, n, return_states=True)
            else:
                if model.arg_model.event_policy != 'cwr_residual':
                    raise ValueError('Frozen density banks require a cwr_residual checkpoint; select other metrics for legacy cwr models')
                outputs, trajectories = worker.replay(model, paths, collect_flows=False, return_states=True)
            pf = outputs['log_paths_pf'].double().sum(-1).cpu().numpy()
            pb = outputs['log_paths_pb'].double().sum(-1).cpu().numpy()
            for j, (state, path) in enumerate(zip(outputs['states'], trajectories)):
                if not state.is_done:
                    raise ValueError('Evaluation encountered a nonterminal trajectory')
                reward, log_prior = float(state.log_reward), float(state.accumulated_log_prior)
                offset = float(model.env.reward_fn.C)
                likelihood = reward-offset-log_prior
                scores = [reward, log_prior, likelihood, pf[j], pb[j]]
                if not np.isfinite(scores).all():
                    raise ValueError('Evaluation encountered a nonfinite score; no draws were discarded')
                ts = model.env.save_to_tree_sequence(state)
                samples = list(map(int, ts.samples()))
                if samples != list(range(model.env.num_sequences)):
                    raise ValueError('Generated sample order differs from the model environment')
                validate_tree_sequence(ts, samples, model.env.sequence_length, model.env.num_sequences)
                positions = (np.arange(grid_size)+.5)*ts.sequence_length/grid_size
                topology = [list(topology_signature(ts.at(p), samples)) for p in positions]
                unique = bool(torch.all(outputs['log_paths_pb'][j] == 0))
                records.append(dict(actions=[action_as_dict(a) for a in path.actions], fingerprint=action_fingerprint(path.actions),
                    log_reward=reward, log_prior=log_prior, log_likelihood=likelihood,
                    log_policy_density=float(pf[j]), log_backward_probability=float(pb[j]),
                    log_weight=float(log_importance_weights([reward], [pf[j]], [pb[j]])[0]),
                    reward_constant=offset, event_count=len(path),
                    recombinations=sum(a.event_type == 'recomb' for a in path.actions),
                    topology_sha256=object_hash(topology), unique_backward_history=unique,
                    density_space='full_history' if unique else 'trajectory_balance',
                    provenance=dict(source='prior' if prior else 'fixed' if fixed is not None else 'policy', seed=seed)))
                trees.append(ts)
    return records, trees


def validate_bank(bank, path, identity):
    if not isinstance(bank, dict) or not bank.get('records'):
        raise ValueError('Density bank must contain complete history records')
    if 'identity' in bank:
        if bank['identity'] != identity:
            raise ValueError('Density bank dataset, prior, or time representation is incompatible')
    else:
        # Legacy banks record identity in their adjacent immutable protocol.
        protocol_path = Path(path).parent/'protocol.json'
        if not protocol_path.is_file():
            raise ValueError('Legacy density bank requires its adjacent protocol.json')
        protocol = read_json(protocol_path)
        if protocol.get('dataset_sha256') != identity['dataset_sha256']:
            raise ValueError('Legacy density bank belongs to a different dataset')
        origin = protocol.get('bank_origin', {})
        if origin.get('sha256') and origin['sha256'] != digest(path):
            raise ValueError('Legacy bank differs from the recorded frozen origin')
        if bank.get('protocol_sha256') != origin.get('protocol_sha256', digest(protocol_path)):
            raise ValueError('Legacy bank protocol hash differs from its recorded provenance')
    seen = set()
    for row in bank['records']:
        if not row.get('actions') or action_fingerprint(row['actions']) != row.get('fingerprint'):
            raise ValueError('Invalid density bank action fingerprint')
        if row['fingerprint'] in seen:
            raise ValueError('Density bank contains duplicate histories')
        seen.add(row['fingerprint'])


def get_bank(model, options, identity, checkpoint_hash):
    explicit = options.get('density_bank')
    path = Path(explicit) if explicit else Path(options['output_dir'])/'density_bank.json.gz'
    if path.exists():
        bank = read_json(path)
        validate_bank(bank, path, identity)
        if not explicit:
            counts = [sum(r.get('stratum') == name for r in bank['records'])
                      for name in ('low', 'medium', 'high')]
            if counts != [options['bank_per_stratum']]*3:
                raise ValueError(f'Frozen automatic bank has stratum sizes {counts}, which differ from '
                                 'bank_per_stratum. Match the original size, explicitly select density_bank, '
                                 'or use a new output_dir to create another bank.')
        return bank, path
    if explicit:
        raise ValueError(f'Configured density bank does not exist: {path}')
    count = options['bank_candidates']
    if count % 2 or count < 3*options['bank_per_stratum']:
        raise ValueError('bank_candidates must be even and at least three times bank_per_stratum')
    candidates = []
    for prior, offset in ((True, 700001), (False, 800003)):
        rows, _ = collect(model, count//2, options['batch_size'],
            (options['seed']+offset) % 2**32, prior=prior, grid_size=options['grid_size'])
        candidates.extend(rows)
    bank = select_bank(candidates, options['bank_per_stratum'], set())
    bank.update(schema_version=1, identity=identity, source_checkpoint_sha256=checkpoint_hash,
                frozen_utc=datetime.now(timezone.utc).isoformat(),
                selection='Equal prior/policy candidate counts; equal low/medium/high reward strata; evaluation-only',
                grid_size=options['grid_size'])
    validate_bank(bank, path, identity)
    write_json(path, bank)
    return bank, path


def weight_summary(rows):
    offsets = [r['reward_constant'] for r in rows]
    if not offsets or any(c != offsets[0] for c in offsets):
        raise ValueError('Fresh evaluation rows must share one reward offset')
    return importance_stats([r['log_weight'] for r in rows], reward_constant=offsets[0])


def posterior_report(trees, names, ne, grid_size, truth, baselines, rank_bins=20):
    methods = {'gfn': dict(trees=trees, maps=[list(map(int, t.samples())) for t in trees], kind='posterior'), **baselines}
    features, result = {}, dict(methods={}, comparisons={}, unavailable_baselines=[])
    for name, sample in methods.items():
        feature = summarize_ensemble(sample['trees'], sample['maps'], ne, grid_size)
        features[name] = feature
        entry = dict(kind=sample['kind'], draw_count=len(sample['trees']),
                     medoid_index=feature['medoid_index'], medoid_mean_grid_rf=feature['medoid_mean_grid_rf'],
                     covariance=feature['covariance'],
                     mean_pair_tmrca=float(feature['times'].mean()),
                     truth_status='unavailable' if truth is None else 'ok',
                     tmrca_calibration=dict(status='truth_unavailable'))
        if truth is not None:
            metrics, details = ensemble_truth_metrics(sample['trees'], sample['maps'], truth.truth,
                                                      truth.truth_samples, ne, feature, rank_bins=rank_bins)
            calibration = details.pop('tmrca_calibration')
            if sample['kind'] == 'point' or len(sample['trees']) < 2:
                # Retain legacy descriptive interval coverage, but do not
                # advertise point estimates as calibrated posterior ensembles.
                entry['tmrca_calibration'] = dict(status='requires_multiple_draws', kind=sample['kind'])
                metrics.pop('eval_truth_tmrca_rank_kl')
                metrics.pop('eval_truth_tmrca_rank_tie_fraction')
            else:
                entry['tmrca_calibration'] = dict(calibration, status='ok', kind=sample['kind'])
            entry.update(metrics=metrics, truth_details=details)
        result['methods'][name] = entry
    if 'singer' in features and baselines['singer']['kind'] == 'posterior':
        result['comparisons']['gfn_vs_singer'] = compare_ensembles(features['gfn'], features['singer'])
        result['comparisons']['gfn_vs_singer']['interpretation'] = 'Reference agreement; SINGER uses an SMC approximation, not the full Hudson target'
    else:
        result['comparisons']['gfn_vs_singer'] = dict(status='requires_singer_posterior_samples')
    result['unavailable_baselines'] = [x for x in ('singer', 'tsinfer_tsdate', 'relate') if x not in baselines]
    result['protocol'] = dict(sample_names=names, positions=features['gfn']['positions'].tolist(),
        pairs=[list(p) for p in features['gfn']['pairs']], time_units='2 Ne', time_divisor=2*ne,
        truth_weighting='Exact genomic spans, equal haplotype-pair weights',
        reference_weighting='Equal genomic midpoint-grid and haplotype-pair weights',
        rank_bins=rank_bins, coverage_levels=[.5, .7, .9],
        calibration_scope='Truth-based summaries of this dataset; Relate ensembles describe conditional time uncertainty')
    return result


def flatten_scalars(value, prefix=''):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(flatten_scalars(child, f'{prefix}.{key}' if prefix else key))
    elif value is None or isinstance(value, (str, int, float, bool)):
        result[prefix] = value
    return result


def write_calibration_report(folder, result):
    """Export all repeat/pooled summaries; plot the pooled posterior ensembles."""
    summaries = [(f"repeat_{r['repeat']}", r['metrics'].get('posterior_summary')) for r in result['repeats']]
    summaries.append(('pooled', result.get('posterior_summary')))
    rank_rows, coverage_rows, pooled = [], [], []
    for population, summary in summaries:
        for name, entry in (summary or {}).get('methods', {}).items():
            calibration = entry.get('tmrca_calibration', {})
            if calibration.get('status') != 'ok':
                continue
            rank = calibration['rank_histogram']
            shared = dict(population=population, method=name, kind=entry['kind'], draw_count=entry['draw_count'])
            for index, (probability, uniform) in enumerate(zip(rank['probabilities'], rank['uniform_probabilities'])):
                rank_rows.append(dict(shared, bin_index=index,
                    rank_min=int(rank['bin_edges'][index]+.5), rank_max=int(rank['bin_edges'][index+1]-.5),
                    probability=probability, uniform_probability=uniform,
                    kl_from_uniform_nats=rank['kl_from_uniform'], tie_cell_fraction=rank['tie_cell_fraction']))
            for interval in calibration['interval_coverage']['intervals']:
                coverage_rows.append(dict(shared, **{key: interval[key] for key in
                    ('level', 'coverage', 'coverage_error', 'mean_width', 'lower_quantile', 'upper_quantile')},
                    time_units='2 Ne'))
            if population == 'pooled':
                pooled.append((name, entry, calibration))
    for filename, rows in [('tmrca_ranks.csv', rank_rows), ('interval_coverage.csv', coverage_rows)]:
        if rows:
            with (folder/filename).open('w', newline='') as handle:
                writer = csv.DictWriter(handle, list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if not pooled:
        return ['TMRCA calibration unavailable: simulated truth and multiple ARG draws are required.', '']
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(pooled), figsize=(5*len(pooled), 4), squeeze=False)
    for ax, (name, entry, calibration) in zip(axes[0], pooled):
        rank = calibration['rank_histogram']
        edges = (np.asarray(rank['bin_edges'])+.5)/(rank['draw_count']+1)
        ax.bar(edges[:-1], rank['probabilities'], width=np.diff(edges), align='edge', alpha=.7, label='Observed')
        ax.stairs(rank['uniform_probabilities'], edges, color='black', linestyle='--', label='Uniform-rank reference')
        ax.set(xlabel='Normalized rank category', ylabel='Span-weighted probability',
               title=f"{name} ({entry['kind']})\nKL = {rank['kl_from_uniform']:.4g} nats")
        ax.legend(fontsize='small')
    fig.tight_layout(); fig.savefig(folder/'tmrca_ranks.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], 'k--', label='Nominal coverage')
    lines = ['| Method | Rank KL (nats) | 50% coverage | 70% coverage | 90% coverage |',
             '|---|---:|---:|---:|---:|']
    for name, entry, calibration in pooled:
        intervals = calibration['interval_coverage']['intervals']
        ax.plot([i['level'] for i in intervals], [i['coverage'] for i in intervals], 'o-',
                label=f"{name} ({entry['kind']})")
        coverages = ' | '.join(f"{i['coverage']:.3%}" for i in intervals)
        lines.append(f"| {name} ({entry['kind']}) | {calibration['rank_histogram']['kl_from_uniform']:.6g} | {coverages} |")
    ax.set(xlabel='Nominal interval level', ylabel='Span-weighted truth coverage', xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize='small')
    fig.tight_layout(); fig.savefig(folder/'interval_coverage.png', dpi=160); plt.close(fig)
    return lines + ['', 'These describe one dataset. Linked positions/pairs are dependent; sampling repeats '
                    'do not replace independently simulated datasets for calibration. '
                    'KL is descriptive, with no independent-observation significance test. '
                    'Exact ties are averaged over possible ranks and their fraction is recorded in the CSV. '
                    'Relate ensembles describe conditional branch-time uncertainty.', '',
                    '[Rank histogram and KL CSV](tmrca_ranks.csv) · [Coverage CSV](interval_coverage.csv)', '']


def write_report(folder, result, fixed_rows, fresh_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [dict(population='pooled', **flatten_scalars(result['pooled']))]
    rows.extend(dict(population=f"repeat_{r['repeat']}", **flatten_scalars(r['metrics'])) for r in result['repeats'])
    if result.get('density_fit'):
        rows.append(dict(population='fixed_bank', **flatten_scalars(result['density_fit'])))
    if result.get('posterior_summary'):
        rows.append(dict(population='posterior_summary', **flatten_scalars(result['posterior_summary'])))
    fields = ['population']+sorted(set().union(*(set(r) for r in rows))-{'population'})
    with (folder/'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields); writer.writeheader(); writer.writerows(rows)
    lines = ['# ARG checkpoint evaluation', '', f"Checkpoint: `{result['checkpoint']}`", '',
             f"Training update: {result['step']}; fresh draws: {len(fresh_rows)}.", '',
             'The fixed bank measures relative density fit. ESS uses fresh policy draws only.', '']
    if 'ess' in result['pooled']:
        stats = result['pooled']['ess']
        lines.extend([f"Pooled ESS: {stats['ess']:.3f} / {stats['episodes']} ({stats['ess_fraction']:.3%}).",
                      f"Log-weight SD: {stats['log_weight_std']:.6g}; largest weight: {stats['max_normalized_weight']:.6g}.", ''])
    if result.get('density_fit'):
        fit = result['density_fit']['prior_relative']['global_fit']
        lines.extend([f"Fixed-bank prior-relative slope: {fit['slope']}; Pearson r: {fit['pearson']}; centered RMSE: {fit['rmse']}.",
                      f"Density interpretation: {result['density_space']}.", ''])
    if result.get('posterior_summary'):
        lines.extend(['| Method | TMRCA RMSE | Mean RF | Medoid RF |', '|---|---:|---:|---:|'])
        for name, entry in result['posterior_summary']['methods'].items():
            m = entry.get('metrics', {})
            lines.append(f"| {name} | {m.get('eval_truth_pair_tmrca_rmse', 'unavailable')} | {m.get('eval_truth_rooted_rf_mean', 'unavailable')} | {m.get('eval_truth_rooted_rf_medoid', 'unavailable')} |")
        lines.extend(['', 'Unavailable baselines: '+', '.join(result['posterior_summary']['unavailable_baselines']), ''])
        lines.extend(write_calibration_report(folder, result))
    lines.extend(['[Machine-readable results](results.json) · [Metrics CSV](metrics.csv)', ''])
    (folder/'report.md').write_text('\n'.join(lines))
    if fixed_rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, kind in zip(axes, ('raw', 'prior_relative')):
            x = np.array([r['log_reward'] if kind == 'raw' else r['log_likelihood'] for r in fixed_rows])
            y = np.array([r['log_policy_density']-r['log_backward_probability']-(r['log_prior'] if kind == 'prior_relative' else 0) for r in fixed_rows])
            intercept = result['density_fit'][kind]['global_fit']['slope_one_intercept']
            ax.scatter(x, y, s=7); bounds = np.array([x.min(), x.max()]); ax.plot(bounds, bounds+intercept, 'k--')
            ax.set(xlabel='Log reward' if kind == 'raw' else 'Log likelihood',
                   ylabel='Log PF − log PB' if kind == 'raw' else 'Log PF − log PB − log prior', title=kind)
        fig.tight_layout(); fig.savefig(folder/'density_fit.png', dpi=160); plt.close(fig)
    if 'ess' in result['pooled']:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([str(r['repeat']) for r in result['repeats']], [r['metrics']['ess']['ess_fraction'] for r in result['repeats']])
        ax.axhline(result['pooled']['ess']['ess_fraction'], color='black', linestyle='--', label='Pooled')
        ax.set(xlabel='Sampling repeat', ylabel='ESS / N', ylim=(0, 1)); ax.legend()
        fig.tight_layout(); fig.savefig(folder/'ess.png', dpi=160); plt.close(fig)
    if result.get('posterior_summary'):
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, sample in result['posterior_summary']['methods'].items():
            curve = sample['covariance']['curves']
            if curve:
                ax.plot([r['distance_bp'] for r in curve], [r['covariance'] for r in curve], label=name)
        ax.set(xlabel='Genomic separation (bp)', ylabel='Posterior TMRCA covariance (2Ne)²')
        if ax.lines: ax.legend()
        fig.tight_layout(); fig.savefig(folder/'posterior_covariance.png', dpi=160); plt.close(fig)


def run_evaluation(options):
    """Callable runner; all global and model sampling state is restored on exit."""
    with preserve_global_rng():
        return _run_evaluation(options)


def _run_evaluation(options):
    checkpoint = select_checkpoint(options)
    checkpoint_hash = digest(checkpoint)
    model, metadata, names = load_model(checkpoint, options)
    if digest(checkpoint) != checkpoint_hash:
        raise ValueError('Checkpoint changed while loading; retry with a stable saved checkpoint')
    identity = dict(dataset_sha256=digest(options['dataset_path']),
                    environment_sha256=environment_fingerprint(model.env),
                    time_metadata=model.env.time_metadata)
    truth = discover_truth(options, model.env) if 'posterior_summary' in options['metrics'] else None
    baselines = load_baselines(options['baselines'], names, model.env.sequence_length)
    fixed_rows, bank_path = [], None
    if 'density_fit' in options['metrics']:
        bank, bank_path = get_bank(model, options, identity, checkpoint_hash)
    protocol = dict(version=1, options=options, identity=identity, checkpoint_sha256=checkpoint_hash,
                    sample_names=names, temperature=1.,
                    bank_sha256=digest(bank_path) if bank_path else None,
                    truth_protocol=truth.protocol if truth is not None else None,
                    baselines={name: b['provenance'] for name, b in baselines.items()},
                    source_sha256={str(p.relative_to(ROOT)): digest(p) for p in
                                   [*sorted((ROOT/'eval').glob('*.py')),
                                    *[ROOT/name for name in ('rollout_worker_arg.py', 'infer.py',
                                       'env/env.py', 'env/actions.py', 'env/states.py', 'evo.py', 'tb_gfn.py', 'models.py', 'time_model.py',
                                       'time_env.py', 'breakpoint_model.py', 'trajectory_buffer.py',
                                       'training_exploration.py', 'flow_training.py', 'utils.py',
                                       'lineage_features.py', 'flow_encoder.py', 'flow_likelihood.py')]]})
    protocol_hash = object_hash(protocol)
    step = int(metadata.get('epoch', -1))+1
    folder = Path(options['output_dir'])/f'step_{step:06d}_{checkpoint_hash[:12]}_{protocol_hash[:12]}'
    result_file = folder/'results.json'
    if result_file.is_file():
        existing = read_json(result_file)
        artifacts_valid = bool(existing.get('artifacts')) and all(
            (folder/name).is_file() and digest(folder/name) == sha
            for name, sha in existing.get('artifacts', {}).items())
        if existing.get('complete') and existing.get('protocol_sha256') == protocol_hash and artifacts_valid:
            print(f'Reusing completed evaluation: {folder}', flush=True)
            return existing
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder/'protocol.json', protocol)
    if bank_path:
        print(f'Scoring {len(bank["records"])} frozen histories', flush=True)
        fixed_rows, _ = collect(model, len(bank['records']), options['batch_size'],
            (options['seed']+900007) % 2**32, fixed=[[action_from_dict(a) for a in r['actions']] for r in bank['records']], grid_size=options['grid_size'])
        for actual, saved in zip(fixed_rows, bank['records']):
            if actual['fingerprint'] != saved['fingerprint']:
                raise ValueError('Rescored history actions differ from the frozen bank')
            for key in ('log_reward', 'log_prior', 'log_likelihood'):
                if key in saved and not math.isclose(actual[key], saved[key], rel_tol=0, abs_tol=1e-6):
                    raise ValueError(f'Frozen bank {key} differs under this checkpoint; target mismatch')
            actual.update(stratum=saved.get('stratum', 'unstratified'), provenance=saved.get('provenance', {}))
        write_json(folder/'fixed_scores.json.gz', fixed_rows)
    rows, trees, repeats = [], [], []
    for repeat in range(options['repeats']):
        seed = (options['seed']+repeat*1000003) % 2**32
        print(f'Evaluation repeat {repeat+1}/{options["repeats"]}: {options["num_samples"]} fresh ARGs', flush=True)
        current, sampled = collect(model, options['num_samples'], options['batch_size'], seed, grid_size=options['grid_size'])
        for i, (row, tree) in enumerate(zip(current, sampled)):
            row.update(repeat=repeat, sample_index=i)
            filename = f'repeat_{repeat:03d}/arg_{i:06d}.trees'
            destination = folder/filename; destination.parent.mkdir(parents=True, exist_ok=True)
            tree.dump(destination); row['tree_file'] = filename
        scores = {}
        if 'ess' in options['metrics']: scores['ess'] = weight_summary(current)
        if 'posterior_summary' in options['metrics']:
            scores['posterior_summary'] = posterior_report(sampled, names, model.env.population_size,
                                                           options['grid_size'], truth, baselines, options['rank_bins'])
        repeats.append(dict(repeat=repeat, seed=seed, metrics=scores))
        write_json(folder/f'repeat_{repeat:03d}/scores.json.gz', current)
        rows.extend(current); trees.extend(sampled)
    pooled = {}
    if 'ess' in options['metrics']:
        pooled['ess'] = weight_summary(rows)
        pooled['ess_repeat_mean'] = float(np.mean([r['metrics']['ess']['ess_fraction'] for r in repeats]))
        pooled['ess_repeat_sd'] = float(np.std([r['metrics']['ess']['ess_fraction'] for r in repeats], ddof=0))
    posterior = (posterior_report(trees, names, model.env.population_size, options['grid_size'], truth, baselines, options['rank_bins'])
                 if 'posterior_summary' in options['metrics'] else None)
    result = dict(complete=False, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
        step=step, protocol_sha256=protocol_hash, output_dir=str(folder),
        density_bank=str(bank_path) if bank_path else None,
        density_space='full_history' if all(r['unique_backward_history'] for r in fixed_rows+rows) else 'trajectory_balance',
        density_fit=density_summary(fixed_rows) if fixed_rows else None,
        repeats=repeats, pooled=pooled, posterior_summary=posterior,
        created_utc=datetime.now(timezone.utc).isoformat())
    write_report(folder, result, fixed_rows, rows)
    result['artifacts'] = {str(path.relative_to(folder)): digest(path) for path in sorted(folder.rglob('*'))
                           if path.is_file() and path != result_file and not path.name.endswith('.tmp')}
    result['complete'] = True
    write_json(result_file, result)
    print(f'Evaluation complete: {folder}', flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    for flag in ('checkpoint', 'output-dir', 'density-bank', 'device'):
        parser.add_argument('--'+flag)
    for flag in ('num-samples', 'batch-size', 'repeats', 'seed', 'grid-size', 'bank-per-stratum', 'bank-candidates', 'rank-bins'):
        parser.add_argument('--'+flag, type=int)
    parser.add_argument('--metrics', nargs='+', choices=METRICS)
    args = vars(parser.parse_args(argv))
    config = args.pop('config')
    try:
        result = run_evaluation(load_config(config, args))
    except (ValueError, OSError, yaml.YAMLError, tskit.LibraryError) as exc:
        parser.exit(2, f'Evaluation error: {exc}\n')
    print(json.dumps(dict(output_dir=result['output_dir'], complete=result['complete'])))


if __name__ == '__main__':
    main()
