"""Frozen full-history density fitting and independent fresh-policy evaluation.

No training configuration, checkpoint, replay buffer, or held-out set is written.
Run `all` to prepare a frozen protocol/bank, evaluate, and render the report.
Completed batches are atomic and reusable; incomplete batches are never selected.
"""
import argparse
from collections import Counter, defaultdict, deque
from copy import copy
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import pickle
import socket
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
AUDITS = ROOT / 'validation/reports/subtb_trainability_2026-09-10'
sys.path[:0] = [str(ROOT), str(AUDITS)]
import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp
import torch
from audit_importance import DensityAudit, independent_log_likelihood, numpy
from audit_fixed_balance import segment_oracle, state_fingerprint
from env.env import SimpleTrajectory
from flow_training import preserve_sampling
from infer import environment_from_metadata, load_checkpoint
from rollout_worker_arg import RolloutWorker
from tb_gfn import TBGFlowNetGenerator
from eval.posterior_summary import TerminalSamplingEvaluator, topology_signature
from eval.density_fit import fit_stats, density_summary, select_bank
from eval.ess import importance
from train import evaluate_generator
from trajectory_buffer import action_fingerprint
from utils import action_as_dict, action_from_dict

DEFAULT_MANIFEST = AUDITS / 'sim_500/replay_ablation/controlled_variants_manifest.json'
METRICS = ['ess_fraction', 'max_weight', 'log_weight_std', 'log_evidence',
           'eval_truth_pair_tmrca_rmse', 'eval_topology_unique_local_mean',
           'eval_pairwise_local_rf_mean', 'eval_truth_rooted_rf_mean', 'eval_subtb_loss']
ACTIVE_DEADLINE = math.inf


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    if path.suffix == '.gz':
        with gzip.open(temporary, 'wt') as f:
            json.dump(value, f, allow_nan=False, separators=(',', ':'))
    else:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def read_json(path):
    if Path(path).suffix == '.gz':
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    return json.loads(Path(path).read_text())


def progress(**kwargs):
    print(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), **kwargs)), flush=True)


class Deadline(Exception):
    pass


def check_deadline(args):
    if time.time() >= args.stop_at_unix - 60:
        raise Deadline('Stopped before allocation/training deadline; resume the same manifest later.')


def rng_fingerprint(g):
    h = hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state(), g.env.rng.getstate())))
    h.update(torch.random.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            h.update(state.cpu().numpy().tobytes())
    return h.hexdigest(), [m.training for m in g.modules()]


def history_audit(g, state, actions):
    """Recover chronological events from full terminal nodes, never a simplified TS.

    Rebuild each graph prefix with original allocation/order and check its actual
    inverse enumeration. This avoids the inverse helper's different list ordering.
    """
    nodes, active = state.all_nodes, list(range(g.env.num_sequences))
    next_id, previous_time, nonoverlap = g.env.num_sequences, 0., 0
    prefixes = {i: copy(nodes[i]) for i in active}
    for n in prefixes.values():
        n.parents = []
    events = []
    for action in actions:
        node = nodes[next_id]
        count = 1 if node.event_type == 'coal' else 2
        parents = list(range(next_id, next_id + count))
        children = list(node.children)
        assert node.event_type == action.event_type
        assert children[0] == active[action.active_lineage_i]
        if count == 1:
            assert children[1] == active[action.active_lineage_j]
            nonoverlap += nodes[children[0]].material_segments.intersection_count(nodes[children[1]].material_segments) == 0
        else:
            other = nodes[next_id + 1]
            assert other.children == children and other.time == node.time
            assert node.recombination_side == 'left' and other.recombination_side == 'right'
            assert node.breakpoint == other.breakpoint == action.breakpoint
        dt = float(node.time) - previous_time
        assert dt > 0 and math.isclose(dt, action.delta_t, rel_tol=1e-9, abs_tol=2e-13)
        for child in children:
            prefixes[child].parents = parents[:]
        for parent in parents:
            prefixes[parent] = copy(nodes[parent])
            prefixes[parent].parents = []
        active = [i for i in active if i not in children] + parents
        prefix = SimpleNamespace(all_nodes=prefixes, active_lineages=[prefixes[i] for i in active],
                                 max_node_idx=parents[-1], current_time=float(node.time))
        inverse = g._enumerate_inverse_arg_actions(prefix)
        assert len(inverse) == 1 and inverse[0]['event_type'] == action.event_type
        events.append(dict(event_type=node.event_type, child_ids=children, parent_ids=parents,
                           time=float(node.time), delta_t=action.delta_t,
                           breakpoint=getattr(action, 'breakpoint', None), inverse_count=1))
        previous_time, next_id = float(node.time), next_id + count
    assert next_id == state.max_node_idx + 1 == len(nodes)
    assert active == [n.node_id for n in state.active_lineages]
    for i, n in prefixes.items():
        assert n.parents == nodes[i].parents and n.children == nodes[i].children
    assert state.is_done and g.env.is_terminal(state)
    return events, int(nonoverlap)


class CaptureWorker(RolloutWorker):
    def _rollout_batch(self, *args, **kwargs):
        kwargs['return_states'] = True
        self.outputs, self.paths = super()._rollout_batch(*args, **kwargs)
        return self.outputs, self.paths


class TimedDensityAudit(DensityAudit):
    def before(self, module, args):
        if time.time() >= ACTIVE_DEADLINE-60:
            raise Deadline('Evaluation interrupted at a transition before the allocation deadline')
        super().before(module, args)


def load_model(job, args):
    assert digest(job['checkpoint']) == job['sha256'], 'Checkpoint changed'
    c = load_checkpoint(job['checkpoint'], map_location='cpu')
    m = c['metadata']
    assert m['epoch'] + 1 == job['step']
    assert m['arg_prior'] == 'hudson' and m['action_probability_version'] == 2
    assert m['flow_head_version'] == 5 and m['loss_type'] == 'subtb'
    env = environment_from_metadata(m, seed=m['seed'], device=args.device)
    g = TBGFlowNetGenerator(env, 0, device=args.device, verbose=False,
        initialize_z_from_policy=False, model_kwargs=m['model'], loss_type=m['loss_type'],
        subtb_lambda=m['subtb_lambda'], flow_head_version=m['flow_head_version'])
    g.load(c, load_optimizer=False)
    assert not hasattr(g, '_Z')
    protocol = TerminalSamplingEvaluator.from_dataset(args.dataset, env,
        tmrca_method=m.get('tmrca_method', 'grid'))
    return g, protocol


def structure(env, state, protocol):
    ts = env.save_to_tree_sequence(state)
    sigs = [list(topology_signature(ts.at(p), ts.samples())) for p in protocol.positions]
    return dict(topology=sigs, topology_sha256=hashlib.sha256(json.dumps(sigs).encode()).hexdigest(),
                marginal_tree_count=ts.num_trees, terminal_lineage_count=len(state.active_lineages))


def collect(g, protocol, count, seed, fixed=None):
    if fixed is not None:
        fixed = [[action_from_dict(a) for a in path] for path in fixed]
    before = rng_fingerprint(g)
    worker, details = CaptureWorker(g.env), {}
    audit = TimedDensityAudit(g)
    try:
        metrics = evaluate_generator(worker, g, count, seed, fixed_trajectories=fixed,
            terminal_evaluator=protocol if fixed is None else None, terminal_details=details)
    finally:
        audit.remove()
    assert rng_fingerprint(g) == before, 'Evaluation changed RNG or module modes'
    records = audit.reconstruct(worker.outputs, worker.paths)
    for key, error in audit.errors.items():
        tolerance = 1e-7 if key == 'terminal_log_reward' else (1e-4 if 'pf' in key else 1e-8)
        assert error <= tolerance, (key, error)
    outputs = worker.outputs
    for i, (r, state, path) in enumerate(zip(records, outputs['states'], worker.paths)):
        events, nonoverlap = history_audit(g, state, path.actions)
        length = len(path)
        pf, pb, flows = [numpy(outputs[k][i]) for k in ('log_paths_pf', 'log_paths_pb', 'state_flows')]
        _, balance = segment_oracle(pf[:length], pb[:length], flows[:length+1], g.subtb_lambda)
        assert flows[length] == r['log_reward'] or abs(flows[length] - r['log_reward']) < 1e-7
        # Use the exact production reward and scorer; preserve independent scores.
        r.update(independent_log_policy_density=r['log_pf'], independent_log_reward=r['log_reward'],
                 log_policy_density=float(pf[:length].sum()), log_backward_probability=float(pb[:length].sum()),
                 log_reward=float(outputs['log_rewards'][i]), reward_constant=float(g.env.reward_fn.C),
                 event_count=length, nonoverlap_coalescences=nonoverlap, events=events,
                 fingerprint=action_fingerprint(path.actions), balance=balance, **structure(g.env, state, protocol))
        r['actions'] = [action_as_dict(a) for a in path.actions]
        r['log_weight'] = r['log_reward'] + r['log_backward_probability'] - r['log_policy_density']
        if details:
            r['truth_rooted_rf'] = details['per_arg_truth_rooted_rf'][i]
    return dict(records=records, metrics=metrics, audit_errors=dict(audit.errors),
                protocol_sha256=protocol.protocol['sha256'], rng_and_modes_preserved=True)










def prepare_protocol(args):
    path = args.output/'protocol.json'
    if path.exists():
        p = read_json(path)
        assert p['dataset_sha256'] == digest(args.dataset)
        return p
    source = read_json(args.manifest)
    common = None
    for run in source['runs']:
        steps = {int(p.stem.split('_')[-1]) for p in Path(run['path']).glob('checkpoint_*.pt')}
        common = steps if common is None else common & steps
    assert common, 'No equal-update saved checkpoints'
    latest = max(common)
    jobs = [dict(label='shared_initial', step=0, checkpoint=source['initialization_checkpoint'])]
    jobs += [dict(label=r['name'], step=s, checkpoint=str(Path(r['path'])/f'checkpoint_{s:04d}.pt'))
             for s in [latest] + sorted(common-{latest}) for r in source['runs']]
    for j in jobs:
        j['sha256'] = digest(j['checkpoint'])
    initial = load_checkpoint(jobs[0]['checkpoint'], map_location='cpu')['generator_state_dict']
    forbidden, overlap_sources = set(), []
    heldout = {}
    for run in source['runs']:
        root = Path(run['path'])
        c = load_checkpoint(root/'initial.pt', map_location='cpu')['generator_state_dict']
        assert c.keys() == initial.keys() and all(torch.equal(c[k], initial[k]) for k in c)
        hp = root/'heldout_trajectories.json'
        heldout[str(hp)] = digest(hp)
        forbidden.update(action_fingerprint(a) for a in read_json(hp)['actions'])
        # Inspect all currently retained replay snapshots, including an immutable read of latest.
        for cp in sorted(set(root.glob('checkpoint_*.pt')) | {root/'latest.pt'}):
            c = load_checkpoint(cp, map_location='cpu')
            replay = c['metadata'].get('replay_training_state', {}).get('buffer')
            if replay:
                forbidden.update(replay['entries'])
                overlap_sources.append(dict(path=str(cp), update=c['metadata']['epoch']+1,
                                           entries=len(replay['entries'])))
    p = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        input_manifest=str(args.manifest), input_manifest_sha256=digest(args.manifest),
        dataset=str(args.dataset), dataset_sha256=digest(args.dataset), runs=source['runs'],
        latest_common_update=latest, jobs=jobs, repeats=5, episodes_per_repeat=256,
        bank_per_stratum=256, candidates_per_source=512, fixed_batch_size=32,
        fresh_seeds={str(s): [100007+s+1000003*r for r in range(5)] for s in [0, latest]},
        candidate_seed_base=91000000, selection_seed=91919191, heldout_sha256=heldout,
        forbidden_fingerprints=sorted(forbidden), replay_overlap_sources=overlap_sources,
        source_sha256={str(f.relative_to(ROOT)): digest(f) for f in
            [Path(__file__)] + [ROOT/n for n in ['env/env.py', 'env/actions.py', 'env/states.py','utils.py','evo.py','tb_gfn.py','models.py','time_model.py',
              'breakpoint_model.py','rollout_worker_arg.py','subtb.py','terminal_evaluation.py',
              'eval/density_fit.py','eval/ess.py','eval/posterior_summary.py']]
            + [AUDITS/'audit_importance.py', AUDITS/'audit_fixed_balance.py']},
        allocation_end_utc=source['allocation_end_utc'], training_stop_at_unix=source['stop_at_unix'],
        host=socket.gethostname(), environment='phylogfn_orig', device=args.device,
        measure='counting over discrete full histories/links; Lebesgue over waits in generations/(2Ne)',
        time_jacobian='chronological times from waits: 1; generations density: subtract event_count*log(2Ne)',
        selection='rank tertiles of exact reward, fingerprint ties; seeded round robin topology/recombination/event buckets',
        interpretation='bank strata describe constructed coverage, not posterior mass or exhaustive support')
    write_json(path, p)
    return p


def prepare_bank(args, p):
    path = args.output/'bank.json.gz'
    if path.exists():
        bank = read_json(path)
        origin = p.get('bank_origin')
        assert bank['protocol_sha256'] == (origin['protocol_sha256'] if origin else digest(args.output/'protocol.json'))
        if origin:
            assert digest(path) == origin['sha256'] == digest(origin['path'])
        return bank
    candidates = []
    sources = [('hudson_prior', p['jobs'][0])] + [(j['label'], j) for j in p['jobs'][:5]]
    for index, (label, job) in enumerate(sources):
        cache = args.output/'candidates'/f'{label}.json.gz'
        if cache.exists():
            candidates.extend(read_json(cache)['records'])
            continue
        check_deadline(args)
        g, protocol = load_model(job, args)
        before = state_fingerprint(g)
        rows = []
        for batch in range(2):
            check_deadline(args)
            seed = p['candidate_seed_base'] + index*10000 + batch
            batch_path = args.output/'candidates'/f'{label}_{batch}.json.gz'
            if batch_path.exists():
                result = read_json(batch_path)
            elif label == 'hudson_prior':
                # Prior actions are independently sampled, never drawn from replay/training.
                paths = []
                with preserve_sampling(g, seed):
                    for _ in range(256):
                        state, traj = g.env.get_initial_state(), SimpleTrajectory()
                        while not state.is_done:
                            check_deadline(args)
                            action, prior = g.env._sample_prior_step(state)
                            state = g.env.apply_action(state, action, log_prior=prior)
                            traj.update(action, log_prior=prior, log_reward=state.log_reward)
                        paths.append(traj.actions)
                result = collect(g, protocol, 256, seed, paths)
            else:
                result = collect(g, protocol, 256, seed)
            for i, r in enumerate(result['records']):
                r['provenance'] = dict(source=label, seed=seed, batch=batch, index=i,
                    checkpoint_sha256=job['sha256'] if label != 'hudson_prior' else None,
                    sampling='Hudson prior' if label == 'hudson_prior' else 'fresh frozen policy')
            write_json(batch_path, result)
            rows.extend(result['records'])
            progress(phase='candidates', source=label, batch=batch, count=len(rows))
        assert before == state_fingerprint(g) and all(x.grad is None for x in g.parameters())
        write_json(cache, dict(records=rows))
        candidates.extend(rows)
        del g, protocol
        gc.collect()
    bank = select_bank(candidates, p['bank_per_stratum'], set(p['forbidden_fingerprints']))
    bank['protocol_sha256'] = digest(args.output/'protocol.json')
    bank['frozen_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(path, bank)
    write_json(args.output/'bank_manifest.json', {k:v for k,v in bank.items() if k != 'records'})
    progress(phase='bank_frozen', strata=bank['strata'])
    return bank


def evaluate_jobs(args, p, bank):
    for job in p['jobs']:
        folder = args.output/job['label']/f"step_{job['step']:04d}"
        if (folder/'summary.json').exists():
            completed = read_json(folder/'summary.json')
            assert completed['sha256'] == job['sha256'] == digest(job['checkpoint'])
            assert completed['bank_sha256'] == digest(args.output/'bank.json.gz')
            continue
        check_deadline(args)
        g, protocol = load_model(job, args)
        before = state_fingerprint(g)
        fixed_rows, fresh_rows, repeats = [], [], []
        for batch, start in enumerate(range(0, len(bank['records']), 32)):
            cache = folder/f'fixed_{batch:03d}.json.gz'
            chosen = bank['records'][start:start+32]
            if cache.exists():
                result = read_json(cache)
            else:
                check_deadline(args)
                result = collect(g, protocol, len(chosen), 92929292, [r['actions'] for r in chosen])
                for r, origin in zip(result['records'], chosen):
                    assert r['fingerprint'] == origin['fingerprint']
                    assert abs(r['log_reward']-origin['log_reward']) < 1e-7
                    r.update(stratum=origin['stratum'], provenance=origin['provenance'])
                result.update(checkpoint_sha256=job['sha256'], bank_sha256=digest(args.output/'bank.json.gz'))
                write_json(cache, result)
            assert result['checkpoint_sha256'] == job['sha256']
            assert result['bank_sha256'] == digest(args.output/'bank.json.gz')
            assert [r['fingerprint'] for r in result['records']] == [r['fingerprint'] for r in chosen]
            fixed_rows.extend(result['records'])
        progress(phase='fixed_complete', label=job['label'], step=job['step'])
        if str(job['step']) in p['fresh_seeds']:
            for repeat, seed in enumerate(p['fresh_seeds'][str(job['step'])]):
                cache = folder/f'fresh_{repeat:02d}.json.gz'
                if cache.exists():
                    result = read_json(cache)
                else:
                    check_deadline(args)
                    result = collect(g, protocol, 256, seed)
                    for r in result['records']:
                        r['provenance'] = dict(source='fresh policy evaluation', seed=seed, repeat=repeat,
                                               checkpoint_sha256=job['sha256'])
                        key = (r['log_reward'],r['fingerprint'])
                        r['stratum'] = 'low'
                        for name in ('medium','high'):
                            boundary = bank['strata'][name]
                            if key >= (boundary['minimum_log_reward'],boundary.get('minimum_fingerprint','')):
                                r['stratum'] = name
                        r['outside_candidate_reward_range'] = not (
                            bank['strata']['low']['minimum_log_reward'] <= r['log_reward'] <=
                            bank['strata']['high']['maximum_log_reward'])
                    result.update(checkpoint_sha256=job['sha256'], seed=seed, repeat=repeat)
                    write_json(cache, result)
                assert result['checkpoint_sha256'] == job['sha256'] and result['seed'] == seed
                repeats.append(dict(repeat=repeat, seed=seed, **result['metrics'], **importance(result['records'])))
                fresh_rows.extend(result['records'])
                progress(phase='fresh_complete', label=job['label'], step=job['step'], repeat=repeat,
                         ess=repeats[-1]['ess_fraction'])
        assert before == state_fingerprint(g) and not g.opt.state
        assert all(x.grad is None for x in g.parameters()) and digest(job['checkpoint']) == job['sha256']
        summary = dict(**job, fixed=density_summary(fixed_rows),
            fixed_subtb=float(np.mean([r['balance']['loss'] for r in fixed_rows])), repeats=repeats,
            fresh=density_summary(fresh_rows) if fresh_rows else None,
            pooled=importance(fresh_rows) if fresh_rows else None,
            bank_sha256=digest(args.output/'bank.json.gz'),
            parameters_unchanged=True, no_optimizer_updates=True, rng_modes_preserved=True,
            all_histories_reconstructible=True, all_log_pb_zero=True)
        write_json(folder/'summary.json', summary)
        del g, protocol
        gc.collect()


def analytical_reference():
    a, b = 1., 8*(2*10000*1e-7)/3
    evidence = b/(16*(a+b))
    def prior(t): return a*math.exp(-a*t)
    def likelihood(t): return -math.expm1(-b*t)/16
    def posterior(t): return prior(t)*likelihood(t)/evidence
    z, err = quad(lambda t: prior(t)*likelihood(t), 0, np.inf, epsabs=1e-13, epsrel=1e-11)
    norm, _ = quad(posterior, 0, np.inf, epsabs=1e-11)
    tv, _ = quad(lambda t: abs(prior(t)-posterior(t))/2, 0, np.inf, epsabs=1e-10)
    def js_integrand(t):
        q, p = prior(t), posterior(t)
        m = (p+q)/2
        return (p*math.log(p/m) if p else 0)/2 + (q*math.log(q/m) if q else 0)/2
    js, _ = quad(js_integrand, 0, np.inf, epsabs=1e-10)
    estimates = []
    for repeat in range(5):
        rng = np.random.default_rng(93939393+repeat)
        waits = rng.exponential(1/a, 256)
        estimates.append(float(np.mean(-np.expm1(-b*waits)/16)))
    assert abs(z-evidence) < 1e-12 and abs(norm-1) < 1e-10
    return dict(problem='two labelled tips A/C, one JC69 site, no recombination; t in coalescent units',
        prior_rate=a, likelihood_exponent=b, exact_evidence=evidence, exact_log_evidence=math.log(evidence),
        quadrature_evidence=z, quadrature_error=err, posterior_integral=norm,
        prior_vs_posterior_tv=tv, prior_vs_posterior_js=js,
        posterior_oracle_tv=0., posterior_oracle_js=0.,
        prior_importance_evidence_repeats=estimates,
        posterior_mean=1/a+1/(a+b), posterior_variance=1/a**2+1/(a+b)**2,
        interpretation='Restricted analytical calibration only; no training or transfer claim for 500 bp')


def main():
    global ACTIVE_DEADLINE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['all', 'prepare', 'evaluate', 'report', 'extend'])
    parser.add_argument('--base-evaluation', type=Path)
    parser.add_argument('--run', action='append', help='Extension run LABEL=DIRECTORY; repeat for continuations')
    parser.add_argument('--steps', nargs='+', type=int, default=[200], help='Equal-update extension checkpoints')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output', type=Path, default=AUDITS/'sim_500/density_fit')
    parser.add_argument('--dataset', type=Path, default=ROOT/'validation/datasets/sim_500/rep0/sim_500.fa')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stop-at-unix', type=float)
    parser.add_argument('--reuse-plots', action='store_true', help='Reuse present per-checkpoint scatter plots; refresh tables and overview curves')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == 'extend' and args.stop_at_unix is None:
        from launch_subtb_temperature_cosine import allocation_deadline
        _, args.stop_at_unix = allocation_deadline('auto')
    args.stop_at_unix = args.stop_at_unix or read_json(args.manifest)['stop_at_unix']
    ACTIVE_DEADLINE = args.stop_at_unix
    torch.set_num_threads(2)
    if args.command == 'report':
        from subtb_density_report import report
        progress(phase='report_started')
        report(args.output,reuse_plots=args.reuse_plots)
        progress(phase='report_complete')
        return
    try:
        check_deadline(args)
        if args.command == 'extend':
            from subtb_density_extension import prepare_extension
            p, bank = prepare_extension(args)
        else:
            p = prepare_protocol(args)
        write_json(args.output/f'invocation_{time.time_ns()}.json', dict(
            argv=sys.argv, evaluator_sha256=digest(Path(__file__)),
            slurm_job_id=os.environ.get('SLURM_JOB_ID'), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            host=socket.gethostname(), stop_at_unix=args.stop_at_unix, device=args.device,
            protocol_sha256=digest(args.output/'protocol.json')))
        write_json(args.output/'analytical_reference.json', analytical_reference())
        if args.command != 'extend':
            bank = prepare_bank(args, p)
        if args.command in ('all', 'evaluate', 'extend'):
            evaluate_jobs(args, p, bank)
        write_json(args.output/'execution.json', dict(status='complete' if args.command != 'prepare' else 'prepared'))
    except Deadline as exc:
        write_json(args.output/'execution.json', dict(status='deadline', reason=str(exc)))
        progress(phase='deadline', message=str(exc))
    except Exception as exc:
        write_json(args.output/'execution.json', dict(status='failed', error_type=type(exc).__name__, reason=str(exc)))
        raise
    finally:
        if (args.output/'protocol.json').exists():
            from subtb_density_report import report
            progress(phase='report_started')
            report(args.output,reuse_plots=args.reuse_plots)
            progress(phase='report_complete')


if __name__ == '__main__':
    main()
