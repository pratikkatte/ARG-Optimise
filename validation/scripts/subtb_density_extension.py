"""Extend an immutable density-fit bank to additional runs/continuations."""
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil

from evaluate_subtb_density_fit import ROOT, read_json, write_json, digest, load_checkpoint, check_deadline


def run_chain(root):
    """Follow recorded resume origins, without guessing directory-name patterns."""
    chain, seen = [], set()
    while root is not None:
        root = Path(root).resolve()
        if root in seen:
            raise ValueError('Cycle in continuation provenance')
        seen.add(root)
        report = read_json(root/'results.json')
        chain.append(root)
        parent = report.get('options', {}).get('resume')
        root = Path(parent).parent if parent else None
    return list(reversed(chain))


def prepare_extension(args):
    if not args.base_evaluation or not args.run:
        raise ValueError('extend requires --base-evaluation and --run LABEL=DIRECTORY')
    base = args.base_evaluation.resolve()
    if base == args.output.resolve() or base in args.output.resolve().parents:
        raise ValueError('Extension output must be separate from the immutable base evaluation')
    request = dict(base=str(base), runs=args.run, steps=sorted(set(args.steps)))
    if not request['steps'] or any(s <= 0 for s in request['steps']):
        raise ValueError('Extension requires positive comparison update counts')
    destination = args.output/'protocol.json'
    if destination.exists():
        p = read_json(destination)
        if p.get('extension_request') != request:
            raise ValueError('Extension output was frozen for a different request')
        if digest(base/'bank.json.gz') != p['bank_origin']['sha256']:
            raise ValueError('Base bank changed')
        return p, read_json(args.output/'bank.json.gz')
    check_deadline(args)
    old = read_json(base/'protocol.json')
    bank = read_json(base/'bank.json.gz')
    if digest(args.dataset) != old['dataset_sha256']:
        raise ValueError('Extension dataset differs from frozen bank')
    if bank['protocol_sha256'] != old.get('bank_origin', {}).get('protocol_sha256', digest(base/'protocol.json')):
        raise ValueError('Base bank protocol fingerprint differs')
    bindings = {r['name']: [r['path']] for r in old['runs']}
    for specification in args.run:
        label, separator, path = specification.partition('=')
        if not separator or not label or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in label):
            raise ValueError('Run must be a lowercase LABEL=DIRECTORY binding')
        bindings.setdefault(label, []).append(path)
    for roots in bindings.values():
        if args.output.resolve() in [Path(root).resolve() for root in roots]:
            raise ValueError('Evaluation output cannot be a training directory')
    initial = next(j for j in old['jobs'] if j['step'] == 0)
    initial_sha = digest(initial['checkpoint'])
    runs, jobs, forbidden = [], [dict(initial)], set(old['forbidden_fingerprints'])
    heldout, sources = dict(old['heldout_sha256']), []
    for label, roots in bindings.items():
        paths = list(dict.fromkeys(path for root in roots for path in run_chain(root)))
        snapshots = {}
        for root in paths:
            report = read_json(root/'results.json')
            if report.get('initialization_checkpoint_sha256') != initial_sha:
                raise ValueError('Run has a different shared initialization: '+str(root))
            hp = root/'heldout_trajectories.json'
            heldout[str(hp)] = digest(hp)
            from trajectory_buffer import action_fingerprint
            forbidden.update(action_fingerprint(a) for a in read_json(hp)['actions'])
            for cp in sorted(root.glob('checkpoint_*.pt')):
                check_deadline(args)
                step = int(cp.stem.split('_')[-1])
                sha = digest(cp)
                if step in snapshots and snapshots[step]['sha256'] != sha:
                    raise ValueError(f'Ambiguous checkpoint for {label} at {step}')
                snapshots[step] = dict(label=label, step=step, checkpoint=str(cp), sha256=sha)
            # Check all retained buffer snapshots, including the current atomic latest.
            for cp in sorted(set(root.glob('checkpoint_*.pt')) | {root/'latest.pt'}):
                check_deadline(args)
                before = digest(cp)
                saved = load_checkpoint(cp, map_location='cpu')
                if digest(cp) != before:
                    raise ValueError('Checkpoint changed during overlap scan; retry: '+str(cp))
                replay = saved['metadata'].get('replay_training_state', {}).get('buffer')
                if replay:
                    forbidden.update(replay['entries'])
                    sources.append(dict(path=str(cp), sha256=before, entries=len(replay['entries'])))
        for step in request['steps']:
            if step <= 0 or step not in snapshots:
                raise ValueError(f'{label} lacks requested immutable checkpoint {step}')
            jobs.append(snapshots[step])
        runs.append(dict(name=label, path=str(paths[-1]), paths=[str(p) for p in paths]))
    if set(r['fingerprint'] for r in bank['records']) & forbidden:
        raise ValueError('Frozen bank overlaps newly inspected replay/held-out histories; do not silently replace bank')
    p = copy.deepcopy(old)
    p.update(created_utc=datetime.now(timezone.utc).isoformat(), extension_request=request,
             input_manifest=str(base/'protocol.json'), input_manifest_sha256=digest(base/'protocol.json'),
             runs=runs, jobs=jobs, latest_common_update=max(request['steps']),
             fresh_seeds={str(s): [100007+s+1000003*r for r in range(5)] for s in [0]+request['steps']},
             forbidden_fingerprints=sorted(forbidden), heldout_sha256=heldout,
             replay_overlap_sources=old['replay_overlap_sources']+sources,
             training_stop_at_unix=args.stop_at_unix, device=args.device,
             allocation_end_utc=datetime.fromtimestamp(args.stop_at_unix+300, timezone.utc).isoformat(),
             bank_origin=dict(path=str(base/'bank.json.gz'), sha256=digest(base/'bank.json.gz'),
                              protocol_sha256=bank['protocol_sha256']),
             base_source_sha256=old['source_sha256'],
             source_sha256={name: digest(ROOT/name) for name in set(old['source_sha256']) | {
                 'time_env.py', 'policy_temperature_schedule.py', 'learning_rate_schedule.py',
                 'eval/density_fit.py', 'eval/ess.py', 'eval/posterior_summary.py',
                 'validation/scripts/subtb_density_extension.py', 'validation/scripts/subtb_density_report.py'}},
             reused_artifacts=[])
    for job in jobs:
        origin = base/job['label']/f"step_{job['step']:04d}"
        summary = origin/'summary.json'
        if not summary.exists():
            continue
        cached = read_json(summary)
        if cached['sha256'] != job['sha256'] or cached['bank_sha256'] != p['bank_origin']['sha256']:
            continue
        target = args.output/job['label']/f"step_{job['step']:04d}"
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(origin.glob('*.json*')):
            if path.name == 'summary.json' and not cached['fresh']:
                continue  # Reuse fixed batches but collect the five newly requested repeats.
            shutil.copyfile(path, target/path.name)
            p['reused_artifacts'].append(dict(source=str(path), sha256=digest(path),
                                             target=str((target/path.name).relative_to(args.output))))
    for name in ('bank.json.gz', 'bank_manifest.json', 'analytical_reference.json'):
        shutil.copyfile(base/name, args.output/name)
    if (base/'candidates/hudson_prior.json.gz').exists():
        (args.output/'candidates').mkdir(exist_ok=True)
        shutil.copyfile(base/'candidates/hudson_prior.json.gz', args.output/'candidates/hudson_prior.json.gz')
    archive = args.output/'source_archive'
    archive.mkdir(exist_ok=True)
    for name, sha in p['source_sha256'].items():
        shutil.copyfile(ROOT/name, archive/(sha+'_'+Path(name).name))
    write_json(destination, p)
    return p, bank
