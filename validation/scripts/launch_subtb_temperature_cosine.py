"""Launch only the fifth 500 bp variant inside an existing GPU allocation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT/'config_learned_event_500_temperature_cosine.yaml'
INITIAL = ROOT/'runs/sim_500_subtb_shared_initialization/initial.pt'
BENCHMARK = ROOT/'validation/reports/subtb_trainability_2026-09-10/benchmark.py'


def allocation_deadline(value, *, now=None, environ=None, query=subprocess.check_output):
    environ = os.environ if environ is None else environ
    now = time.time() if now is None else now
    if value == 'auto':
        job = environ.get('SLURM_JOB_ID')
        if not job:
            raise ValueError('No SLURM_JOB_ID; supply --allocation-end with an explicit ISO timestamp')
        line = query(['scontrol', 'show', 'job', '-o', job], text=True)
        match = re.search(r'\bEndTime=(\S+)', line)
        if match is None:
            raise ValueError('Slurm did not report an allocation EndTime')
        value = match.group(1)
    try:
        end = datetime.fromisoformat(value).astimezone(timezone.utc)
    except ValueError as exc:
        raise ValueError('Allocation end must be a finite ISO timestamp') from exc
    stop = end.timestamp() - 300
    if stop <= now + 300:
        raise ValueError('Need more than ten minutes remaining for startup and checkpoint margin')
    return end.isoformat(), stop


def preflight(output, resume, allocation_end):
    import torch
    from train import parse_train_args
    from infer import load_checkpoint
    from training.schedules import LearningRateConfig, PolicyTemperatureConfig

    if output.exists():
        raise ValueError('Output already exists; choose a new directory to preserve the run')
    end, stop = allocation_deadline(allocation_end)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment/allocation')
    baseline = parse_train_args(['--config', str(ROOT/'config_learned_event_500.yaml')])
    args = parse_train_args(['--config', str(CONFIG)])
    allowed = {'config', 'output_path', 'policy_temperature_schedule', 'policy_temperature_start',
               'policy_temperature_anneal_steps', 'lr_schedule', 'lr_schedule_steps',
               'lr_warmup_steps', 'lr_min_factor'}
    differences = {k for k, v in vars(args).items() if v != getattr(baseline, k)}
    if differences - allowed:
        raise ValueError('Fifth variant changes other baseline settings: '+str(sorted(differences - allowed)))
    expected = dict(epochs=10000, batch_size=128, seed=7, flow_head_version=5, subtb_lambda=2.,
                    policy_lr=1e-4, flow_lr=1e-3, exploration_fraction=0., replay_fraction=0.,
                    policy_temperature_schedule='linear', policy_temperature_start=1.5,
                    policy_temperature_anneal_steps=2000, lr_schedule='cosine',
                    lr_schedule_steps=10000, lr_warmup_steps=0, lr_min_factor=.1)
    for key, value in expected.items():
        if getattr(args, key) != value:
            raise ValueError('Variant protocol differs in '+key)
    origin = resume.resolve() if resume else INITIAL
    saved = load_checkpoint(origin, map_location='cpu')
    meta = saved['metadata']
    step = meta['epoch'] + 1
    if '_Z' in saved['generator_state_dict'] or meta['flow_head_version'] != 5:
        raise ValueError('Fifth variant requires the shared neural source flow with no trainable logZ')
    if meta['sequence_length'] != 500 or meta['action_probability_version'] != 2:
        raise ValueError('Checkpoint is not the corrected 500 bp experiment')
    if 'sampling_state' not in meta or not (origin.parent/'heldout_trajectories.json').is_file():
        raise ValueError('Checkpoint must retain training RNG and the original held-out set')
    if resume:
        PolicyTemperatureConfig.from_namespace(args).validate_resume(meta.get('policy_temperature_state'), step)
        state = saved.get('lr_scheduler_state_dict')
        if (state is None or state['config'] != LearningRateConfig.from_namespace(args).__dict__
                or state['completed_updates'] != step or state['base_lrs'] != [args.policy_lr, args.flow_lr]):
            raise ValueError('Checkpoint cosine schedule differs from fifth variant')
        if meta.get('initialization_checkpoint_sha256') != hashlib.sha256(INITIAL.read_bytes()).hexdigest():
            raise ValueError('Resume checkpoint does not descend from shared initialization')
    elif step != 0 or saved['opt_state_dict']['state']:
        raise ValueError('Fresh variant must start at the shared update-zero initialization')
    if step >= args.epochs:
        raise ValueError('The requested experiment has already reached 10,000 updates')
    command = [sys.executable, '-u', str(BENCHMARK), '--config', str(CONFIG),
               '--resume' if resume else '--initialize-from', str(origin), '--updates', str(args.epochs),
               '--fixed-episodes', '128', '--checkpoint-every', '5', '--device', 'cuda',
               '--output', str(output), '--stop-at-unix', str(stop)]
    return dict(command=command, hostname=socket.gethostname(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                allocation_end_utc=end, stop_at_unix=stop, origin=str(origin), start_step=step,
                origin_sha256=hashlib.sha256(origin.read_bytes()).hexdigest(),
                config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                dataset_sha256=hashlib.sha256((ROOT/args.dataset_path).read_bytes()).hexdigest(),
                gpu=torch.cuda.get_device_name(0), python=sys.executable,
                label='baseline_temperature_cosine', evaluation_temperature=1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--allocation-end', default='auto', help='auto from Slurm, or ISO allocation end timestamp')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    record = preflight(output, args.resume, args.allocation_end)
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, **record), indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    (output/'launcher_source.py').write_bytes(Path(__file__).read_bytes())
    with (output/'training.log').open('xb') as log:
        process = subprocess.Popen(record['command'], cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
            env={**os.environ, 'OMP_NUM_THREADS': '2', 'MPLCONFIGDIR': '/tmp/argopt_mplcache'})
    record.update(pid=process.pid, launched_at=datetime.now(timezone.utc).isoformat(), detached=True,
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    (output/'launch.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
