"""Prepare a frozen, resumable Slurm training bundle without submitting a job."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.checkpoints import load_checkpoint
from training.configuration import resolve_config


def prepare(args):
    bundle = Path(args.output).resolve()
    source = Path(args.source).resolve()
    if bundle.exists() and any(bundle.iterdir()):
        raise ValueError('Use an empty output bundle; existing experiments are preserved')
    if args.hours < 2 or args.epochs < 1:
        raise ValueError('Use at least two hours and a positive target update count')
    original = Path(args.checkpoint or args.config).resolve()
    if args.checkpoint:
        checkpoint = load_checkpoint(original)
        saved = checkpoint['metadata'].get('resolved_config')
        if saved is None or checkpoint['trainer'] is None:
            raise ValueError('A resolved configuration and resumable trainer state are required')
        if args.epochs <= checkpoint['trainer']['completed_updates']:
            raise ValueError('Target updates must exceed completed checkpoint updates')
        # Pass only mutable execution/evaluation options; train.py restores the
        # exact architecture, objective, optimizer, replay, scheduler and RNG.
        options = dict(dataset_path=saved.get('dataset_path'), resume_checkpoint=str(bundle / 'start.pt'))
    else:
        options = resolve_config(yaml.safe_load(original.read_text()))
        if options.get('resume_checkpoint'):
            raise ValueError('Use --checkpoint to prepare a resumed run')
    dataset = options.get('dataset_path')
    if not dataset:
        raise ValueError('An explicit dataset directory is required')
    dataset = Path(dataset)
    if not dataset.is_absolute():
        dataset = ROOT / dataset
    if not dataset.is_dir():
        raise ValueError('Dataset directory does not exist: ' + str(dataset))
    soft_wall = (args.hours - 1) * 3600
    options.update(dataset_path=str(dataset.resolve()), output_path=str(bundle / 'run'),
        device='cuda', epochs_num=args.epochs, cpu_threads=1, max_wall_seconds=soft_wall,
        checkpoint_every=250, eval_every=250, eval_episodes=256, eval_batch_size=32,
        terminal_eval_repeats=3, terminal_eval_repeat_every=1000)
    if args.wandb:
        options.update(wandb=True, wandb_name=bundle.name)
    directories = ('env', 'eval', 'gfn', 'policy', 'training')
    files = ('train.py', 'generator.py', 'infer.py', 'utils.py', 'breakpoint_model.py')
    if not all((source / name).exists() for name in directories + files):
        raise ValueError('Source directory does not contain the complete training runtime')
    bundle.mkdir(parents=True, exist_ok=True)
    frozen = bundle / 'source'
    frozen.mkdir()
    for directory in directories:
        shutil.copytree(source / directory, frozen / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in files:
        shutil.copy2(source / name, frozen / name)
    if args.checkpoint:
        shutil.copy2(original, bundle / 'start.pt')
    else:
        shutil.copy2(original, bundle / 'requested_config.yaml')
    (bundle / 'config.yaml').write_text(yaml.safe_dump(options, sort_keys=False))
    (bundle / 'logs').mkdir()
    job_name = re.sub(r'[^a-zA-Z0-9_-]', '_', bundle.name)[:80]
    script = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=standard
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=100G
#SBATCH --time={args.hours}:00:00
#SBATCH --signal=USR1@1800
#SBATCH --no-requeue
#SBATCH --output={shlex.quote(str(bundle / 'logs/slurm-%j.out'))}
#SBATCH --error={shlex.quote(str(bundle / 'logs/slurm-%j.err'))}

set -euo pipefail
ARG_BUNDLE={shlex.quote(str(bundle))}
ARG_PYTHON={shlex.quote(sys.executable)}
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ARG_BUNDLE/source"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$ARG_BUNDLE/source"
exec 9> "$ARG_BUNDLE/run.lock"
flock -n 9 || {{ echo 'Another launch owns this run bundle' >&2; exit 1; }}
ARGS=("$ARG_PYTHON" -u "$ARG_BUNDLE/source/train.py")
if [[ $# -eq 1 && "$1" == --resume ]]; then
    test -f "$ARG_BUNDLE/run/checkpoints/latest.pt" || {{
        echo 'No latest.pt checkpoint to resume' >&2; exit 1;
    }}
    ARGS+=(--resume-checkpoint "$ARG_BUNDLE/run/checkpoints/latest.pt"
           --dataset-path {shlex.quote(str(dataset.resolve()))}
           --output-path "$ARG_BUNDLE/run" --device cuda --epochs {args.epochs}
           --max-wall-seconds {soft_wall} --cpu-threads 1)
elif [[ $# -eq 0 ]]; then
    ARGS+=(--config "$ARG_BUNDLE/config.yaml")
else
    echo 'Usage: sbatch submit.sbatch [--resume]' >&2
    exit 2
fi
srun --unbuffered "${{ARGS[@]}}"
'''
    (bundle / 'submit.sbatch').write_text(script)
    manifest = dict(prepared_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        submitted=False, original_input=str(original), source_directory=str(source),
        python=sys.executable, target_updates=args.epochs, allocation_hours=args.hours,
        soft_wall_seconds=soft_wall, initial_checkpoint=bool(args.checkpoint),
        input_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(frozen)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in frozen.rglob('*.py')})
    (bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument('--config')
    origin.add_argument('--checkpoint')
    parser.add_argument('--source', default=str(ROOT))
    parser.add_argument('--output', required=True)
    parser.add_argument('--hours', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--wandb', action='store_true')
    bundle = prepare(parser.parse_args())
    print('Prepared; no job submitted:', bundle)
    print('First allocation:', shlex.join(['sbatch', str(bundle / 'submit.sbatch')]))
    print('Resume allocation:', shlex.join(['sbatch', str(bundle / 'submit.sbatch'), '--resume']))


if __name__ == '__main__':
    main()
