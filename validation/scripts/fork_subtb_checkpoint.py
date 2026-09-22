"""Fork a resumable checkpoint changing only the SubTB geometric weight.

The scientific target, model, optimizer moments, replay, RNG and scheduler stay
unchanged. Provenance distinguishes the objective experiment from exact resume.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from training.checkpoints import load_checkpoint
from training.configuration import resolve_config


def prepare(checkpoint, output, subtb_lambda):
    parent, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory')
    if isinstance(subtb_lambda, bool) or not math.isfinite(subtb_lambda) or subtb_lambda < 0:
        raise ValueError('subtb_lambda must be finite and nonnegative')
    data = load_checkpoint(parent)
    saved = data['metadata'].get('resolved_config')
    if saved is None or data.get('trainer') is None:
        raise ValueError('Resolved configuration and resumable trainer state required')
    previous = resolve_config(saved)
    config = resolve_config({**saved, 'subtb_lambda':float(subtb_lambda)})
    if {k for k in config if config[k] != previous[k]} != {'subtb_lambda'}:
        raise ValueError('Fork must change only subtb_lambda, and must change it')
    assert data['metadata']['generator_config']['subtb_lambda'] == previous['subtb_lambda']
    provenance = dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        parent_checkpoint=str(parent), parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        parent_completed_updates=data['trainer']['completed_updates'],
        target_fingerprint=data['metadata']['environment_fingerprint'],
        changes=dict(subtb_lambda=dict(before=previous['subtb_lambda'], after=config['subtb_lambda'])),
        parent_training_fork=data['metadata'].get('training_fork'),
        retained=['observations','scientific_rates','model_parameters','optimizer_moments',
                  'replay','random_state','scheduler','completed_update_count','tb_loss_weight','batch_size'],
        interpretation=__doc__)
    data['metadata'] = {**data['metadata'], 'resolved_config':config,
        'generator_config':{**data['metadata']['generator_config'], 'subtb_lambda':config['subtb_lambda']},
        'training_fork':provenance}
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / 'start.pt.tmp'
    torch.save(data, temporary)
    temporary.replace(output / 'start.pt')
    (output / 'fork.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return output / 'start.pt'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--subtb-lambda', type=float, required=True)
    args = parser.parse_args()
    print(prepare(args.checkpoint, args.output, args.subtb_lambda))
