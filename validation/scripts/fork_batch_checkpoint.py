"""Fork a checkpoint with an explicitly recorded effective-batch change.

Scientific data, policy/flow parameters, optimizer moments, replay, RNG and
scheduler are retained. The parent file is never modified. This is a new
training experiment, not a claim of exact continuation with the old batch.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from training.checkpoints import load_checkpoint
from training.configuration import resolve_config


def prepare(checkpoint, output, batch_size, grad_accum_steps):
    parent, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory; parent and existing experiments are preserved')
    data = load_checkpoint(parent)
    saved = data['metadata'].get('resolved_config')
    if saved is None or data.get('trainer') is None:
        raise ValueError('A resolved configuration and resumable trainer state are required')
    changes = dict(batch_size=batch_size, grad_accum_steps=grad_accum_steps)
    config = resolve_config({**saved, **changes})
    if not 1 <= grad_accum_steps <= batch_size:
        raise ValueError('grad_accum_steps must be between one and batch_size')
    previous = resolve_config(saved)
    unexpected = {k for k in config if config[k] != previous[k]} - changes.keys()
    if unexpected:
        raise ValueError('Unexpected configuration changes: '+', '.join(sorted(unexpected)))
    provenance = dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        parent_checkpoint=str(parent), parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        parent_completed_updates=data['trainer']['completed_updates'],
        target_fingerprint=data['metadata']['environment_fingerprint'],
        changes={k:dict(before=previous[k],after=config[k]) for k in changes},
        retained=['observations','scientific_rates','model_parameters','optimizer_moments',
                  'replay','random_state','scheduler','completed_update_count'],
        interpretation=__doc__)
    data['metadata'] = {**data['metadata'], 'resolved_config':config,
        'run_config':{**data['metadata'].get('run_config',{}),'batch_size':batch_size},
        'training_fork':provenance}
    output.mkdir(parents=True,exist_ok=True)
    temporary = output/'start.pt.tmp'
    torch.save(data,temporary)
    temporary.replace(output/'start.pt')
    (output/'fork.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return output/'start.pt'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,required=True)
    p.add_argument('--grad-accum-steps',type=int,required=True)
    args = p.parse_args()
    print('Prepared batch-change experiment; no training launched:',
          prepare(args.checkpoint,args.output,args.batch_size,args.grad_accum_steps))


if __name__ == '__main__':
    main()
