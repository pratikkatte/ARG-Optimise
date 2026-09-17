"""Fresh-policy diagnostics with optional, explicitly requested truth evaluation."""
import argparse
import json
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from training.checkpoints import load_checkpoint, generator_from_checkpoint
from infer import collect_samples, sample_summary, resolve_device
from eval.posterior_summary import TerminalSamplingEvaluator
from gfn.rollout import RolloutFailure


def load_model(path, options=None):
    options = options or {}
    data = load_checkpoint(path)
    return generator_from_checkpoint(data, resolve_device(options.get('device','auto')))


def run_evaluation(options):
    if not options.get('checkpoint') or not options.get('output_dir'):
        raise ValueError('Evaluation requires checkpoint and output_dir')
    torch.set_num_threads(int(options.get('cpu_threads',1)))
    model = load_model(options['checkpoint'], options)
    output = Path(options['output_dir'])
    if output.exists() and any(output.iterdir()):
        raise ValueError('Evaluation requires an empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    try:
        records, trees, _ = collect_samples(model, options.get('num_samples',16), options.get('batch_size',2),
                    options.get('seed',100007), options.get('max_events',10000))
    except RolloutFailure as exc:
        (output/'failure.json').write_text(json.dumps(dict(error=str(exc), histories=exc.histories, completed_samples=getattr(exc,'completed_records',[])), indent=2))
        raise
    result = dict(environment_fingerprint=model.env.dataset_fingerprint,
                  source='fresh_untempered_policy', posterior_calibration_established=False,
                  **sample_summary(records))
    if options.get('dataset_path'):
        evaluator = TerminalSamplingEvaluator.from_dataset(options['dataset_path'],model.env,
                     grid_size=options.get('grid_size',100),tmrca_method='point_accuracy')
        metrics, details = evaluator.summarize_trees(trees)
        result.update(truth_metrics=metrics, truth_protocol=evaluator.protocol)
        (output/'truth_details.json').write_text(json.dumps(details, indent=2, default=lambda x:x.tolist()))
    for i, ts in enumerate(trees):
        ts.dump(output/f'arg_{i:04d}.trees')
    (output/'samples.json').write_text(json.dumps(records, indent=2, allow_nan=False))
    (output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--dataset-path',help='Optional simulator directory, used ONLY for truth evaluation')
    p.add_argument('--num-samples',type=int,default=16);p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--seed',type=int,default=100007);p.add_argument('--device',default='auto')
    p.add_argument('--max-events',type=int,default=10000)
    print(json.dumps(run_evaluation(vars(p.parse_args(argv))),indent=2))


if __name__ == '__main__':
    main()
