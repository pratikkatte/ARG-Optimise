"""Configured fresh-policy diagnostics and a separate held-out density bank."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from training.checkpoints import load_checkpoint, generator_from_checkpoint, seed_everything
from training.configuration import validate_evaluation
from training.evaluation import preserve_sampling
from training.trainer import sample_compatible_trajectories
from training.trajectories import DiverseTrajectoryBuffer, action_fingerprint
from infer import collect_samples, sample_summary, resolve_device, validate_terminal
from eval.posterior_summary import TerminalSamplingEvaluator
from eval.density_fit import density_summary, select_bank
from gfn.rollout import RolloutWorker, RolloutFailure
from utils import action_as_dict


def load_model(path, options=None):
    options = options or {}
    data = load_checkpoint(path)
    return generator_from_checkpoint(data,resolve_device(options.get('device','auto')))


def _write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist()))


@torch.no_grad()
def evaluate_density_bank(model, options, checkpoint):
    seed = options.get('seed',100007)+900000
    with preserve_sampling(model):
        seed_everything(seed); model.env.rng.seed(seed)
        paths = sample_compatible_trajectories(model.env,options.get('bank_candidates',768),
                                              options.get('max_events',10000))
        worker = RolloutWorker(model.env,max_events=options.get('max_events',10000))
        catalog = DiverseTrajectoryBuffer(model.env,capacity=2,grid_size=options.get('grid_size',100))
        records = []
        batch = options.get('batch_size',32)
        for start in range(0,len(paths),batch):
            subset = paths[start:start+batch]
            out,_ = worker.replay(model,subset,collect_flows=False,return_states=True)
            for i,(path,state) in enumerate(zip(subset,out['states'])):
                reference = validate_terminal(model.env,state)
                topology = catalog._topology(model.env,state)
                records.append(dict(fingerprint=action_fingerprint(path.actions),
                    topology_sha256=hashlib.sha256(repr(topology).encode()).hexdigest(),
                    provenance=dict(source='compatible_proposal',seed=seed),
                    log_proposal=sum(path.log_proposals),log_prior=state.accumulated_log_prior,
                    log_reward=state.log_reward,log_likelihood=state.partial_log_likelihood,
                    log_policy_density=float(out['log_paths_pf'][i].sum()),log_backward_probability=0.,
                    event_count=len(path),recombinations=sum(a.event_type=='recomb' for a in path.actions),
                    independent_log_likelihood=reference.log_likelihood,
                    actions=[action_as_dict(a) for a in path.actions]))
        replay = (checkpoint.get('trainer') or {}).get('replay') or {}
        bank = select_bank(records,options.get('bank_per_stratum',64),set(replay.get('entries',{})))
        bank.update(source='held_out_compatible_proposal_bank',importance_ess=None,
                    density_fit=density_summary(bank['records']),all_candidates=records)
        return bank


def run_evaluation(options):
    if not options.get('checkpoint') or not options.get('output_dir'):
        raise ValueError('Evaluation requires checkpoint and output_dir')
    metrics = options.get('metrics',['ess','posterior_summary'] if options.get('dataset_path') else ['ess'])
    if set(metrics)-{'density_fit','ess','posterior_summary'}:
        raise ValueError('Unknown evaluation metric')
    if 'posterior_summary' in metrics and not options.get('dataset_path'):
        raise ValueError('posterior_summary requires dataset_path for optional truth evaluation')
    torch.set_num_threads(int(options.get('cpu_threads',1)))
    checkpoint = load_checkpoint(options['checkpoint'])
    model = generator_from_checkpoint(checkpoint,resolve_device(options.get('device','auto')))
    output = Path(options['output_dir'])
    if output.exists() and any(output.iterdir()):
        raise ValueError('Evaluation requires an empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    repeats = options.get('repeats',1)
    if repeats<1:
        raise ValueError('repeats must be positive')
    summaries = []
    try:
        evaluator = (TerminalSamplingEvaluator.from_dataset(options['dataset_path'],model.env,
            grid_size=options.get('grid_size',100),tmrca_method=options.get('tmrca_method','point_accuracy'),
            rank_bins=options.get('rank_bins',20)) if 'posterior_summary' in metrics else None)
        for repeat in range(repeats):
            records, trees, _ = collect_samples(model,options.get('num_samples',16),options.get('batch_size',2),
                        options.get('seed',100007)+repeat,options.get('max_events',10000))
            target = output if repeats==1 else output/f'repeat_{repeat:02d}'
            target.mkdir(exist_ok=True)
            result = dict(environment_fingerprint=model.env.dataset_fingerprint,
                          source='fresh_untempered_policy',posterior_calibration_established=False,
                          repeat=repeat,seed=options.get('seed',100007)+repeat,**sample_summary(records))
            if 'density_fit' in metrics:
                result['fresh_density_fit'] = density_summary(records)
            if evaluator is not None:
                truth_metrics, details = evaluator.summarize_trees(trees)
                result.update(truth_metrics=truth_metrics,truth_protocol=evaluator.protocol)
                _write(target/'truth_details.json',details)
            for i,ts in enumerate(trees):
                ts.dump(target/f'arg_{i:04d}.trees')
            _write(target/'samples.json',records); _write(target/'report.json',result)
            summaries.append(result)
        result = summaries[0] if repeats==1 else dict(repeats=summaries,
            source='fresh_untempered_policy',num_completed=sum(x['num_completed'] for x in summaries))
        if 'density_fit' in metrics:
            bank = evaluate_density_bank(model,options,checkpoint)
            _write(output/'density_bank.json',bank)
            result = {**result,'density_bank':dict(path='density_bank.json',
                     count=len(bank['records']),candidate_count=bank['candidate_count'],
                     source=bank['source'],importance_ess=None,density_fit=bank['density_fit'])}
        _write(output/'report.json',result)
        return result
    except Exception as exc:
        payload = dict(error=str(exc),completed_repeats=summaries)
        if isinstance(exc,RolloutFailure):
            payload.update(histories=exc.histories,completed_samples=getattr(exc,'completed_records',[]))
        _write(output/'failure.json',payload)
        raise


def parse_eval_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    for key in ('checkpoint','output-dir','dataset-path','device','tmrca-method'):
        p.add_argument('--'+key)
    for key in ('num-samples','batch-size','seed','max-events','repeats','grid-size','rank-bins',
                'bank-per-stratum','bank-candidates','cpu-threads'):
        p.add_argument('--'+key,type=int)
    p.add_argument('--metrics',nargs='+',choices=['density_fit','ess','posterior_summary'])
    args=vars(p.parse_args(argv)); config=args.pop('config')
    options={}
    if config:
        root=yaml.safe_load(Path(config).read_text())
        options=validate_evaluation(root.get('evaluation')) or {}
        options.update(dataset_path=root.get('dataset_path'),max_events=root.get('max_events',10000),
                       cpu_threads=root.get('cpu_threads',1),tmrca_method=root.get('tmrca_method','point_accuracy'))
        options['output_dir']=str(Path(root['output_path'])/'posterior_evaluation')
        if options.get('checkpoint')=='best_eval':
            options['checkpoint']=str(Path(root['output_path'])/'checkpoints/best_eval.pt')
    options.update({k:v for k,v in args.items() if v is not None})
    if not options.get('checkpoint') or not options.get('output_dir'):
        p.error('Provide --checkpoint and --output-dir, or a training --config with an evaluation mapping')
    return options


def main(argv=None):
    print(json.dumps(run_evaluation(parse_eval_args(argv)),indent=2))


if __name__ == '__main__':
    main()
