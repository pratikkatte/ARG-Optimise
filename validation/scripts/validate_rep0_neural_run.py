#!/usr/bin/env python3
"""Audit initialization/training ARGs and reload a rep0 checkpoint for 16 fresh draws."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import tskit
from generator import GFlowNetGenerator
from training.checkpoints import load_checkpoint, environment_from_metadata, seed_everything
from gfn.rollout import RolloutWorker
from infer import validate_terminal, run_inference
from utils import action_from_dict
from eval.posterior_summary import TerminalSamplingEvaluator


def audit(checkpoint_path, output, dataset_path=None):
    torch.set_num_threads(1)
    data=load_checkpoint(checkpoint_path)
    meta=data['metadata'];trainer=data['trainer']
    if trainer is None or trainer['completed_updates']!=20:
        raise ValueError('The rep0 smoke audit expects exactly 20 updates')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    env=environment_from_metadata(meta)
    replay=trainer['replay']
    if replay is None or len(replay['reservoir']) != replay['seen']:
        raise ValueError('Smoke audit needs the reservoir to retain every generated training history')
    results=[]
    for key in replay['reservoir']:
        entry=replay['entries'][key]
        actions=[action_from_dict(a) for a in json.loads(entry['actions_json'])]
        state=env.replay(actions);reference=validate_terminal(env,state)
        if abs(state.log_reward-entry['log_reward'])>1e-9 or abs(state.accumulated_log_prior-entry['log_prior'])>1e-9:
            raise AssertionError('Stored replay reward or prior changed')
        results.append(dict(key=key,events=len(actions),log_reward=state.log_reward,
                            likelihood_error=abs(reference.log_likelihood-state.partial_log_likelihood)))
    # Reproduce the initial untrained policy stream. No truth or checkpoint
    # trained weights are used to reconstruct these initial draws.
    seed=meta['run_config']['seed'];seed_everything(seed)
    initial_env=environment_from_metadata(meta,seed)
    initial=GFlowNetGenerator(initial_env,model_kwargs=meta['model'],initialize_z_from_policy=False,
                              **meta['generator_config'])
    worker=RolloutWorker(initial_env,max_events=meta['run_config']['max_events'])
    initial_targets=[];initial_errors=[]
    with torch.no_grad():
        for _ in range(initial.init_z_sample_count):
            outputs,_=worker.rollout(initial,return_states=True)
            state=outputs['states'][0];reference=validate_terminal(initial_env,state)
            initial_errors.append(abs(reference.log_likelihood-state.partial_log_likelihood))
            initial_targets.append(float(outputs['log_rewards'][0]-outputs['log_paths_pf'][0].sum()))
    initial_targets=np.array(initial_targets)
    if abs(initial_targets.mean()-data['generator_state_dict']['flow_init_offset'].item())>1e-9:
        raise AssertionError('Initial-policy stream does not reproduce the saved flow center')
    np.testing.assert_allclose(max(1.,initial_targets.std()),data['generator_state_dict']['flow_output_scale'].item(),
                               rtol=1e-12,atol=1e-9)
    manifest=run_inference(checkpoint_path,output/'inference',num_args=16,batch_size=2,seed=100007,
                            max_events=meta['run_config']['max_events'])
    report=dict(passed=True,updates=20,training_histories_checked=len(results),
                initialization_histories_checked=len(initial_errors),
                initialization_max_likelihood_error=max(initial_errors),
                training_max_likelihood_error=max(r['likelihood_error'] for r in results),
                inference=manifest['summary'],training_histories=results)
    if dataset_path:
        evaluator=TerminalSamplingEvaluator.from_dataset(dataset_path,env,grid_size=100,tmrca_method='point_accuracy')
        trees=[tskit.load(output/'inference'/row['trees_file']) for row in manifest['samples']]
        metrics,details=evaluator.summarize_trees(trees)
        report.update(truth_metrics=metrics,truth_protocol=evaluator.protocol)
        (output/'truth_details.json').write_text(json.dumps(details,indent=2,default=lambda x:x.tolist()))
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--dataset-path')
    args=p.parse_args()
    print(json.dumps(audit(args.checkpoint,args.output_dir,args.dataset_path),indent=2))


if __name__=='__main__':
    main()
