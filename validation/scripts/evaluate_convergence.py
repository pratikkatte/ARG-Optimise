"""Fresh importance diagnostics and one stationary history bank across checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from training.checkpoints import load_checkpoint,generator_from_checkpoint,seed_everything
from training.evaluation import evaluate_generator,preserve_sampling
from training.trainer import sample_compatible_trajectories
from training.trajectories import action_fingerprint
from training.reporting import write_json
from utils import action_as_dict,action_from_dict
from gfn.rollout import RolloutWorker
from eval.density_fit import density_summary
from infer import validate_terminal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--steps',nargs='+',type=int,required=True)
    p.add_argument('--samples',type=int,default=512)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--bank-size',type=int,default=96)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--device',default='cuda')
    args=p.parse_args();torch.set_num_threads(1)
    if min(args.samples,args.bank_size,args.batch_size)<1 or args.repeats<0:
        p.error('Sample and batch counts must be positive; repeats may be zero for bank-only checks')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    bank_path=output/'fixed_bank.json'
    comparison_path=output/'comparison.json'
    results=json.loads(comparison_path.read_text()) if comparison_path.exists() else []
    for step in args.steps:
        target=Path(args.run)/'checkpoints'/f'checkpoint_{step:04d}.pt'
        print('loading',target,flush=True)
        checkpoint=load_checkpoint(target)
        g=generator_from_checkpoint(checkpoint,args.device);g.eval()
        if bank_path.exists():
            bank=json.loads(bank_path.read_text())
            if bank['environment_fingerprint']!=g.env.dataset_fingerprint:
                raise ValueError('Fixed bank belongs to a different scientific target')
        else:
            print('sampling independent compatible bank',args.bank_size,flush=True)
            with preserve_sampling(g):
                seed=1700000011;seed_everything(seed);g.env.rng.seed(seed)
                paths=sample_compatible_trajectories(g.env,args.bank_size,10000)
            bank=dict(environment_fingerprint=g.env.dataset_fingerprint,
                      source='independent_compatible_proposal',seed=seed,
                      histories=[[action_as_dict(a) for a in path.actions] for path in paths])
            write_json(bank_path,bank)
        paths=[[action_from_dict(a) for a in history] for history in bank['histories']]
        training_keys=set(((checkpoint.get('trainer') or {}).get('replay') or {}).get('entries',{}))
        if any(action_fingerprint(path) in training_keys for path in paths):
            raise ValueError('Fixed evaluation history overlaps the training replay buffer')
        rows=[]
        print('scoring fixed bank',len(paths),flush=True)
        with torch.no_grad(),preserve_sampling(g):
            worker=RolloutWorker(g.env)
            for start in range(0,len(paths),args.batch_size):
                outputs,scored=worker.replay(g,paths[start:start+args.batch_size],collect_flows=False,return_states=True)
                for i,state in enumerate(outputs['states']):
                    reference=validate_terminal(g.env,state)
                    rows.append(dict(log_reward=state.log_reward,log_likelihood=state.partial_log_likelihood,
                        log_prior=state.accumulated_log_prior,
                        log_policy_density=float(outputs['log_paths_pf'][i].sum()),log_backward_probability=0.,
                        log_policy_factors=dict(zip(('event','lineages','breakpoint','time'),
                            outputs['log_factors'][i].sum(0).tolist())),
                        likelihood_error=abs(reference.log_likelihood-state.partial_log_likelihood),
                        event_count=len(scored[i]),recombinations=sum(a.event_type=='recomb' for a in scored[i].actions)))
                print('scored',len(rows),'/',len(paths),flush=True)
        write_json(output/f'step_{step:06d}_bank.json.gz',dict(records=rows,
                   bank_sha256=hashlib.sha256(bank_path.read_bytes()).hexdigest()))
        fresh=[]
        for repeat in range(args.repeats):
            print('fresh evaluation',repeat+1,'/',args.repeats,flush=True)
            metrics,details=evaluate_generator(g,args.samples,args.batch_size,
                seed=1900000011+repeat,max_events=10000)
            fresh.append(metrics)
            write_json(output/f'step_{step:06d}_fresh_{repeat:02d}.json.gz',dict(metrics=metrics,details=details))
        record=dict(step=step,checkpoint=str(target.resolve()),
                    checkpoint_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    protocol=dict(samples=args.samples,repeats=args.repeats,batch_size=args.batch_size,
                                  fresh_seed=1900000011,bank_seed=bank['seed']),fresh=fresh,
                    fixed_bank=dict(count=len(rows),importance_ess=None,
                        max_likelihood_error=max(r['likelihood_error'] for r in rows),density=density_summary(rows)))
        results=[r for r in results if r['step']!=step]+[record]
        results.sort(key=lambda r:r['step']);write_json(comparison_path,results)
        print('step',step,'ESS',[r['eval_ess'] for r in fresh],
              'log-weight spread',[r['eval_log_weight_std'] for r in fresh],
              'fixed slope',record['fixed_bank']['density']['prior_relative']['global_fit']['slope'],flush=True)


if __name__=='__main__':
    main()
