#!/usr/bin/env python3
"""Fixed-budget posterior checks on analytically tractable infinite-sites examples."""
import argparse
import json
import time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scipy.stats import gamma, kstest
from env.env import SimpleARGEnvironment
from env.snp_data import SNPData
from generator import GFlowNetGenerator
from gfn.rollout import RolloutWorker, RolloutFailure
from training.trainer import Trainer, TrajectoryMixConfig
from training.checkpoints import seed_everything
from infer import collect_samples, sample_summary


def fixture(name):
    n = 3 if name=='three_singleton' else 2
    sites = int(name!='two_invariant')
    genotypes=np.zeros((n,sites),dtype=np.uint8)
    if sites:
        genotypes[0,0]=1
    data=SNPData(genotypes,np.array([.5]) if sites else np.array([]),2.,tuple(range(sites)),
                 ('A',)*sites,('C',)*sites,tuple(f'h{i}' for i in range(n)))
    return SimpleARGEnvironment(snp_data=data,population_size=10,mutation_rate=.025,recombination_rate=0)


def validate(name, seed, output, steps=2000, draws=5000):
    started=time.monotonic();seed_everything(seed)
    env=fixture(name)
    model=GFlowNetGenerator(env,init_z_sample_count=8)
    worker=RolloutWorker(env,max_events=10)
    trainer=Trainer(model,worker,TrajectoryMixConfig(replay_fraction=0),seed=seed)
    folder=output/f'{name}_seed{seed}';folder.mkdir()
    with (folder/'training.jsonl').open('w') as handle:
        for step in range(1,steps+1):
            info=trainer.train_epoch(batch_size=32)
            handle.write(json.dumps(info,allow_nan=False)+'\n')
            if step%100==0:
                handle.flush();print(json.dumps(dict(fixture=name,seed=seed,step=step,loss=info['loss'],elapsed=time.monotonic()-started)),flush=True)
    model.save(folder/'model.pt',trainer=trainer,metadata=dict(acceptance_fixture=name,training_seed=seed,updates=steps))
    records,_,paths=collect_samples(model,draws,batch_size=64,seed=100007+seed,max_events=10)
    waits=np.array([[a.delta_t for a in path.actions] for path in paths])
    theta=env.kappa*env.sequence_length
    metrics={}
    if name.startswith('two_'):
        shape=env.num_variants+1;rate=1+2*theta
        expected=shape/rate
        metrics.update(expected_time_mean=expected,observed_time_mean=float(waits[:,0].mean()),
                       time_mean_relative_error=float(abs(waits[:,0].mean()/expected-1)),
                       cdf_error=float(kstest(waits[:,0],gamma(a=shape,scale=1/rate).cdf).statistic))
        passed=metrics['time_mean_relative_error']<.05 and metrics['cdf_error']<.03
    else:
        alpha,beta=3+3*theta,1+2*theta
        expected_probs=np.array([beta,beta,alpha+beta])/(alpha+3*beta)
        pairs=[(0,1),(0,2),(1,2)]
        observed=np.array([sum(tuple(sorted((p.actions[0].active_lineage_i,p.actions[0].active_lineage_j)))==pair
                                  for p in paths)/draws for pair in pairs])
        means=np.array([(6*beta+alpha)/(alpha*(3*beta+alpha)),
                        (3*beta+2*alpha)/(beta*(3*beta+alpha))])
        errors=np.abs(waits.mean(0)/means-1)
        metrics.update(expected_topology_probabilities=expected_probs.tolist(),observed_topology_probabilities=observed.tolist(),
                       topology_probability_error=float(np.max(np.abs(observed-expected_probs))),
                       expected_wait_means=means.tolist(),observed_wait_means=waits.mean(0).tolist(),
                       time_mean_relative_error=float(errors.max()))
        passed=metrics['time_mean_relative_error']<.05 and metrics['topology_probability_error']<.03
    result=dict(fixture=name,seed=seed,steps=steps,draws=draws,passed=bool(passed),
                elapsed_seconds=time.monotonic()-started,metrics=metrics,**sample_summary(records))
    (folder/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[7,17,27])
    p.add_argument('--fixtures',nargs='+',choices=['two_invariant','two_singleton','three_singleton'],
                   default=['two_invariant','two_singleton','three_singleton'])
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--draws',type=int,default=5000)
    args=p.parse_args();torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    results=[]
    for seed in args.seeds:
        for name in args.fixtures:
            try:
                results.append(validate(name,seed,args.output_dir,args.steps,args.draws))
            except RolloutFailure as exc:
                failure=dict(fixture=name,seed=seed,passed=False,error=str(exc),histories=exc.histories)
                results.append(failure)
                print(json.dumps(failure),flush=True)
            (args.output_dir/'report.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    raise SystemExit(0 if all(r['passed'] for r in results) else 1)


if __name__=='__main__':
    main()
