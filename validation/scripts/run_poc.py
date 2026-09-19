#!/usr/bin/env python3
"""Train SubTB on the tiny recombining dataset and measure full-history TV.

Reference posterior samples and the exact CTMC evidence are evaluation-only.
All training updates use the normal repository trainer and physical reward.
"""
import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from env.env import SimpleARGEnvironment
from env.snp_data import load_snp_dataset
from generator import GFlowNetGenerator
from gfn.rollout import RolloutWorker
from training.trainer import Trainer, TrajectoryMixConfig
from training.schedules import LearningRateConfig, WarmupCosineScheduler
from training.checkpoints import (seed_everything, rng_state, restore_rng, load_checkpoint,
                                  generator_from_checkpoint)
from utils import action_as_dict
from validation.poc_dataset import generate_poc, load_poc_config
from validation.poc_reference import exact_evidence, rejection_sample, sample_prior, trajectory
from validation.poc_metrics import distribution_metrics


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def write_gzip(path, value):
    with gzip.open(path, 'wt') as handle:
        json.dump(value, handle, allow_nan=False)


def read_gzip(path):
    with gzip.open(path, 'rt') as handle:
        return json.load(handle)


def reconcile_training_log(directory, completed_updates):
    """Preserve interrupted suffixes before resuming from the saved optimizer.

    A process can write updates after its last checkpoint. Those updates are
    not part of a resumed model's history and must not become duplicate rows.
    """
    path=directory/'training.jsonl'
    if not path.exists():
        if completed_updates:
            raise ValueError('Checkpoint exists but its training log is missing')
        return
    original=path.read_text()
    lines=original.splitlines(keepends=True)
    kept=[]
    for line in lines:
        try:
            row=json.loads(line)
        except json.JSONDecodeError:
            break
        if row['step']>completed_updates:
            break
        if row['step']!=len(kept)+1:
            raise ValueError('Training log is inconsistent before the checkpoint')
        kept.append(line)
    if len(kept)!=completed_updates:
        raise ValueError('Training log does not cover the saved checkpoint')
    retained=''.join(kept)
    if retained!=original:
        recovery=directory/'interrupted_logs'
        recovery.mkdir(exist_ok=True)
        destination=recovery/f'training_before_resume_{time.time_ns()}.jsonl'
        destination.write_text(original)
        path.write_text(retained)


def source_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('env','policy','gfn','training','validation') for p in sorted((ROOT/folder).rglob('*.py'))} | {
            'generator.py': hashlib.sha256((ROOT/'generator.py').read_bytes()).hexdigest()}


def prepare_reference(config, env, output):
    e = config['evaluation']; d = config['dataset']
    directory = output/'reference'; directory.mkdir(exist_ok=True)
    path = directory/'posterior.json.gz'
    protocol = dict(dataset=d, count=e['reference_samples'], seed=e['reference_seed'])
    if path.exists():
        if json.loads((directory/'protocol.json').read_text()) != protocol:
            raise ValueError('Reference protocol changed; use a new output_dir')
        records = read_gzip(path)
        if len(records) != e['reference_samples']:
            raise ValueError('Reference bank is incomplete')
    else:
        print('Generating independent exact posterior reference', flush=True)
        records, stats = rejection_sample(e['reference_samples'], e['reference_seed'], 2,
                                          env.kappa, 2*env.population_size*env.recombination_rate)
        write_gzip(path, records)
        write_json(directory/'sampling.json', stats)
        write_json(directory/'protocol.json', protocol)
    evidence = exact_evidence(2, env.kappa, 2*env.population_size*env.recombination_rate)
    max_prior_error = max_likelihood_error = 0.
    for record in records[:e['independent_checks']]:
        state = env.replay(trajectory(record).actions)
        max_prior_error = max(max_prior_error, abs(state.accumulated_log_prior-record['log_prior']))
        max_likelihood_error = max(max_likelihood_error, abs(env.evaluate_terminal(state).log_likelihood-record['log_likelihood']))
    if max(max_prior_error,max_likelihood_error) > 1e-9:
        raise AssertionError('Independent reference and environment disagree')
    evidence.update(max_prior_error=max_prior_error, max_likelihood_error=max_likelihood_error,
                    expected_rejection_acceptance=evidence['evidence']*4*math.e**2)
    write_json(directory/'evidence.json', evidence)
    baseline_path=directory/'prior_metrics.json'
    if not baseline_path.exists():
        rng=np.random.default_rng(e['reference_seed']+1)
        prior=[sample_prior(rng,2,env.kappa,2*env.population_size*env.recombination_rate).as_dict()
               for _ in range(e['reference_samples'])]
        baseline=distribution_metrics(records,prior,[r['log_prior'] for r in records],
                                      [r['log_prior'] for r in prior],evidence['log_evidence'])
        baseline['sampling_distribution']='physical prior'
        write_json(baseline_path,baseline)
    return records, evidence


@torch.no_grad()
def score_reference(model, records, batch_size, max_events):
    worker=RolloutWorker(model.env, max_events=max_events)
    scores=[]
    for start in range(0,len(records),batch_size):
        paths=[trajectory(r) for r in records[start:start+batch_size]]
        outputs,_=worker.replay(model,paths,collect_flows=False)
        scores.extend(outputs['log_paths_pf'].sum(1).tolist())
    return scores


@torch.no_grad()
def policy_samples(model, count, batch_size, seed, max_events, independent_checks):
    seed_everything(seed)
    worker=RolloutWorker(model.env,max_events=max_events)
    records=[]; scores=[]; max_error=0.
    for start in range(0,count,batch_size):
        outputs,paths=worker.rollout(model,min(batch_size,count-start),return_states=True)
        scores.extend(outputs['log_paths_pf'].sum(1).tolist())
        for state,path in zip(outputs['states'],paths):
            # In the two-tip case each observed singleton branch equals local TMRCA.
            tmrca=state.completed_site_lengths.copy()
            independent_formula=2*math.log(model.env.kappa)+float(np.log(tmrca).sum())-2*model.env.kappa*float(tmrca.sum())
            max_error=max(max_error,abs(state.partial_log_likelihood-independent_formula))
            if len(records)<independent_checks:
                max_error=max(max_error,abs(state.partial_log_likelihood-model.env.evaluate_terminal(state).log_likelihood))
            records.append(dict(actions=[action_as_dict(a) for a in path.actions],
                log_prior=state.accumulated_log_prior,log_likelihood=state.partial_log_likelihood,
                tmrca=tmrca.tolist(),recombinations=sum(a.event_type=='recomb' for a in path.actions)))
    if max_error>1e-9:
        raise AssertionError('Generated history likelihood disagrees with independent calculation')
    return records,scores,max_error


@torch.no_grad()
def evaluate(model, config, reference, evidence, directory, step, final=False):
    e=config['evaluation']; t=config['training']
    count=e['final_samples'] if final else e['progress_samples']
    if count>len(reference):
        raise ValueError('Reference sample count must cover each evaluation')
    state=rng_state(model.env); was_training=model.training
    try:
        model.eval()
        bank=reference[:count]
        q_reference=score_reference(model,bank,e['batch_size'],t['max_events'])
        samples,q_policy,max_error=policy_samples(model,count,e['batch_size'],
            e['policy_seed']+t['seed']+step,t['max_events'],e['independent_checks'])
        result=distribution_metrics(bank,samples,q_reference,q_policy,evidence['log_evidence'])
        result.update(step=step,training_seed=t['seed'],time_head=config['model']['continuous_time_head'],
                      final=final,max_likelihood_error=max_error,
                      learned_log_evidence=float(model.compute_log_Z())-model.env.reward_fn.C,
                      reference_log_evidence=evidence['log_evidence'],
                      evaluation_target='full chronological timed ARG histories; unweighted temperature-one policy')
        result['tv_target']=e['tv_target']
        result['meets_tv_target']=result['full_history_tv']['ci95'][1]<e['tv_target']
        write_json(directory/f'metrics_{step:06d}.json',result)
        if final:
            write_gzip(directory/'final_policy_samples.json.gz',samples)
            write_gzip(directory/'final_density_scores.json.gz',dict(reference_log_q=q_reference,policy_log_q=q_policy,
                policy_log_p=[r['log_prior']+r['log_likelihood']-evidence['log_evidence'] for r in samples],
                reference_log_p=[r['log_prior']+r['log_likelihood']-evidence['log_evidence'] for r in bank]))
            write_json(directory/'result.json',result)
            for i,record in enumerate(samples[:4]):
                model.env.save_to_tree_sequence(model.env.replay(trajectory(record).actions),directory/f'policy_example_{i}.trees')
        return result
    finally:
        restore_rng(model.env,state)
        model.train(was_training)


def plot_run(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curve=sorted([json.loads(p.read_text()) for p in directory.glob('metrics_*.json')],key=lambda r:r['step'])
    if not curve:
        return
    last=curve[-1]
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),constrained_layout=True)
    x=[r['step'] for r in curve];y=[r['full_history_tv']['estimate'] for r in curve]
    se=[1.96*r['full_history_tv']['standard_error'] for r in curve]
    axes[0].errorbar(x,y,yerr=se,marker='o',capsize=3)
    axes[0].axhline(last['tv_target'],color='grey',linestyle='--',label='POC target')
    axes[0].set(xlabel='Training updates',ylabel='Full-history TV',ylim=(0,None));axes[0].legend()
    counts=last['recombination_count'];xs=np.array(counts['counts'])
    axes[1].bar(xs-.18,counts['reference_probabilities'],width=.36,label='Exact reference')
    axes[1].bar(xs+.18,counts['policy_probabilities'],width=.36,label='GFlowNet')
    axes[1].set(xlabel='Recombination count',ylabel='Probability');axes[1].legend()
    scores_path=directory/'final_policy_samples.json.gz'
    if scores_path.exists():
        samples=read_gzip(scores_path)
        root=directory.parent.parent
        reference=read_gzip(root/'reference/posterior.json.gz')[:len(samples)]
        for rows,label in ((reference,'Exact reference'),(samples,'GFlowNet')):
            times=np.sort([r['tmrca'][0] for r in rows])
            axes[2].plot(times,np.arange(1,len(times)+1)/len(times),label=label)
        axes[2].set(xlabel='Left-locus TMRCA (2Ne)',ylabel='CDF');axes[2].legend()
    for extension in ('png','pdf'):
        fig.savefig(directory/f'posterior_comparison.{extension}',dpi=180)
    plt.close(fig)
    with (directory/'convergence.csv').open('w') as handle:
        writer=csv.writer(handle);writer.writerow(['step','full_history_tv','tv_standard_error','squared_hellinger','forward_kl_nats','ess_fraction'])
        for r in curve:
            writer.writerow([r['step'],r['full_history_tv']['estimate'],r['full_history_tv']['standard_error'],
                r['squared_hellinger']['estimate'],r['forward_kl_nats']['estimate'],r['importance_ess_fraction']])


def run_seed(config, env, reference, evidence, directory, resume=False):
    directory.mkdir(parents=True,exist_ok=True)
    config_path=directory/'resolved_config.yaml'
    if config_path.exists():
        if yaml.safe_load(config_path.read_text())!=config:
            raise ValueError('Existing run configuration differs; choose another --tag')
        if not resume:
            raise ValueError('Existing run requires --resume or a new --tag')
    config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    t=config['training'];seed_everything(t['seed']);env.rng.seed(t['seed'])
    last_path=directory/'checkpoints/last.pt'
    if resume and last_path.exists():
        saved=load_checkpoint(last_path)
        model=generator_from_checkpoint(saved,t['device'],optimizer=True,restore_random=True)
        worker=RolloutWorker(model.env,max_events=t['max_events'])
        trainer=Trainer(model,worker,TrajectoryMixConfig(**{k:t[k] for k in TrajectoryMixConfig.__dataclass_fields__}),seed=t['seed'])
        trainer.load_state_dict(saved['trainer'])
    else:
        model=GFlowNetGenerator(env,device=t['device'],model_kwargs=config['model'],
            init_z_sample_count=t['init_z_sample_count'],initialize_z_from_policy=False,
            policy_lr=t['policy_lr'],flow_lr=t['flow_lr'],grad_clip=t['grad_clip'],subtb_lambda=t['subtb_lambda'])
        model.initialize_flow_center(batch_size=t['batch_size'])
        model.scheduler=WarmupCosineScheduler(model.opt,LearningRateConfig('cosine',t['steps'],min_factor=t['lr_min_factor']))
        trainer=Trainer(model,RolloutWorker(env,max_events=t['max_events']),
            TrajectoryMixConfig(**{k:t[k] for k in TrajectoryMixConfig.__dataclass_fields__}),seed=t['seed'])
    write_json(directory/'provenance.json',dict(source_hashes=source_hashes(),torch=torch.__version__,numpy=np.__version__,
        device=str(model.device),device_name=torch.cuda.get_device_name(model.device) if model.device.type=='cuda' else 'CPU',
        reference_used_for_training=False,checkpoint_selection='fixed final update; no reference-metric selection'))
    reconcile_training_log(directory,trainer.completed_updates)
    if trainer.completed_updates==0:
        metrics=evaluate(model,config,reference,evidence,directory,0)
        print(json.dumps(dict(seed=t['seed'],step=0,tv=metrics['full_history_tv'])),flush=True)
    start=time.monotonic()
    with (directory/'training.jsonl').open('a') as handle:
        for step in range(trainer.completed_updates+1,t['steps']+1):
            model.train()
            row=trainer.train_epoch(batch_size=t['batch_size'])
            row['elapsed_seconds']=time.monotonic()-start
            handle.write(json.dumps(row,allow_nan=False)+'\n')
            if step%25==0:
                handle.flush()
                print(json.dumps(dict(seed=t['seed'],step=step,loss=row['loss'],elapsed=row['elapsed_seconds'])),flush=True)
            if step%t.get('checkpoint_every',config['evaluation']['every'])==0 or step==t['steps']:
                model.save(last_path,trainer=trainer,metadata=dict(poc_config=config))
            if step%config['evaluation']['every']==0 or step==t['steps']:
                model.save(last_path,trainer=trainer,metadata=dict(poc_config=config))
                metrics=evaluate(model,config,reference,evidence,directory,step,final=step==t['steps'])
                print(json.dumps(dict(seed=t['seed'],step=step,tv=metrics['full_history_tv'],ess=metrics['importance_ess_fraction'])),flush=True)
                plot_run(directory)
    if not (directory/'result.json').exists():
        evaluate(model,config,reference,evidence,directory,t['steps'],final=True)
        plot_run(directory)
    return json.loads((directory/'result.json').read_text())


def summarize(output):
    results=[]
    for path in sorted((output/'runs').glob('*/result.json')):
        row=json.loads(path.read_text());row['run']=str(path.parent.relative_to(output));results.append(row)
    write_json(output/'results.json',results)
    lines=['# Tiny ARG posterior recovery','',
        'Two haplotypes; two one-base loci; two polarized singleton SNPs; nonzero recombination. ',
        'TV compares full timed histories. The reference uses independent rejection sampling and ',
        'a finite-state CTMC normalizer that includes arbitrarily many recombination events. ',
        'Intervals below quantify Monte Carlo error for each fixed trained policy, not variation across training seeds.','',
        '| Run | Updates | TV (95% MC interval) | Hellinger² | ESS/N | Meets configured TV target |',
        '|---|---:|---|---:|---:|---|']
    for r in results:
        tv=r['full_history_tv'];lo,hi=tv['ci95']
        lines.append(f"| [{r['run']}]({r['run']}/posterior_comparison.pdf) | {r['step']} | {tv['estimate']:.5f} [{lo:.5f}, {hi:.5f}] | {r['squared_hellinger']['estimate']:.6f} | {r['importance_ess_fraction']:.4f} | {r['meets_tv_target']} |")
    lines += ['', 'All completed runs are listed, including pilots and failed accuracy gates. ',
        'Low marginal errors alone are not used to declare posterior recovery. ',
        'This two-tip example does not test local topology diversity or uncertainty in breakpoint position.','']
    (output/'RESULTS.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'validation/config/poc.yaml')
    p.add_argument('--steps',type=int);p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--head',choices=['gamma','gamma_mixture'])
    p.add_argument('--tag',default='main');p.add_argument('--resume',action='store_true')
    p.add_argument('--final-samples',type=int);p.add_argument('--device',choices=['cpu','cuda'])
    p.add_argument('--reference-only',action='store_true')
    args=p.parse_args();config=load_poc_config(args.config)
    if args.steps is not None:config['training']['steps']=args.steps
    if args.seeds is not None:config['training']['seeds']=args.seeds
    if args.device:config['training']['device']=args.device
    if args.head:config['model']['continuous_time_head']=args.head
    if args.final_samples is not None:config['evaluation']['final_samples']=args.final_samples
    torch.set_num_threads(config['training']['cpu_threads'])
    if config['training']['device']=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('POC requests CUDA; run with A100 access or explicitly pass --device cpu')
    output=ROOT/config['output_dir'];output.mkdir(parents=True,exist_ok=True)
    directory,_,_=generate_poc(config)
    d=config['dataset']
    env=SimpleARGEnvironment(snp_data=load_snp_dataset(directory),population_size=d['population_size'],
        mutation_rate=d['mutation_rate'],recombination_rate=d['recombination_rate'])
    reference,evidence=prepare_reference(config,env,output)
    print(json.dumps(evidence),flush=True)
    if args.reference_only:return
    for seed in config['training']['seeds']:
        run_config=copy.deepcopy(config);run_config['training']['seed']=seed
        run_dir=output/'runs'/f'{args.tag}_{config["model"]["continuous_time_head"]}_seed{seed}'
        try:
            run_seed(run_config,env,reference,evidence,run_dir,args.resume)
        except Exception as exc:
            write_json(run_dir/'failure.json',dict(error=str(exc),type=type(exc).__name__,
                                                  histories=getattr(exc,'histories',None)))
            raise
        finally:
            summarize(output)


if __name__=='__main__':
    main()
