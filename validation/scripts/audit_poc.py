#!/usr/bin/env python3
"""Verify saved posterior POC artifacts and independently reproduce metrics.

An accuracy gate is reported separately from artifact correctness. Failed
accuracy gates remain valid scientific results and are never hidden.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
import yaml

from gfn.rollout import RolloutWorker
from training.checkpoints import load_checkpoint, generator_from_checkpoint
from validation.poc_metrics import distribution_metrics
from validation.poc_reference import exact_evidence, trajectory


def load_gzip(path):
    with gzip.open(path,'rt') as handle:
        return json.load(handle)


def compare(actual, expected, path='metrics'):
    if isinstance(actual,dict):
        for key,value in actual.items():
            compare(value,expected[key],path+'.'+key)
    elif isinstance(actual,list):
        if len(actual)!=len(expected):
            raise AssertionError(path+': mismatched length')
        for i,(x,y) in enumerate(zip(actual,expected)):
            compare(x,y,path+f'[{i}]')
    elif isinstance(actual,(int,float)):
        if not np.isclose(actual,expected,rtol=1e-10,atol=1e-12):
            raise AssertionError(path+': saved metric disagrees with recomputation')
    elif actual!=expected:
        raise AssertionError(path+': saved value differs')


@torch.no_grad()
def audit_run(directory, score_count=128):
    config=yaml.safe_load((directory/'resolved_config.yaml').read_text())
    result=json.loads((directory/'result.json').read_text())
    root=directory.parent.parent
    samples=load_gzip(directory/'final_policy_samples.json.gz')
    scores=load_gzip(directory/'final_density_scores.json.gz')
    count=config['evaluation']['final_samples']
    reference=load_gzip(root/'reference/posterior.json.gz')[:count]
    if not len(samples)==len(reference)==count or not result['final']:
        raise AssertionError('Final sample set is missing or incomplete')
    scientific=config['dataset']
    evidence=exact_evidence(2,2*scientific['population_size']*scientific['mutation_rate'],
                           2*scientific['population_size']*scientific['recombination_rate'])
    compare(evidence['log_evidence'],result['reference_log_evidence'],'log_evidence')
    recalculated=distribution_metrics(reference,samples,scores['reference_log_q'],
                                     scores['policy_log_q'],evidence['log_evidence'])
    compare(recalculated,result)
    for rows,key in ((reference,'reference_log_p'),(samples,'policy_log_p')):
        expected=[r['log_prior']+r['log_likelihood']-evidence['log_evidence'] for r in rows]
        np.testing.assert_allclose(scores[key],expected,rtol=0,atol=1e-12)
    checkpoint=load_checkpoint(directory/'checkpoints/last.pt')
    if not checkpoint['trainer']['completed_updates']==result['step']==config['training']['steps']:
        raise AssertionError('Reported step is not the saved fixed-budget final checkpoint')
    logged=[json.loads(line) for line in (directory/'training.jsonl').read_text().splitlines()]
    if [r['step'] for r in logged]!=list(range(1,result['step']+1)):
        raise AssertionError('Training log has missing or duplicate updates')
    model=generator_from_checkpoint(checkpoint,device='cpu')
    model.eval()
    if model.model_kwargs['continuous_time_head']!=config['model']['continuous_time_head']:
        raise AssertionError('Checkpoint has the wrong time head')
    if model.model_kwargs['continuous_time_head']=='gamma_mixture':
        if model.arg_model.time_head.components!=config['model']['time_mixture_components']:
            raise AssertionError('Checkpoint mixture count is incorrect')
    rng=np.random.default_rng(490001)
    indexes=np.sort(rng.choice(count,size=min(score_count,count),replace=False))
    worker=RolloutWorker(model.env,max_events=config['training']['max_events'])
    max_policy_error=max_prior_error=max_likelihood_error=0.
    for rows,log_q in ((samples,scores['policy_log_q']),(reference,scores['reference_log_q'])):
        selected=[rows[i] for i in indexes]
        out,_=worker.replay(model,[trajectory(r) for r in selected],collect_flows=False,return_states=True)
        actual=out['log_paths_pf'].sum(1).numpy()
        expected=np.asarray(log_q)[indexes]
        max_policy_error=max(max_policy_error,float(np.max(np.abs(actual-expected))))
        # CPU/GPU float32 encoders need a small cross-device scoring tolerance.
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=1e-4)
        for state,row in zip(out['states'],selected):
            max_prior_error=max(max_prior_error,abs(state.accumulated_log_prior-row['log_prior']))
            max_likelihood_error=max(max_likelihood_error,
                abs(model.env.evaluate_terminal(state).log_likelihood-row['log_likelihood']))
    if max(max_prior_error,max_likelihood_error)>1e-9:
        raise AssertionError('Independent prior/likelihood checks failed')
    required=['posterior_comparison.png','posterior_comparison.pdf','convergence.csv']
    required += [f'policy_example_{i}.trees' for i in range(4)]
    if not all((directory/name).stat().st_size>0 for name in required):
        raise AssertionError('Missing or empty review artifacts')
    provenance=json.loads((directory/'provenance.json').read_text())
    if provenance['reference_used_for_training'] or provenance['checkpoint_selection']!='fixed final update; no reference-metric selection':
        raise AssertionError('Training/reference protocol differs')
    changed=[]
    for name,digest in provenance['source_hashes'].items():
        path=ROOT/name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            changed.append(name)
    gate=recalculated['full_history_tv']['ci95'][1]<config['evaluation']['tv_target']
    if gate!=result['meets_tv_target']:
        raise AssertionError('Saved accuracy gate is incorrect')
    # For independent stratified samples, the sum of squared observation
    # weights is 1/(4*Np)+1/(4*Nq). Hoeffding applies without a normal or
    # equal-variance assumption because each TV integrand lies in [0,1].
    radius=math.sqrt(math.log(2/.05)/8*(1/len(reference)+1/len(samples)))
    estimate=recalculated['full_history_tv']['estimate']
    report=dict(artifact_checks_passed=True,full_history_tv=recalculated['full_history_tv'],
        tv_hoeffding95=dict(radius=radius,interval=[max(0.,estimate-radius),min(1.,estimate+radius)],
            confidence=.95,assumptions='Independent exact-reference and unweighted policy samples; fixed policy independent of evaluation bank'),
        meets_tv_target=gate,training_updates=result['step'],sample_count=count,
        checkpoint_score_checks_per_distribution=len(indexes),max_policy_density_error=max_policy_error,
        max_prior_error=max_prior_error,max_likelihood_error=max_likelihood_error,
        source_files_changed_since_run=changed,accuracy_target=config['evaluation']['tv_target'],
        interpretation='Artifact verification and numerical posterior accuracy are separate checks.')
    (directory/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--score-count',type=int,default=128)
    args=p.parse_args()
    if args.score_count<1:
        p.error('--score-count must be positive')
    torch.set_num_threads(1)
    print(json.dumps(audit_run(args.run,args.score_count),indent=2))


if __name__=='__main__':
    main()
