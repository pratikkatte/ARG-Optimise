"""Sample valid perturbed histories and score all of them at temperature one."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
import numpy as np
import torch
from training.checkpoints import load_checkpoint, generator_from_checkpoint
from infer import validate_terminal, seed_everything
from gfn.rollout import RolloutWorker
from env.snp_data import load_snp_dataset
from utils import action_as_dict

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=64)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    out=(PROJECT/cfg['output']).parent/'stress'; out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    for di,job in enumerate(cfg['datasets']):
        ck=Path(job['checkpoint']); data=load_checkpoint(ck)
        g=generator_from_checkpoint(data,torch.device('cuda'),optimizer=False); g.eval()
        observed=load_snp_dataset(PROJECT/'paper/datasets'/job['dataset']/'rep0')
        np.testing.assert_array_equal(observed.genotypes,g.env.snp_data.genotypes)
        np.testing.assert_array_equal(observed.positions,g.env.snp_data.positions)
        worker=RolloutWorker(g.env,max_events=10000)
        for ti,temp in enumerate([1.,1.25,1.5]):
            destination=out/f"{job['dataset']}_T{temp:g}.json"
            if destination.exists():
                previous=json.loads(destination.read_text())
                assert previous['checkpoint_sha256']==hashlib.sha256(ck.read_bytes()).hexdigest()
                if previous.get('status')=='complete' and len(previous['samples'])>=args.samples:
                    continue
                if previous.get('status')=='failed':
                    raise RuntimeError(f'Failed evaluation requires inspection: {destination}')
            else:
                previous=None
            result=dict(dataset=job['dataset'],checkpoint=str(ck),
                checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),
                proposal_temperature=temp,scoring_temperature=1.,max_events=10000,
                status='running',samples=[],batch_seeds=[],attempted=0)
            if previous is not None:
                result=previous
                result['interrupted_batch_repeated_with_same_seed']=True
                result['attempted']=len(result['samples'])
            started=time.monotonic()
            try:
                for start in range(len(result['samples']),args.samples,16):
                    seed=20260926+di*10000+ti*1000+start//16
                    seed_everything(seed); g.env.rng.seed(seed)
                    count=min(16,args.samples-start); result['attempted']+=count
                    outputs,paths=worker.rollout(g,count,random_spec={'T':temp},return_states=True)
                    scored,_=worker.replay(g,paths,collect_flows=False)
                    if temp==1.:
                        torch.testing.assert_close(outputs['log_paths_pf'],scored['log_paths_pf'],rtol=1e-9,atol=1e-9)
                    for i,(state,path) in enumerate(zip(outputs['states'],paths)):
                        ref=validate_terminal(g.env,state)
                        result['samples'].append(dict(log_policy_density=float(scored['log_paths_pf'][i].sum()),
                            log_likelihood=state.partial_log_likelihood,independent_log_likelihood=ref.log_likelihood,
                            log_prior=state.accumulated_log_prior,event_count=len(path),
                            recombinations=sum(a.event_type=='recomb' for a in path.actions),
                            actions=[action_as_dict(a) for a in path.actions]))
                    result['batch_seeds'].append(seed)
                    destination.write_text(json.dumps(result)+'\n')
                    print(job['dataset'],temp,len(result['samples']),round(time.monotonic()-started,1),'seconds',flush=True)
                result['status']='complete'
            except Exception as exc:
                result['status']='failed'; result['error']=str(exc)
                if hasattr(exc,'histories'):result['failed_histories']=exc.histories
                raise
            finally:
                destination.write_text(json.dumps(result)+'\n')
        del g,data; torch.cuda.empty_cache()

if __name__=='__main__':
    main()
