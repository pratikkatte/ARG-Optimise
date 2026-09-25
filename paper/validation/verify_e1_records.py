"""Verify checkpoint identity, observations, and saved trajectory scores."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from training.checkpoints import load_checkpoint, generator_from_checkpoint
from gfn.rollout import RolloutWorker
from infer import validate_terminal
from utils import action_from_dict
from env.snp_data import load_snp_dataset

@torch.no_grad()
def main():
    torch.set_num_threads(1)
    cfg=json.loads((Path(__file__).with_name('appendix_e1.json')).read_text())
    checks=[]
    for j in cfg['datasets']:
        path=Path(j['manifest']); d=json.loads(path.read_text())
        ck=Path(j['checkpoint']); c=load_checkpoint(ck); m=c['metadata']
        assert m['wandb_id']==j['run_id'] and m['step']==j['step']
        assert hashlib.sha256(ck.read_bytes()).hexdigest()==d['checkpoint_sha256']
        g=generator_from_checkpoint(c,torch.device('cpu'),optimizer=False); g.eval()
        observed=load_snp_dataset((ROOT/'validation/datasets/paper_datasets')/j['dataset']/'rep0')
        np.testing.assert_array_equal(observed.genotypes,g.env.snp_data.genotypes)
        np.testing.assert_array_equal(observed.positions,g.env.snp_data.positions)
        # Include both ends of the ensemble, rather than only its first batch.
        indices=[0,len(d['samples'])-1]
        records=[d['samples'][i] for i in indices]
        paths=[[action_from_dict(a) for a in r['actions']] for r in records]
        outputs,_=RolloutWorker(g.env).replay(g,paths,collect_flows=False,return_states=True)
        errors=[]
        for i,(r,state) in enumerate(zip(records,outputs['states'])):
            validate_terminal(g.env,state)
            error=abs(float(outputs['log_paths_pf'][i].sum())-r['log_policy_density'])
            assert error<1e-5,(j['dataset'],error)
            assert abs(state.accumulated_log_prior-r['log_prior'])<1e-7
            assert abs(state.partial_log_likelihood-r['log_likelihood'])<1e-7
            errors.append(error)
        check=dict(dataset=j['dataset'],checkpoint=str(ck),checkpoint_sha256=d['checkpoint_sha256'],
            run_id=j['run_id'],step=j['step'],manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            evaluated_draws=len(d['samples']),rescored_indices=indices,max_rescoring_error=max(errors),
            sampling_seed=d.get('base_seed'),batch_size=d.get('batch_size'),
            likelihood_error_all_draws=max(abs(r['log_likelihood']-r['independent_log_likelihood']) for r in d['samples']))
        checks.append(check); print(json.dumps(check),flush=True)
    out=Path(cfg['output']).parent
    out.mkdir(parents=True,exist_ok=True)
    (out/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')

if __name__=='__main__':
    main()
