"""Summarize a fixed 64-history prefix per exploratory proposal group."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CONFIG=Path(__file__).with_name('appendix_e1.json')
OUT=(ROOT/json.loads(CONFIG.read_text())['output']).parent

def main():
    fits=json.loads((OUT/'plots/density_metrics.json').read_text())
    summaries=[]
    for ds in ['r1','r2','r4']:
        for temp in [1.,1.25,1.5]:
            data=json.loads((OUT/'stress'/f'{ds}_T{temp:g}.json').read_text())
            assert data['status']=='complete' and len(data['samples'])>=64
            assert data['checkpoint_sha256']==fits[ds]['checkpoint_sha256']
            samples=data['samples'][:64]
            ll=np.array([r['log_likelihood'] for r in samples])
            nr=np.array([r['recombinations'] for r in samples])
            e=np.array([r['log_policy_density']-r['log_likelihood']-r['log_prior']
                        -fits[ds]['raw']['offset'] for r in samples])
            summaries.append(dict(dataset=ds,proposal_temperature=temp,n=64,
                log_likelihood_min=float(ll.min()),log_likelihood_max=float(ll.max()),
                log_likelihood_median=float(np.median(ll)),
                median_recombinations=float(np.median(nr)),
                residual_rmse=float(np.sqrt(np.mean(e**2))),residual_mean=float(e.mean()),
                completed_stored=len(data['samples']),attempted_stored=data['attempted'],
                max_likelihood_error=max(abs(r['log_likelihood']-r['independent_log_likelihood']) for r in samples)))
    (OUT/'stress_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    lines=[]
    for d in summaries:
        lines.append(f"{d['dataset']} & {d['proposal_temperature']:g} & {d['log_likelihood_median']:.2f} & {d['median_recombinations']:g} & {d['residual_rmse']:.3f} \\\\")
    (OUT/'stress_table_rows.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':
    main()
