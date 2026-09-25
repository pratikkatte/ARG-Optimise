"""Render Appendix E.1 from explicitly selected manifests and W&B histories."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

def stats(x, y):
    offset = float(np.mean(y-x))
    residual = y-x-offset
    return dict(slope=float(np.polyfit(x,y,1)[0]),
                pearson=float(np.corrcoef(x,y)[0,1]), offset=offset,
                rmse=float(np.sqrt(np.mean(residual**2))))

def save(fig, name, out):
    fig.savefig(out/(name+'.pdf'), bbox_inches='tight')
    fig.savefig(out/(name+'.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    out=Path(cfg['output']); out.mkdir(parents=True,exist_ok=True)
    jobs=cfg['datasets']; n=len(jobs)
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False,
                         'axes.spines.right':False, 'pdf.fonttype':42})
    fig,axs=plt.subplots(2,n,figsize=(3.2*n,6.1),squeeze=False,layout='constrained')
    residual_fig,resaxs=plt.subplots(1,n,figsize=(3.2*n,2.8),squeeze=False,layout='constrained')
    metrics={}
    for col,j in enumerate(jobs):
        path=Path(j['manifest']); manifest=json.loads(path.read_text())
        assert manifest['status']=='complete'
        assert manifest['checkpoint_sha256']==hashlib.sha256(Path(j['checkpoint']).read_bytes()).hexdigest()
        records=manifest['samples']
        ll=np.array([r['log_likelihood'] for r in records])
        prior=np.array([r['log_prior'] for r in records])
        q=np.array([r['log_policy_density'] for r in records])
        assert np.isfinite(np.stack([ll,prior,q])).all()
        assert max(abs(r['independent_log_likelihood']-r['log_likelihood']) for r in records)<1e-7
        raw=stats(ll+prior,q); relative=stats(ll,q-prior)
        np.testing.assert_allclose(raw['rmse'],relative['rmse'],atol=1e-9)
        metrics[j['dataset']]=dict(n=len(records),checkpoint=j['checkpoint'],
            checkpoint_sha256=manifest['checkpoint_sha256'],raw=raw,prior_relative=relative)
        for row,(x,y,s) in enumerate([(ll+prior,q,raw),(ll,q-prior,relative)]):
            ax=axs[row,col]
            ax.scatter(x,y,s=5,alpha=.22,color='#286a98',rasterized=True)
            limits=np.array([x.min(),x.max()])
            ax.plot(limits,limits+raw['offset'],'--',color='#ce6a36',lw=1.4,label='Slope one')
            ax.text(.04,.96,f"Slope = {s['slope']:.3f}\nPearson r = {s['pearson']:.3f}",
                    transform=ax.transAxes,va='top',bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
            ax.set_xlabel('Log likelihood + log prior' if row==0 else 'Log likelihood')
            ax.set_ylabel('Log policy density' if row==0 else 'Log policy density − log prior')
            if row==0: ax.set_title(f"{j['dataset']} · checkpoint {j['step']}\nn = {len(records):,}")
            ax.legend(loc='lower right',frameon=False)
        residual=q-ll-prior-raw['offset']; nr=np.array([r['recombinations'] for r in records])
        ax=resaxs[0,col]; ax.scatter(nr,residual,s=5,alpha=.22,color='#286a98',rasterized=True)
        ax.axhline(0,color='#ce6a36',linestyle='--',lw=1.2)
        ax.set(xlabel='Recombination events',ylabel='Centered log-density residual',title=j['dataset'])
        metrics[j['dataset']]['residual_recombination_pearson']=float(np.corrcoef(nr,residual)[0,1])
    save(fig,'figure_4',out); save(residual_fig,'density_residual_complexity',out)
    (out/'density_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    fig,axs=plt.subplots(3,n,figsize=(3.2*n,7.5),squeeze=False,layout='constrained')
    summaries={}
    for col,j in enumerate(jobs):
        history=json.loads((out.parent/'training'/f"{j['run_id']}.json").read_text())
        rows=pd.DataFrame(history['rows'])
        train=rows[rows.subtb_loss.notna()].copy()
        # A resumed W&B run may contain overlapping steps. Choose an explicit
        # segment, never silently join different training branches.
        train['segment']=(train.step.diff()<0).cumsum()
        train=train[train.segment==j.get('training_segment',0)]
        segment_start,segment_end=train._step.min(),train._step.max()
        train=train[train.step<=j['step']]
        for row,key,label in [(0,'subtb_loss','SubTB loss'),(1,'tb_loss','Auxiliary TB loss')]:
            ax=axs[row,col]; v=train[key].to_numpy(); x=train.step.to_numpy()
            ax.plot(x,v,color='#286a98',alpha=.15,lw=.45)
            ax.plot(x,pd.Series(v).rolling(50,min_periods=1).median(),color='#286a98',lw=1)
            ax.set_ylabel(label); ax.set_yscale('log')
            ax.axvline(j['step'],color='#ce6a36',ls='--',lw=1)
        if 'eval_log_weight_std' in rows:
            ev=rows[rows.eval_log_weight_std.notna()].copy()
            # Restrict evaluation records to the selected training segment's
            # logging interval; async evaluations use their checkpoint step.
            ev=ev[(ev._step>=segment_start) & (ev._step<=segment_end+1)]
            ev=ev[ev.eval_checkpoint_step<=j['step']]
            ev=ev.sort_values('eval_checkpoint_step').drop_duplicates('eval_checkpoint_step',keep='last')
            axs[2,col].plot(ev.eval_checkpoint_step,ev.eval_log_weight_std,'o-',ms=2,lw=.8,color='#286a98')
            summaries[j['dataset']]=dict(training_rows=len(train),evaluation_rows=len(ev),
                eval_sample_counts=sorted(ev.eval_episodes.dropna().unique().tolist()),
                first_evaluation=ev[['eval_checkpoint_step','eval_log_weight_std']].head(1).to_dict('records'),
                last_evaluation=ev[['eval_checkpoint_step','eval_log_weight_std']].tail(1).to_dict('records'))
        axs[0,col].set_title(f"r{col + 1}")
        axs[2,col].set(xlabel='Optimizer update',ylabel='Log-density residual SD')
        axs[2,col].axvline(j['step'],color='#ce6a36',ls='--',lw=1)
    save(fig,'figure_5',out)
    (out/'training_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':
    main()
