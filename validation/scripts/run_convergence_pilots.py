"""Run reproducible paper-dataset pilots from a frozen source snapshot.

The parent owns and records every child PID, polls its actual exit status, and
never restarts a live or failed experiment automatically. No W&B uploads.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import yaml

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--ratios',nargs='+',choices=['r1','r2','r4'],default=['r1','r2','r4'])
    parser.add_argument('--parallel',type=int,default=1)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--depth',type=int,default=2)
    parser.add_argument('--wall-seconds',type=float,default=10800)
    parser.add_argument('--subtb-lambda',type=float)
    parser.add_argument('--tb-loss-weight',type=float)
    args=parser.parse_args()
    if args.steps<1 or args.parallel<1 or args.batch_size<1 or len(set(args.ratios))!=len(args.ratios):
        parser.error('Positive counts and distinct datasets are required')
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Use an empty experiment directory; existing runs are never overwritten')
    output.mkdir(parents=True,exist_ok=True)
    source=output/'source';source.mkdir()
    for directory in ('env','eval','gfn','policy','training'):
        shutil.copytree(ROOT/directory,source/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('train.py','generator.py','infer.py','utils.py','breakpoint_model.py'):
        shutil.copy2(ROOT/name,source/name)
    configurations={}
    for ratio in args.ratios:
        c=yaml.safe_load((ROOT/f'config/paper_datasets/stable/{ratio}.yaml').read_text())
        c.update(dataset_path=str(ROOT/c['dataset_path']),output_path=str(output/ratio),
            epochs=args.steps,batch_size=args.batch_size,grad_accum_steps=1,
            transformer_depth=args.depth,initial_recombination_bias=-1.5,
            lr_warmup_steps=50,eval_every=100,checkpoint_every=100,
            terminal_eval_repeat_every=100,terminal_eval_repeats=3,
            max_wall_seconds=args.wall_seconds,wandb=False)
        if args.subtb_lambda is not None:
            c['subtb_lambda']=args.subtb_lambda
        if args.tb_loss_weight is not None:
            c['tb_loss_weight']=args.tb_loss_weight
        target=output/(ratio+'.yaml');target.write_text(yaml.safe_dump(c,sort_keys=False))
        configurations[ratio]=str(target)
    manifest=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        python=sys.executable,launcher_pid=os.getpid(),arguments=vars(args),
        source_sha256={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in source.rglob('*.py')},configurations=configurations,runs={})
    def save():
        temporary=output/'manifest.tmp'
        temporary.write_text(json.dumps(manifest,indent=2));temporary.replace(output/'manifest.json')
    active={};pending=list(args.ratios);stop=[False]
    def shutdown(signum,frame):
        stop[0]=True
        for process,handle in active.values():
            if process.poll() is None:process.send_signal(signal.SIGUSR1)
    for signum in (signal.SIGTERM,signal.SIGUSR1):signal.signal(signum,shutdown)
    env=dict(os.environ,OMP_NUM_THREADS='1',PYTHONNOUSERSITE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    while active or (pending and not stop[0]):
        while pending and len(active)<args.parallel and not stop[0]:
            ratio=pending.pop(0);handle=(output/(ratio+'.log')).open('w')
            process=subprocess.Popen([sys.executable,'-u',str(source/'train.py'),'--config',configurations[ratio]],
                cwd=source,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[ratio]=(process,handle)
            manifest['runs'][ratio]=dict(pid=process.pid,state='running',started_at=time.time())
            save();print('started',ratio,process.pid,flush=True)
        for ratio,(process,handle) in list(active.items()):
            code=process.poll()
            if code is not None:
                handle.close();active.pop(ratio)
                manifest['runs'][ratio].update(state='exited',exit_code=code,finished_at=time.time())
                save();print('exited',ratio,code,flush=True)
        if active:time.sleep(5)
    manifest['unstarted']=pending;save()
    return int(any(r.get('exit_code',0) for r in manifest['runs'].values()))


if __name__=='__main__':
    raise SystemExit(main())
