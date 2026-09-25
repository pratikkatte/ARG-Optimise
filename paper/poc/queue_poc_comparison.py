#!/usr/bin/env python3
"""Queue the single-Gamma control after the existing detached main experiment.

This script never restarts an experiment. It verifies the recorded process
identity, waits for it, then requires completed fixed-budget main results.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import psutil
import yaml

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'paper/outputs/poc'


def record(**value):
    path=OUTPUT/'comparison_execution.json'
    value.update(supervisor_pid=os.getpid(),hostname=socket.gethostname(),updated_unix_time=time.time())
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def main():
    execution=json.loads((OUTPUT/'detached_execution.json').read_text())
    if execution['hostname']!=socket.gethostname():
        raise RuntimeError('Queue must run on the main experiment host')
    try:
        process=psutil.Process(execution['pid'])
    except psutil.NoSuchProcess:
        process=None
    if process is not None:
        if (abs(process.create_time()-execution['launched_unix_time'])>5
                or process.cmdline()!=execution['command']):
            raise RuntimeError('Recorded main pid now identifies a different process')
        record(status='waiting_for_main',main_pid=process.pid)
        while process.is_running() and process.status()!=psutil.STATUS_ZOMBIE:
            time.sleep(15)
    config=yaml.safe_load((ROOT/'paper/poc/config.yaml').read_text())
    for seed in config['training']['seeds']:
        path=OUTPUT/'runs'/f'main_gamma_mixture_seed{seed}'/'result.json'
        if not path.exists() or json.loads(path.read_text())['step']!=config['training']['steps']:
            record(status='main_incomplete',missing_or_incomplete_run=str(path))
            return 1
    command=[sys.executable,'-u','paper/poc/run_poc.py','--config','paper/poc/config.yaml',
             '--seeds','7','--head','gamma','--tag','single_gamma']
    log=OUTPUT/'single_gamma_training.stdout.log'
    with log.open('ab',buffering=0) as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        record(status='running',pid=child.pid,command=command,stdout=str(log.relative_to(ROOT)))
        code=child.wait()
    record(status='completed' if code==0 else 'failed',exit_code=code,command=command,stdout=str(log.relative_to(ROOT)))
    return code


if __name__=='__main__':
    raise SystemExit(main())
