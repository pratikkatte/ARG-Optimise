"""One independent evaluator, with a durable queue of immutable checkpoints."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid

from training.reporting import write_json


def publish_checkpoint(source, destination):
    """Atomically publish an immutable checkpoint without serializing it twice."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def evaluate_request(request, output):
    """Load a private model; never access the training process's model or RNG."""
    import torch
    from training.checkpoints import load_checkpoint, generator_from_checkpoint
    from training.evaluation import evaluate_generator

    torch.set_num_threads(1)
    step, config = request['step'], request['config']
    checkpoint = Path(request['checkpoint'])
    data = load_checkpoint(checkpoint)
    if data['trainer']['completed_updates'] != step:
        raise ValueError('Evaluation checkpoint update does not match its request')
    generator = generator_from_checkpoint(data, config['eval_async_device'])
    generator.eval()
    truth = None
    if config['terminal_eval']:
        from eval.posterior_summary import TerminalSamplingEvaluator
        truth = TerminalSamplingEvaluator.from_dataset(config['dataset_path'], generator.env,
            grid_size=config['terminal_eval_grid_size'], tmrca_method=config['tmrca_method'])
    reports = []
    for repeat in range(request['repeats']):
        seed = config['eval_seed'] + step * 1000 + repeat
        metrics, details = evaluate_generator(generator, config['eval_episodes'],
            config['eval_batch_size'], seed=seed, max_events=config['max_events'],
            density=config['eval_density_slope'], independent=config['eval_independent_likelihood'],
            terminal_evaluator=truth)
        reports.append(metrics)
        write_json(output/'evaluation'/f'step_{step:06d}_repeat_{repeat:02d}.json.gz',
                   dict(metrics=metrics, details=details, checkpoint_step=step, seed=seed))
    numeric = {key:sum(row[key] for row in reports)/len(reports)
               for key, value in reports[0].items() if isinstance(value, (int, float))
               and all(row[key] is not None for row in reports)}
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return dict(status='completed', step=step, metrics=numeric, reports=reports,
                checkpoint=str(checkpoint), checkpoint_sha256=digest,
                best_metric=config['best_checkpoint_metric'], device=config['eval_async_device'])


def refresh_reports(output, queue):
    """Rebuild derived files from committed results; retries cannot duplicate rows."""
    results = [json.loads(path.read_text()) for path in sorted((queue/'results').glob('*.json'))]
    completed = [row for row in results if row['status'] == 'completed']
    history = {}
    if (output/'evaluation.jsonl').exists():
        for line in (output/'evaluation.jsonl').read_text().splitlines():
            row = json.loads(line)
            history[row['step'], row['repeat']] = row
    for result in completed:
        for repeat, metrics in enumerate(result['reports']):
            history[result['step'], repeat] = dict(step=result['step'], repeat=repeat, **metrics)
    temporary = output/'evaluation.jsonl.tmp'
    with temporary.open('w') as handle:
        for key in sorted(history):
            handle.write(json.dumps(history[key], allow_nan=False)+'\n')
    temporary.replace(output/'evaluation.jsonl')
    if completed:
        best = min(completed, key=lambda row: row['metrics'][row['best_metric']])
        descriptor = output/'checkpoints/best_eval.json'
        previous = json.loads(descriptor.read_text()) if descriptor.exists() else None
        score = best['metrics'][best['best_metric']]
        if previous is None or score < previous['score']:
            import torch
            from training.checkpoints import load_checkpoint
            destination = output/'checkpoints/best_eval.pt'
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Preserve a better checkpoint from earlier synchronous evaluation.
            old = load_checkpoint(destination)['metadata'] if destination.exists() and previous is None else {}
            old_score = old.get('best_eval_score')
            if old_score is None or score < old_score:
                data = load_checkpoint(best['checkpoint'])
                data['metadata'].update(best_eval_score=score, best_eval_metric=best['best_metric'],
                                        best_eval_loss=best['metrics']['eval_subtb_loss'])
                temporary = destination.with_suffix('.pt.async.tmp')
                torch.save(data, temporary)
                temporary.replace(destination)
                write_json(descriptor, dict(step=best['step'], metric=best['best_metric'], score=score,
                    source_checkpoint_sha256=best['checkpoint_sha256']))
    return results


def worker_main(output, parent_pid):
    output = Path(output)
    queue = output/'async_eval'
    # Slurm's pre-timeout signal should make the trainer stop and drain its queue.
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    with (queue/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        refresh_reports(output, queue)
        while os.getppid() == parent_pid:
            pending = [path for path in sorted((queue/'requests').glob('*.json'))
                       if not (queue/'results'/path.name).exists()]
            if not pending:
                if (queue/'stop.json').exists():
                    rows = refresh_reports(output, queue)
                    write_json(queue/'status.json', dict(state='completed', pid=os.getpid(),
                        completed=sum(r['status']=='completed' for r in rows),
                        failed=sum(r['status']=='failed' for r in rows), pending=0))
                    return
                time.sleep(.2)
                continue
            path = pending[0]
            request = json.loads(path.read_text())
            write_json(queue/'status.json', dict(state='evaluating', pid=os.getpid(),
                step=request['step'], pending=len(pending)))
            try:
                result = evaluate_request(request, output)
            except Exception as exc:
                result = dict(status='failed', step=request['step'], checkpoint=request['checkpoint'],
                              error=str(exc), traceback=traceback.format_exc())
                traceback.print_exc()
            write_json(queue/'results'/path.name, result)
            refresh_reports(output, queue)
            print('Evaluation', request['step'], result['status'], flush=True)


class AsyncEvaluator:
    """Submission never waits for evaluation. Only shutdown drains the worker."""
    def __init__(self, output, config, worker_command=None):
        self.output = Path(output).resolve()
        self.queue = self.output/'async_eval'
        for name in ('requests', 'results'):
            (self.queue/name).mkdir(parents=True, exist_ok=True)
        self.config = {key:config[key] for key in (
            'eval_episodes', 'eval_batch_size', 'eval_seed', 'max_events', 'eval_density_slope',
            'eval_independent_likelihood', 'terminal_eval', 'dataset_path',
            'terminal_eval_grid_size', 'tmrca_method', 'best_checkpoint_metric', 'eval_async_device')}
        if self.config['dataset_path']:
            self.config['dataset_path'] = str(Path(self.config['dataset_path']).resolve())
        self.delivered_path = self.queue/'delivered.json'
        self.delivered = set(json.loads(self.delivered_path.read_text())) if self.delivered_path.exists() else set()
        (self.queue/'stop.json').unlink(missing_ok=True)
        source = Path(__file__).resolve().parents[1]
        env = {**os.environ, 'PYTHONPATH':str(source)+os.pathsep+os.environ.get('PYTHONPATH',''),
               'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1'}
        if self.config['eval_async_device'].startswith('cpu'):
            env['CUDA_VISIBLE_DEVICES'] = ''
        # CUDA workers inherit the same Slurm-visible GPU as the training process.
        command = worker_command or [sys.executable, '-u', '-m', 'training.async_evaluation',
            '--output', str(self.output), '--parent-pid', str(os.getpid())]
        self.log = (self.queue/'worker.log').open('a')
        try:
            self.process = subprocess.Popen(command, cwd=source, env=env,
                                            stdout=self.log, stderr=subprocess.STDOUT)
        except BaseException:
            self.log.close()
            raise

    def submit(self, checkpoint, step, repeats):
        path = self.queue/'requests'/f'step_{step:06d}.json'
        request = dict(checkpoint=str(Path(checkpoint).resolve()), step=step,
                       repeats=repeats, config=self.config)
        if path.exists():
            if json.loads(path.read_text()) != request:
                raise ValueError('Refusing to replace an existing evaluation request')
        else:
            write_json(path, request)

    def take_result(self):
        for path in sorted((self.queue/'results').glob('*.json')):
            if path.name not in self.delivered:
                result = json.loads(path.read_text())
                self.delivered.add(path.name)
                write_json(self.delivered_path, sorted(self.delivered))
                return result
        return None

    def finish(self, timeout):
        """Training has ended; drain queued work for a bounded shutdown period."""
        write_json(self.queue/'stop.json', dict(training_finished=True))
        deadline = time.monotonic()+timeout
        while True:
            result = self.take_result()
            if result is not None:
                yield result
                continue
            if self.process.poll() is not None:
                # A result may have been committed between the scan and poll.
                result = self.take_result()
                if result is not None:
                    yield result
                    continue
                break
            if time.monotonic() >= deadline:
                self.abort()
                break
            time.sleep(.1)
        self.log.close()
        pending = [p.name for p in (self.queue/'requests').glob('*.json')
                   if not (self.queue/'results'/p.name).exists()]
        if self.process.returncode or pending:
            write_json(self.queue/'status.json', dict(state='incomplete',
                worker_exit_code=self.process.returncode, pending=pending,
                recovery='Unfinished requests are retained and processed on resume.'))
            print('Async evaluation incomplete; see', self.queue/'status.json', flush=True)

    def abort(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--parent-pid', type=int, required=True)
    arguments = parser.parse_args()
    worker_main(arguments.output, arguments.parent_pid)
