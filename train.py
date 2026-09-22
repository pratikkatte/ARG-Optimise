"""Train an infinite-sites shared-encoder GFlowNet from an explicit YAML configuration."""
import json
from pathlib import Path
import time
import traceback
import signal
import threading
import torch
import yaml
from env.snp_data import load_snp_dataset
from env.env import SimpleARGEnvironment
from generator import GFlowNetGenerator, checkpoint_model_config
from gfn.rollout import RolloutWorker, RolloutFailure
from training.trainer import Trainer
from training.checkpoints import seed_everything, load_checkpoint, generator_from_checkpoint, restore_rng
from training.configuration import (parse_train_args, resolve_config, config_notes, lr_config,
                                    temperature_config, mix_config)
from training.schedules import WarmupCosineScheduler
from training.evaluation import evaluate_generator
from training.reporting import write_json as _json
from training.async_evaluation import AsyncEvaluator, publish_checkpoint


def _append(path,value):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(value,allow_nan=False)+'\n')


def train(dataset_path=None, output_path=None, device=None, **options):
    # Omitted arguments inherit checkpoint settings on resume. Fresh runs obtain
    # their defaults from resolve_config; Python defaults must not overwrite them.
    requested = dict(options)
    requested.update({key:value for key,value in dict(dataset_path=dataset_path,
        output_path=output_path,device=device).items() if value is not None})
    if 'epochs' in requested:
        requested['epochs_num'] = requested.pop('epochs')
    checkpoint = load_checkpoint(requested['resume_checkpoint']) if requested.get('resume_checkpoint') else None
    if checkpoint:
        saved = checkpoint['metadata'].get('resolved_config')
        if saved is None:
            # Earlier infinite-sites checkpoints have the smaller v1 run configuration.
            saved = {**checkpoint['metadata'].get('run_config',{}),
                     **checkpoint['metadata']['generator_config'],
                     'model_kwargs':checkpoint['metadata']['model']}
            if checkpoint['trainer']:
                saved.update(checkpoint['trainer']['config'])
            for source,target in (('population_size','effective_population_size'),
                                  ('mutation_rate','mutation_rate'),('recombination_rate','recombination_rate'),
                                  ('reward_C','reward_C')):
                saved[target] = checkpoint['metadata']['environment'][source]
        # Discard retired logging settings from older checkpoints.
        saved = {k:v for k,v in saved.items() if k not in ('debug_progress','progress_every_seconds')}
        saved['model_kwargs'] = checkpoint_model_config(checkpoint['metadata']['model'])
        saved.setdefault('flow_scale_mode', 'empirical')
        saved.setdefault('best_checkpoint_metric', 'eval_subtb_loss')
        c = resolve_config({**saved,**requested})
        previous = resolve_config(saved)
        mutable = {'dataset_path','output_path','device','epochs_num','resume_checkpoint','cpu_threads',
                   'verbose','wandb','wandb_project','wandb_entity','wandb_name','wandb_mode',
                   'checkpoint_every','eval_every','eval_episodes','eval_batch_size','eval_seed',
                   'eval_async','eval_async_device','eval_async_shutdown_seconds',
                   'eval_density_slope','eval_independent_likelihood','terminal_eval',
                   'terminal_eval_grid_size','terminal_eval_repeats','terminal_eval_repeat_every',
                   'tmrca_method','evaluation','max_events','init_z_batch_size','chunk_steps',
                   'max_wall_seconds','eval_initial'}
        for key in c.keys()-mutable:
            if c[key] != previous[key]:
                raise ValueError('Cannot change '+key+' when resuming; use a fresh run')
    else:
        c = resolve_config(requested)
    if not c['output_path']:
        raise ValueError('output_path is required')
    torch.set_num_threads(c['cpu_threads']); seed_everything(c['seed'])
    if str(c['device']).startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable in this PyTorch installation')
    if checkpoint:
        g = generator_from_checkpoint(checkpoint,c['device'],optimizer=True)
        if checkpoint['trainer'] is None:
            raise ValueError('Checkpoint has no resumable trainer state')
        if c['dataset_path']:
            observed = load_snp_dataset(c['dataset_path'])
            candidate = SimpleARGEnvironment(snp_data=observed,**checkpoint['metadata']['environment'])
            if candidate.dataset_fingerprint != g.env.dataset_fingerprint:
                raise ValueError('Resume dataset differs from the checkpoint')
        rate_overrides = checkpoint['metadata'].get('rate_overrides',{})
    else:
        if c['dataset_path'] is None or not Path(c['dataset_path']).is_dir():
            raise ValueError('dataset_path must be an infinite-sites replicate directory; FASTA is retired')
        data = load_snp_dataset(c['dataset_path'])
        params = json.loads((Path(c['dataset_path'])/'metadata.json').read_text())['parameters']
        rate_overrides = {key:c[option] for key,option in (
                    ('population_size','effective_population_size'),('mutation_rate','mutation_rate'),
                    ('recombination_rate','recombination_rate')) if c[option] is not None}
        rates = {key:rate_overrides.get(key,params[key]) for key in
                    ('population_size','mutation_rate','recombination_rate')}
        env = SimpleARGEnvironment(snp_data=data,seed=c['seed'],reward_C=c['reward_C'],**rates)
        g = GFlowNetGenerator(env,device=c['device'],model_kwargs=c['model_kwargs'],
                initialize_z_from_policy=False,**{k:c[k] for k in ('init_z_sample_count','policy_lr',
                     'flow_lr','grad_clip','subtb_lambda','loss_type','flow_head_version','flow_warmup_steps',
                     'encoder_lr','flow_encoder_grad_scale','tb_loss_weight','flow_scale_mode')})
        if c['lr_schedule']=='cosine':
            g.scheduler = WarmupCosineScheduler(g.opt,lr_config(c))
    # Persist actual scientific rates, not just optional overrides or metadata paths.
    c.update(effective_population_size=g.env.population_size,mutation_rate=g.env.mutation_rate,
             recombination_rate=g.env.recombination_rate)
    if c['lr_schedule']=='cosine':
        c['lr_schedule_steps'] = g.scheduler.config.total_steps
    output = Path(c['output_path'])
    if output.exists() and any(output.iterdir()) and not checkpoint:
        raise ValueError('Fresh training requires an empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    g.max_events = c['max_events']
    worker = RolloutWorker(g.env,max_events=c['max_events'])
    trainer = Trainer(g,worker,mix_config(c),seed=c['seed'],chunk_steps=c['chunk_steps'],
                      temperature_config=temperature_config(c))
    best_eval = checkpoint['metadata'].get('best_eval_loss') if checkpoint else None
    best_score = (checkpoint['metadata'].get('best_eval_score',best_eval)
                  if checkpoint else None)
    run = None
    notes = config_notes(c)
    (output/'resolved_config.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
    _json(output/'configuration_notes.json',notes)
    wandb_id = checkpoint['metadata'].get('wandb_id') if checkpoint else None
    exit_code, phase, evaluation_context = 0, 'initialization', None
    started = time.monotonic()
    stop_reason, previous_handlers = None, {}
    async_evaluator = None

    def request_shutdown(signum, frame):
        nonlocal stop_reason
        stop_reason = signal.Signals(signum).name

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGUSR1):
            previous_handlers[signum] = signal.signal(signum, request_shutdown)

    def metadata(step):
        result = dict(resolved_config=c,configuration_notes=notes,rate_overrides=rate_overrides,
            run_config={k:c[k] for k in ('batch_size','max_events','chunk_steps','seed','dataset_path')},
            step=step,best_eval_loss=best_eval,best_eval_score=best_score,
            best_eval_metric=c['best_checkpoint_metric'],wandb_id=wandb_id)
        if checkpoint and 'training_fork' in checkpoint['metadata']:
            result['training_fork'] = checkpoint['metadata']['training_fork']
        return result

    def evaluate(step, repeats):
        nonlocal phase, evaluation_context, best_eval, best_score
        phase = 'evaluation'
        evaluator = None
        if c['terminal_eval']:
            if not c['dataset_path']:
                raise ValueError('terminal_eval requires dataset_path for optional truth validation')
            from eval.posterior_summary import TerminalSamplingEvaluator
            evaluator = TerminalSamplingEvaluator.from_dataset(c['dataset_path'],g.env,
                  grid_size=c['terminal_eval_grid_size'],tmrca_method=c['tmrca_method'])
        reports = []
        for repeat in range(repeats):
            evaluation_context = dict(step=step,repeat=repeat,seed=c['eval_seed']+step*1000+repeat)
            metrics,details = evaluate_generator(g,c['eval_episodes'],c['eval_batch_size'],
                seed=evaluation_context['seed'],max_events=c['max_events'],
                density=c['eval_density_slope'],independent=c['eval_independent_likelihood'],
                terminal_evaluator=evaluator)
            reports.append(metrics)
            _append(output/'evaluation.jsonl',dict(step=step,repeat=repeat,**metrics))
            _json(output/'evaluation'/f'step_{step:06d}_repeat_{repeat:02d}.json.gz',dict(metrics=metrics,details=details))
        numeric = {k:sum(r[k] for r in reports)/len(reports) for k,v in reports[0].items()
                   if isinstance(v,(int,float)) and all(r[k] is not None for r in reports)}
        mean_loss, score = numeric['eval_subtb_loss'], numeric[c['best_checkpoint_metric']]
        best_eval = mean_loss if best_eval is None else min(best_eval,mean_loss)
        improved = best_score is None or score<best_score
        if improved:
            best_score = score
            g.save(output/'checkpoints'/'best_eval.pt',trainer=trainer,metadata=metadata(step))
        return numeric

    def status(value, reason=None):
        payload = dict(status=value,reason=reason,completed_updates=trainer.completed_updates,
                       target_updates=c['epochs_num'],wall_seconds=time.monotonic()-started)
        _json(output/'run_status.json',payload)
        if run is not None and hasattr(run,'summary'):
            run.summary.update(payload)

    def async_metrics(result):
        nonlocal best_eval, best_score
        if result is None:
            return {}
        if result['status'] == 'failed':
            print(f'Async evaluation failed at update {result["step"]}: '
                  f'{result["error"]}. Training continues; see async_eval/results.', flush=True)
            return dict(eval_checkpoint_step=result['step'], eval_async_failed=1)
        numeric = result['metrics']
        score, loss = numeric[c['best_checkpoint_metric']], numeric['eval_subtb_loss']
        best_score = score if best_score is None else min(best_score, score)
        best_eval = loss if best_eval is None else min(best_eval, loss)
        return dict(eval_checkpoint_step=result['step'], eval_async_failed=0, **numeric)

    try:
        if c['wandb']:
            try:
                import wandb
            except ImportError as exc:
                raise ValueError('wandb: true requires the wandb package; install it or set wandb: false') from exc
            run = wandb.init(project=c['wandb_project'],entity=c['wandb_entity'],
                name=c['wandb_name'] or output.name,mode=c['wandb_mode'],dir=str(output),
                config=c,id=wandb_id,resume='allow' if wandb_id else None)
            wandb_id = run.id if run is not None else None
            if c['eval_async'] and run is not None:
                run.define_metric('train_update')
                run.define_metric('*', step_metric='train_update')
                run.define_metric('eval_checkpoint_step')
                run.define_metric('eval_*', step_metric='eval_checkpoint_step')
        if c['eval_async']:
            async_evaluator = AsyncEvaluator(output, c)
        if checkpoint:
            trainer.load_state_dict(checkpoint['trainer']); restore_rng(g.env,checkpoint['rng'])
        else:
            initialization_started = time.perf_counter()
            if c['verbose']:
                print(f'Initializing flow ({c["init_z_sample_count"]} trajectories)...', flush=True)
            g.initialize_flow_center(batch_size=c['init_z_batch_size'], verbose=c['verbose'])
            if c['verbose']:
                print(f'Initialization complete ({time.perf_counter()-initialization_started:.1f}s)', flush=True)
            if c['eval_initial'] and c['eval_episodes']:
                g.save(output/'checkpoints'/'checkpoint_0000.pt',trainer=trainer,metadata=metadata(0))
                if async_evaluator is not None:
                    async_evaluator.submit(output/'checkpoints'/'checkpoint_0000.pt',0,c['terminal_eval_repeats'])
                else:
                    numeric = evaluate(0,c['terminal_eval_repeats'])
                    if run is not None:
                        run.log(numeric,step=0)
                    if c['verbose']:
                        print(f'Initial evaluation: ESS={numeric["eval_ess"]:.3f}, '
                              f'log_weight_std={numeric["eval_log_weight_std"]:.3f}',flush=True)
        if async_evaluator is not None:
            # A recoverable initial state also exists before the first interval.
            g.save(output/'checkpoints'/'latest.pt',trainer=trainer,
                   metadata=metadata(trainer.completed_updates))
        for step in range(trainer.completed_updates+1,c['epochs_num']+1):
            phase, evaluation_context = 'training', None
            epoch_started = time.perf_counter()
            info = trainer.train_epoch(batch_size=c['batch_size'],grad_accum_steps=c['grad_accum_steps'])
            _append(output/'training.jsonl',info)
            logged = dict(info)
            if async_evaluator is not None:
                logged.update(async_metrics(async_evaluator.take_result()))
            eval_text = ''
            due = c['eval_episodes'] and (step%c['eval_every']==0 or step==c['epochs_num'])
            # Sampling repeats do not require the optional tree-truth evaluator.
            repeated = c['eval_episodes'] and step%c['terminal_eval_repeat_every']==0
            # This checkpoint contains the completed optimizer/replay/RNG state.
            # An evaluation exception must never erase an already completed update.
            scheduled = step%c['checkpoint_every']==0 or step==c['epochs_num']
            evaluation_due = bool(due or (c['eval_episodes'] and repeated))
            snapshot = output/'checkpoints'/f'checkpoint_{step:04d}.pt'
            if scheduled or (async_evaluator is not None and evaluation_due):
                g.save(snapshot,trainer=trainer,metadata=metadata(step))
            if stop_reason or (c['max_wall_seconds'] and time.monotonic()-started>=c['max_wall_seconds']):
                if async_evaluator is not None:
                    if not (scheduled or evaluation_due):
                        g.save(snapshot,trainer=trainer,metadata=metadata(step))
                    publish_checkpoint(snapshot, output/'checkpoints'/'latest.pt')
                    async_evaluator.submit(snapshot,step,c['terminal_eval_repeats'] if repeated else 1)
                else:
                    g.save(output/'checkpoints'/'latest.pt',trainer=trainer,metadata=metadata(step))
                if run is not None:
                    if async_evaluator is not None:
                        run.log(dict(train_update=step, **logged))
                    else:
                        run.log(logged,step=step)
                interruption_reason = stop_reason or 'wall_time_budget'
                status('interrupted',interruption_reason)
                if c['verbose']:
                    print(f'Stopped after update {step}: {interruption_reason}. '
                          f'Saved resumable checkpoint: {output/"checkpoints"/"latest.pt"}',
                          flush=True)
                return g,trainer
            if evaluation_due:
                if async_evaluator is not None:
                    publish_checkpoint(snapshot, output/'checkpoints'/'latest.pt')
                    async_evaluator.submit(snapshot,step,c['terminal_eval_repeats'] if repeated else 1)
                    eval_text = '  evaluation_queued'
                else:
                    g.save(output/'checkpoints'/'latest.pt',trainer=trainer,metadata=metadata(step))
                    numeric = evaluate(step,c['terminal_eval_repeats'] if repeated else 1)
                    eval_text = f'  eval_subtb_loss={numeric["eval_subtb_loss"]:.4f}'
                    logged.update(numeric)
            if run is not None:
                if async_evaluator is not None:
                    # W&B's automatic history index stays monotonic across resume,
                    # including evaluation records logged after training stopped.
                    run.log(dict(train_update=step, **logged))
                else:
                    run.log(logged,step=step)
            if c['verbose']:
                loss_label = 'objective_loss' if c['tb_loss_weight'] else 'subtb_loss'
                print(f'Epoch {step}/{c["epochs_num"]}  {loss_label}={info["loss"]:.4f}'
                      f'  time={time.perf_counter()-epoch_started:.1f}s{eval_text}', flush=True)
        status('completed','target_updates_reached')
        return g,trainer
    except BaseException as exc:
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        payload = dict(error=str(exc),error_type=type(exc).__name__,completed_updates=trainer.completed_updates,
                       phase=phase,evaluation=evaluation_context,traceback=traceback.format_exc())
        if hasattr(exc, 'details'):
            payload['details'] = exc.details
        if isinstance(exc,RolloutFailure):
            payload['histories'] = exc.histories
        _json(output/'failure.json',payload)
        status('failed',str(exc))
        # Print before W&B finalizes, so its console log contains the failure.
        traceback.print_exc()
        raise
    finally:
        if async_evaluator is not None:
            if c['verbose']:
                print('Training has ended; draining queued evaluations.', flush=True)
            try:
                for result in async_evaluator.finish(c['eval_async_shutdown_seconds']):
                    numeric = async_metrics(result)
                    if run is not None:
                        run.log(numeric)
            except Exception as exc:
                async_evaluator.abort()
                _json(output/'async_eval/supervisor_failure.json',
                      dict(error=str(exc),traceback=traceback.format_exc()))
                print('Async evaluation shutdown failed:', exc, flush=True)
        for signum, handler in previous_handlers.items():
            signal.signal(signum,handler)
        if run is not None:
            run.finish(exit_code=exit_code)


if __name__ == '__main__':
    train(**parse_train_args())
