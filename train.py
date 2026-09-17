"""Train an infinite-sites shared-encoder GFlowNet from an explicit YAML configuration."""
import json
from pathlib import Path
import time
import torch
import yaml
from env.snp_data import load_snp_dataset
from env.env import SimpleARGEnvironment
from generator import GFlowNetGenerator
from gfn.rollout import RolloutWorker, RolloutFailure
from training.trainer import Trainer
from training.checkpoints import seed_everything, load_checkpoint, generator_from_checkpoint, restore_rng
from training.configuration import (parse_train_args, resolve_config, config_notes, lr_config,
                                    temperature_config, mix_config)
from training.schedules import WarmupCosineScheduler
from training.evaluation import evaluate_generator


def _json(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist()))
    temporary.replace(path)


def _append(path,value):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(value,allow_nan=False)+'\n')


def train(dataset_path=None, output_path=None, device='cpu', **options):
    requested = dict(options,dataset_path=dataset_path,output_path=output_path,device=device)
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
        c = resolve_config({**saved,**requested})
        previous = resolve_config(saved)
        mutable = {'dataset_path','output_path','device','epochs_num','resume_checkpoint','cpu_threads',
                   'verbose','wandb','wandb_project','wandb_entity','wandb_name','wandb_mode',
                   'checkpoint_every','eval_every','eval_episodes','eval_batch_size','eval_seed',
                   'eval_density_slope','eval_independent_likelihood','terminal_eval',
                   'terminal_eval_grid_size','terminal_eval_repeats','terminal_eval_repeat_every',
                   'tmrca_method','evaluation','max_events','init_z_batch_size','chunk_steps'}
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
                     'flow_lr','grad_clip','subtb_lambda','loss_type','flow_head_version','flow_warmup_steps')})
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
    run = None
    notes = config_notes(c)
    (output/'resolved_config.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
    _json(output/'configuration_notes.json',notes)
    wandb_id = checkpoint['metadata'].get('wandb_id') if checkpoint else None
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
        if checkpoint:
            trainer.load_state_dict(checkpoint['trainer']); restore_rng(g.env,checkpoint['rng'])
        else:
            initialization_started = time.perf_counter()
            if c['verbose']:
                print(f'Initializing flow ({c["init_z_sample_count"]} trajectories)...', flush=True)
            g.initialize_flow_center(batch_size=c['init_z_batch_size'], verbose=c['verbose'])
            if c['verbose']:
                print(f'Initialization complete ({time.perf_counter()-initialization_started:.1f}s)', flush=True)
        for step in range(trainer.completed_updates+1,c['epochs_num']+1):
            epoch_started = time.perf_counter()
            info = trainer.train_epoch(batch_size=c['batch_size'],grad_accum_steps=c['grad_accum_steps'])
            _append(output/'training.jsonl',info)
            logged = dict(info)
            improved = False
            eval_text = ''
            due = c['eval_episodes'] and (step%c['eval_every']==0 or step==c['epochs_num'])
            repeated = c['terminal_eval'] and step%c['terminal_eval_repeat_every']==0
            if due or (c['eval_episodes'] and repeated):
                evaluator = None
                if c['terminal_eval']:
                    if not c['dataset_path']:
                        raise ValueError('terminal_eval requires dataset_path for optional truth validation')
                    from eval.posterior_summary import TerminalSamplingEvaluator
                    evaluator = TerminalSamplingEvaluator.from_dataset(c['dataset_path'],g.env,
                          grid_size=c['terminal_eval_grid_size'],tmrca_method=c['tmrca_method'])
                reports = []
                for repeat in range(c['terminal_eval_repeats'] if repeated else 1):
                    metrics,details = evaluate_generator(g,c['eval_episodes'],c['eval_batch_size'],
                        seed=c['eval_seed']+step*1000+repeat,max_events=c['max_events'],
                        density=c['eval_density_slope'],independent=c['eval_independent_likelihood'],
                        terminal_evaluator=evaluator)
                    reports.append(metrics)
                    _append(output/'evaluation.jsonl',dict(step=step,repeat=repeat,**metrics))
                    _json(output/'evaluation'/f'step_{step:06d}_repeat_{repeat:02d}.json',dict(metrics=metrics,details=details))
                mean_loss = sum(r['eval_subtb_loss'] for r in reports)/len(reports)
                eval_text = f'  eval_subtb_loss={mean_loss:.4f}'
                if best_eval is None or mean_loss<best_eval:
                    best_eval,improved = mean_loss,True
                if run is not None:
                    numeric = {k:sum(r[k] for r in reports)/len(reports) for k,v in reports[0].items()
                               if isinstance(v,(int,float)) and all(r[k] is not None for r in reports)}
                    logged.update(numeric)
            if run is not None:
                run.log(logged,step=step)
            metadata = dict(resolved_config=c,configuration_notes=notes,rate_overrides=rate_overrides,
                run_config={k:c[k] for k in ('batch_size','max_events','chunk_steps','seed','dataset_path')},
                step=step,best_eval_loss=best_eval,wandb_id=wandb_id)
            if step%c['checkpoint_every']==0 or step==c['epochs_num']:
                g.save(output/'checkpoints'/f'checkpoint_{step:04d}.pt',trainer=trainer,metadata=metadata)
            if improved:
                g.save(output/'checkpoints'/'best_eval.pt',trainer=trainer,metadata=metadata)
            if c['verbose']:
                print(f'Epoch {step}/{c["epochs_num"]}  subtb_loss={info["loss"]:.4f}'
                      f'  time={time.perf_counter()-epoch_started:.1f}s{eval_text}', flush=True)
        return g,trainer
    except Exception as exc:
        payload = dict(error=str(exc),completed_updates=trainer.completed_updates)
        if isinstance(exc,RolloutFailure):
            payload['histories'] = exc.histories
        _json(output/'failure.json',payload)
        raise
    finally:
        if run is not None:
            run.finish()


if __name__ == '__main__':
    train(**parse_train_args())
