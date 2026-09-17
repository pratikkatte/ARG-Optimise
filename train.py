"""Train the infinite-sites shared-encoder GFlowNet on one SNP dataset."""
import argparse
import json
from pathlib import Path
import torch
import yaml
from env.snp_data import load_snp_dataset
from env.env import SimpleARGEnvironment
from generator import GFlowNetGenerator, MODEL_VERSION, DEFAULT_MODEL
from gfn.rollout import RolloutWorker, RolloutFailure
from training.trainer import Trainer, TrajectoryMixConfig
from training.checkpoints import seed_everything, load_checkpoint, generator_from_checkpoint, restore_rng


def train(dataset_path, output_path, device='cpu', *, epochs_num=20, batch_size=2, seed=7,
          effective_population_size=None, mutation_rate=None, recombination_rate=None,
          policy_lr=1e-4, flow_lr=1e-3, grad_clip=10., subtb_lambda=.9, init_z_sample_count=8,
          model_kwargs=None, replay_fraction=.25, replay_min_size=8, exploration_fraction=0.,
          max_events=10000, chunk_steps=16, checkpoint_every=10, resume_checkpoint=None,
          cpu_threads=1, **unsupported):
    if unsupported:
        raise ValueError('Unsupported or legacy training settings: '+', '.join(sorted(unsupported)))
    if epochs_num < 1 or batch_size < 1 or cpu_threads < 1 or checkpoint_every < 1:
        raise ValueError('Training counts must be positive')
    torch.set_num_threads(cpu_threads)
    seed_everything(seed)
    if resume_checkpoint:
        checkpoint = load_checkpoint(resume_checkpoint)
        generator = generator_from_checkpoint(checkpoint, device, optimizer=True)
        requested_rates = dict(population_size=effective_population_size, mutation_rate=mutation_rate,
                               recombination_rate=recombination_rate)
        if any(value is not None and value != checkpoint['metadata']['environment'][key]
               for key, value in requested_rates.items()):
            raise ValueError('Cannot change scientific parameters when resuming a checkpoint')
        if model_kwargs is not None and {**DEFAULT_MODEL, **model_kwargs} != generator.model_kwargs:
            raise ValueError('Cannot change the neural architecture on resume')
        if dataset_path is not None:
            observed = load_snp_dataset(dataset_path)
            candidate = SimpleARGEnvironment(snp_data=observed, **checkpoint['metadata']['environment'])
            if candidate.dataset_fingerprint != generator.env.dataset_fingerprint:
                raise ValueError('Resume dataset differs from the checkpoint')
        if checkpoint['trainer'] is None:
            raise ValueError('Checkpoint has no resumable trainer state')
        config = TrajectoryMixConfig(**checkpoint['trainer']['config'])
        if 'run_config' not in checkpoint['metadata']:
            raise ValueError('CLI resume requires saved run_config; programmatic checkpoints can restore through Trainer.load_state_dict')
        run_config = checkpoint['metadata']['run_config']
        batch_size, max_events, chunk_steps = (run_config[k] for k in ('batch_size','max_events','chunk_steps'))
        rate_overrides = checkpoint['metadata'].get('rate_overrides', {})
    else:
        if dataset_path is None or not Path(dataset_path).is_dir():
            raise ValueError('dataset_path must be an infinite-sites replicate directory; FASTA is retired')
        data = load_snp_dataset(dataset_path)
        params = json.loads((Path(dataset_path)/'metadata.json').read_text())['parameters']
        overrides = dict(population_size=effective_population_size, mutation_rate=mutation_rate,
                         recombination_rate=recombination_rate)
        rate_overrides = {k:v for k,v in overrides.items() if v is not None}
        rates = {k:params[k] if v is None else v for k,v in overrides.items()}
        env = SimpleARGEnvironment(snp_data=data, seed=seed, **rates)
        generator = GFlowNetGenerator(env, init_z_sample_count=init_z_sample_count, device=device,
                       policy_lr=policy_lr, flow_lr=flow_lr, grad_clip=grad_clip, subtb_lambda=subtb_lambda,
                       model_kwargs=model_kwargs, initialize_z_from_policy=False)
        config = TrajectoryMixConfig(replay_fraction=replay_fraction, replay_min_size=replay_min_size,
                                     exploration_fraction=exploration_fraction)
        run_config = dict(batch_size=batch_size, max_events=max_events, chunk_steps=chunk_steps, seed=seed,
                          dataset_path=str(Path(dataset_path).resolve()))
    config.validate()
    output = Path(output_path)
    if output.exists() and any(output.iterdir()) and not resume_checkpoint:
        raise ValueError('Fresh training requires an empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    generator.max_events = max_events
    worker = RolloutWorker(generator.env, max_events=max_events)
    trainer = Trainer(generator, worker, config, seed=seed, chunk_steps=chunk_steps)
    metadata = dict(run_config=run_config, rate_overrides=rate_overrides)
    try:
        if resume_checkpoint:
            trainer.load_state_dict(checkpoint['trainer'])
            restore_rng(generator.env, checkpoint['rng'])
        else:
            generator.initialize_flow_center()
        for step in range(trainer.completed_updates+1, epochs_num+1):
            info = trainer.train_epoch(batch_size=batch_size)
            with (output/'training.jsonl').open('a') as handle:
                handle.write(json.dumps(info, allow_nan=False)+'\n')
            print(json.dumps(info, allow_nan=False), flush=True)
            if step % checkpoint_every == 0 or step == epochs_num:
                generator.save(output/'checkpoints'/f'checkpoint_{step:04d}.pt', trainer=trainer,
                               metadata={**metadata, 'step':step})
        return generator, trainer
    except RolloutFailure as exc:
        (output/'failure.json').write_text(json.dumps(dict(error=str(exc), histories=exc.histories), indent=2))
        raise


def parse_train_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    for flag in ('dataset-path','output-path','device','resume-checkpoint'):
        parser.add_argument('--'+flag)
    for flag in ('epochs','batch-size','seed','init-z-sample-count','replay-min-size','max-events','chunk-steps','checkpoint-every','cpu-threads'):
        parser.add_argument('--'+flag, type=int)
    for flag in ('effective-population-size','mutation-rate','recombination-rate','policy-lr','flow-lr','grad-clip','subtb-lambda','replay-fraction','exploration-fraction'):
        parser.add_argument('--'+flag, type=float)
    args = vars(parser.parse_args(argv)); path = args.pop('config')
    config = yaml.safe_load(Path(path).read_text()) if path else {}
    config = {**(config or {}), **{k:v for k,v in args.items() if v is not None}}
    if 'epochs' in config:
        config['epochs_num'] = config.pop('epochs')
    model = {k:config.pop(k) for k in DEFAULT_MODEL if k in config}
    if model:
        config['model_kwargs'] = model
    config.setdefault('dataset_path', None)
    if 'output_path' not in config:
        parser.error('--output-path or output_path in the config is required')
    return config


if __name__ == '__main__':
    train(**parse_train_args())
