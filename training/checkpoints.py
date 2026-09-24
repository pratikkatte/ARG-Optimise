"""Self-contained infinite-sites checkpoints; legacy schemas never migrate silently."""
import random
from pathlib import Path
import numpy as np
import torch
from env.snp_data import SNPData
from env.env import SimpleARGEnvironment
from policy.observations import FEATURE_VERSION

SCHEMA_VERSION = 2


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(env):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], environment=env.rng.getstate())


def restore_rng(env, data):
    random.setstate(data['python']); np.random.set_state(data['numpy']); torch.set_rng_state(data['torch'].cpu())
    if data['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(data['cuda'])
    env.rng.setstate(data['environment'])


def environment_metadata(env):
    d = env.snp_data
    return dict(mutation_model='infinite_sites', feature_version=FEATURE_VERSION,
                environment_fingerprint=env.dataset_fingerprint,
                observations=dict(genotypes=torch.tensor(d.genotypes.copy(), dtype=torch.uint8),
                    positions=torch.tensor(d.positions.copy(), dtype=torch.float64), sequence_length=d.sequence_length,
                    site_ids=d.site_ids, ancestral_states=d.ancestral_states, derived_states=d.derived_states,
                    haplotype_ids=d.haplotype_ids, contig_id=d.contig_id,
                    observation_intervals=d.observation_intervals),
                environment=dict(population_size=env.population_size, mutation_rate=env.mutation_rate,
                    recombination_rate=env.recombination_rate, reward_C=env.reward_fn.C,
                    bp_per_blocks=1, time_policy='cwr_exponential', arg_prior='hudson'))


def validate_metadata(metadata):
    from generator import MODEL_VERSION, FLOW_VERSION
    if (metadata.get('model_version') != MODEL_VERSION or metadata.get('mutation_model') != 'infinite_sites'
            or metadata.get('feature_version') != FEATURE_VERSION or metadata.get('flow_head_version') != FLOW_VERSION):
        raise ValueError('Incompatible checkpoint: an infinite-sites v6 checkpoint is required; JC69 is retired')
    required = {'observations','environment','environment_fingerprint','model','generator_config'}
    if not required <= metadata.keys():
        raise ValueError('Incomplete infinite-sites checkpoint metadata')
    mode = metadata['generator_config'].get('flow_encoder_mode', 'shared')
    if mode not in ('shared', 'frozen_initial'):
        raise ValueError('Checkpoint flow_encoder_mode must be shared or frozen_initial')
    retired = {'flow_encoder_warmup_steps', 'flow_encoder_ramp_steps'}
    for key in ('generator_config', 'resolved_config', 'run_config'):
        config = metadata.get(key) or {}
        if retired.intersection(config):
            raise ValueError('Discarded flow-gradient warm-up checkpoint cannot be loaded; start a fresh run')
    resolved = metadata.get('resolved_config')
    if resolved is not None and resolved.get('flow_encoder_mode', 'shared') != mode:
        raise ValueError('Checkpoint encoder mode disagrees with resolved configuration')


def validate_checkpoint(data):
    if not isinstance(data, dict) or data.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Incompatible checkpoint schema; JC69 checkpoints cannot be migrated')
    validate_metadata(data.get('metadata', {}))
    if 'flow_encoder_gradient' in (data.get('trainer') or {}):
        raise ValueError('Discarded flow-gradient warm-up checkpoint cannot be loaded; start a fresh run')


def environment_from_metadata(metadata, seed=7, device=None):
    validate_metadata(metadata)
    payload = dict(metadata['observations'])
    for key in ('genotypes','positions'):
        value = payload[key]
        payload[key] = value.cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    data = SNPData(**payload)
    env = SimpleARGEnvironment(snp_data=data, seed=seed, **metadata['environment'])
    if env.dataset_fingerprint != metadata['environment_fingerprint']:
        raise ValueError('Checkpoint observation/environment fingerprint mismatch')
    return env


def save_checkpoint(path, generator, trainer=None, metadata=None):
    from generator import MODEL_VERSION, FLOW_VERSION
    meta = {**(metadata or {}), **environment_metadata(generator.env),
            'model_version':MODEL_VERSION, 'flow_head_version':FLOW_VERSION,
            'model':generator.model_kwargs,
            'generator_config':dict(policy_lr=generator.policy_lr, flow_lr=generator.flow_lr,
                                   grad_clip=generator.grad_clip, subtb_lambda=generator.subtb_lambda,
                                   init_z_sample_count=generator.init_z_sample_count,
                                   encoder_lr=generator.encoder_lr,
                                   flow_encoder_mode=generator.flow_encoder_mode,
                                   flow_encoder_grad_scale=generator.flow_encoder_grad_scale,
                                   tb_loss_weight=generator.tb_loss_weight,
                                   flow_scale_mode=generator.flow_scale_mode)}
    data = dict(schema_version=SCHEMA_VERSION, metadata=meta, generator_state_dict=generator.state_dict(),
                opt_state_dict=generator.opt.state_dict(), rng=rng_state(generator.env),
                scheduler=generator.scheduler.state_dict() if generator.scheduler is not None else None,
                trainer=trainer.state_dict() if trainer is not None else None)
    validate_checkpoint(data)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(data, temporary); temporary.replace(path)
    return data


def load_checkpoint(path, map_location='cpu'):
    data = torch.load(path, map_location=map_location, weights_only=False)
    validate_checkpoint(data)
    return data


def restore_generator(generator, data, load_optimizer=True):
    validate_checkpoint(data)
    metadata = data['metadata']
    if metadata['generator_config'].get('flow_encoder_mode', 'shared') != generator.flow_encoder_mode:
        raise ValueError('Checkpoint flow_encoder_mode differs; cross-mode loading is unsupported')
    if metadata['environment_fingerprint'] != generator.env.dataset_fingerprint:
        raise ValueError('Checkpoint observations or environment differ')
    from generator import checkpoint_model_config
    if checkpoint_model_config(metadata['model']) != generator.model_kwargs:
        raise ValueError('Checkpoint neural architecture differs')
    generator.load_state_dict(data['generator_state_dict'], strict=True)
    if load_optimizer:
        generator.opt.load_state_dict(data['opt_state_dict'])
        if data['scheduler'] is not None:
            if 'config' in data['scheduler'] and 'completed_updates' in data['scheduler']:
                from training.schedules import WarmupCosineScheduler
                generator.scheduler = WarmupCosineScheduler.from_state_dict(generator.opt, data['scheduler'])
            else:
                generator.scheduler = torch.optim.lr_scheduler.ExponentialLR(generator.opt, gamma=data['scheduler']['gamma'])
                generator.scheduler.load_state_dict(data['scheduler'])


def generator_from_checkpoint(data, device='cpu', seed=7, optimizer=False, restore_random=False):
    from generator import GFlowNetGenerator, checkpoint_model_config
    validate_checkpoint(data)
    env = environment_from_metadata(data['metadata'], seed)
    generator = GFlowNetGenerator(env, device=device, model_kwargs=checkpoint_model_config(data['metadata']['model']),
                 initialize_z_from_policy=False, **{'flow_scale_mode':'empirical',
                                                   **data['metadata']['generator_config']})
    restore_generator(generator, data, optimizer)
    if restore_random:
        restore_rng(env, data['rng'])
    return generator
