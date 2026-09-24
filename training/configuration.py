"""Explicit configuration contract for infinite-sites training and legacy YAML keys."""
import argparse
import math
from pathlib import Path
import yaml
from generator import DEFAULT_MODEL, FLOW_VERSION, validate_model_config
from .schedules import LearningRateConfig, PolicyTemperatureConfig
from .trainer import TrajectoryMixConfig

DEFAULTS = dict(
    dataset_path=None, output_path=None, device='cpu', epochs_num=20, batch_size=2, seed=7,
    effective_population_size=None, mutation_rate=None, recombination_rate=None, reward_C=3000.,
    policy_lr=1e-4, flow_lr=1e-3, grad_clip=10., subtb_lambda=.9, init_z_sample_count=8, init_z_batch_size=32,
    encoder_lr=None, flow_encoder_grad_scale=1., tb_loss_weight=0., flow_scale_mode='fixed',
    flow_encoder_mode='shared',
    replay_fraction=.25, replay_min_size=8, replay_capacity=2048, replay_grid_size=16,
    replay_per_topology=4, exploration_fraction=0., max_events=10000, chunk_steps=16,
    checkpoint_every=10, resume_checkpoint=None, cpu_threads=1, grad_accum_steps=1,
    verbose=True, wandb=False, wandb_project='ARG-Optimise', wandb_entity=None,
    wandb_name=None, wandb_mode='online',
    lr_schedule='constant', lr_schedule_steps=0, lr_warmup_steps=0,
    lr_warmup_start_factor=.1, lr_min_factor=.1,
    policy_temperature_schedule='constant', policy_temperature_start=1., policy_temperature_anneal_steps=0,
    eval_episodes=0, eval_every=50, eval_batch_size=2, eval_seed=100007,
    eval_async=False, eval_async_device='cuda', eval_async_shutdown_seconds=1800.,
    eval_initial=False, best_checkpoint_metric='eval_log_weight_std', max_wall_seconds=0.,
    eval_density_slope=True, eval_independent_likelihood=True,
    terminal_eval=False, terminal_eval_grid_size=100, terminal_eval_repeats=1,
    terminal_eval_repeat_every=250, tmrca_method='point_accuracy',
    arg_prior='hudson', bp_per_blocks=1, event_policy='cwr_residual', loss_type='subtb',
    flow_head_version=FLOW_VERSION, flow_warmup_steps=0, flow_warmup_episodes=128,
    time_policy='cwr_exponential', time_bins=32, time_delta_bin_width=.001,
    breakpoint_policy='sparse_mixture', breakpoint_hidden_dim=64,
    evaluation=None,
)
EVALUATION_DEFAULTS = dict(checkpoint='best_eval', metrics=['density_fit','ess','posterior_summary'],
    num_samples=1024, repeats=5, batch_size=32, seed=100007, device='auto', grid_size=100,
    rank_bins=20, bank_per_stratum=64, bank_candidates=768)


def lr_config(c):
    return LearningRateConfig(c['lr_schedule'],c['lr_schedule_steps'] or c['epochs_num'],
              c['lr_warmup_steps'],c['lr_warmup_start_factor'],c['lr_min_factor'])


def temperature_config(c):
    return PolicyTemperatureConfig(c['policy_temperature_schedule'],c['policy_temperature_start'],
                                    c['policy_temperature_anneal_steps'])


def mix_config(c):
    return TrajectoryMixConfig(**{key:c[key] for key in TrajectoryMixConfig.__dataclass_fields__})


def validate_evaluation(value):
    if value is None:
        return None
    if not isinstance(value,dict) or value.keys()-EVALUATION_DEFAULTS.keys():
        raise ValueError('Unknown evaluation configuration keys')
    result = {**EVALUATION_DEFAULTS,**value}
    if not isinstance(result['metrics'],list) or set(result['metrics'])-{'density_fit','ess','posterior_summary'}:
        raise ValueError('evaluation.metrics supports density_fit, ess, posterior_summary')
    for key in ('num_samples','repeats','batch_size','grid_size','rank_bins','bank_per_stratum','bank_candidates'):
        if isinstance(result[key],bool) or not isinstance(result[key],int) or result[key]<1:
            raise ValueError('evaluation.'+key+' must be a positive integer')
    if 'density_fit' in result['metrics'] and result['bank_candidates']<3*result['bank_per_stratum']:
        raise ValueError('evaluation.bank_candidates must be at least 3 * bank_per_stratum')
    return result


def resolve_config(options):
    options = dict(options)
    if 'epochs' in options:
        if 'epochs_num' in options:
            raise ValueError('Use one of epochs and epochs_num')
        options['epochs_num'] = options.pop('epochs')
    unknown = options.keys()-(DEFAULTS.keys()|DEFAULT_MODEL.keys()|{'model_kwargs'})
    if unknown:
        raise ValueError('Unsupported training settings: '+', '.join(sorted(unknown)))
    model = {**DEFAULT_MODEL,**(options.pop('model_kwargs',None) or {})}
    for key in DEFAULT_MODEL:
        if key in options:
            model[key] = options.pop(key)
    if model.keys()-DEFAULT_MODEL.keys():
        raise ValueError('Unknown model_kwargs')
    c = {**DEFAULTS,**options}
    problems = []
    for key,expected in dict(arg_prior='hudson',bp_per_blocks=1,event_policy='cwr_residual',
            loss_type='subtb',flow_head_version=FLOW_VERSION,time_policy='cwr_exponential',
            breakpoint_policy='sparse_mixture',flow_warmup_steps=0).items():
        if c[key] != expected:
            problems.append(f'{key}={c[key]!r} is retired for this workflow; use {expected!r}')
    try:
        validate_model_config(model)
    except ValueError as exc:
        problems.append(str(exc))
    if problems:
        raise ValueError('Infinite-sites configuration needs updating:\n- '+'\n- '.join(problems))
    for key in ('epochs_num','batch_size','cpu_threads','checkpoint_every','init_z_sample_count','init_z_batch_size',
                'max_events','chunk_steps','grad_accum_steps','eval_batch_size','terminal_eval_grid_size',
                'terminal_eval_repeats','terminal_eval_repeat_every','time_bins','flow_warmup_episodes',
                'breakpoint_hidden_dim'):
        if isinstance(c[key],bool) or not isinstance(c[key],int) or c[key]<1:
            raise ValueError(key+' must be a positive integer')
    for key in ('eval_every','eval_episodes'):
        if isinstance(c[key],bool) or not isinstance(c[key],int) or c[key]<0:
            raise ValueError(key+' must be a nonnegative integer')
    if c['grad_accum_steps']>c['batch_size']:
        raise ValueError('grad_accum_steps cannot exceed batch_size')
    if c['eval_episodes'] and not c['eval_every']:
        raise ValueError('eval_episodes requires eval_every > 0')
    if c['terminal_eval'] and not c['eval_episodes']:
        raise ValueError('terminal_eval requires eval_episodes > 0')
    for key in ('verbose','wandb','eval_density_slope','eval_independent_likelihood','terminal_eval','eval_initial','eval_async'):
        if not isinstance(c[key],bool):
            raise ValueError(key+' must be a YAML boolean')
    if c['eval_async'] and not c['eval_episodes']:
        raise ValueError('eval_async requires eval_episodes > 0')
    if c['eval_async_device'] != 'cpu' and c['eval_async_device'] != 'cuda' and not (
            isinstance(c['eval_async_device'], str) and c['eval_async_device'].startswith('cuda:')
            and c['eval_async_device'][5:].isdigit()):
        raise ValueError('eval_async_device must be cpu, cuda or cuda:N')
    if not math.isfinite(c['eval_async_shutdown_seconds']) or c['eval_async_shutdown_seconds'] < 0:
        raise ValueError('eval_async_shutdown_seconds must be finite and nonnegative')
    if c['best_checkpoint_metric'] not in ('eval_subtb_loss','eval_log_weight_std'):
        raise ValueError('best_checkpoint_metric must be eval_subtb_loss or eval_log_weight_std')
    if not math.isfinite(c['max_wall_seconds']) or c['max_wall_seconds'] < 0:
        raise ValueError('max_wall_seconds must be finite and nonnegative')
    for key in ('policy_lr','flow_lr','grad_clip','time_delta_bin_width'):
        if not math.isfinite(c[key]) or c[key]<=0:
            raise ValueError(key+' must be positive and finite')
    if c['encoder_lr'] is not None and (not math.isfinite(c['encoder_lr']) or c['encoder_lr'] <= 0):
        raise ValueError('encoder_lr must be positive and finite')
    if not math.isfinite(c['flow_encoder_grad_scale']) or not 0 <= c['flow_encoder_grad_scale'] <= 1:
        raise ValueError('flow_encoder_grad_scale must be in [0,1]')
    if c['flow_encoder_mode'] not in ('shared', 'frozen_initial'):
        raise ValueError('flow_encoder_mode must be shared or frozen_initial')
    if not math.isfinite(c['tb_loss_weight']) or c['tb_loss_weight'] < 0:
        raise ValueError('tb_loss_weight must be finite and nonnegative')
    if c['flow_scale_mode'] not in ('fixed','empirical'):
        raise ValueError('flow_scale_mode must be fixed or empirical')
    if not math.isfinite(c['reward_C']) or not math.isfinite(c['subtb_lambda']) or c['subtb_lambda']<0:
        raise ValueError('Invalid reward offset or SubTB lambda')
    if c['tmrca_method'] not in ('grid','point_accuracy'):
        raise ValueError('tmrca_method must be grid or point_accuracy')
    if c['wandb_mode'] not in ('online','offline','disabled'):
        raise ValueError('wandb_mode must be online, offline, or disabled')
    lr_config(c).validate(); mix_config(c).validate()
    temperature_config(c).validate_training(c['loss_type'],c['event_policy'],c['exploration_fraction'],
                                            c['replay_fraction'],c['flow_warmup_steps'])
    c['evaluation'] = validate_evaluation(c['evaluation'])
    c['model_kwargs'] = model
    return c


def config_notes(c):
    return dict(chunk_steps='Inactive: training uses direct SubTB backpropagation per microbatch without score recomputation.',
                init_z_batch_size='Concurrent initialization ARGs; all init_z_sample_count samples are retained. Changing this changes RNG interleaving.',
                time_bins='Inactive: continuous waiting times have no bins.',
                time_delta_bin_width='Inactive: continuous waiting times have no bin width.',
                breakpoint_hidden_dim='Inactive: width of the retired nucleotide CNN. Use breakpoint_mixture_hidden_dim.',
                flow_warmup_episodes='Inactive: head-only flow prefit is unsupported; flow_warmup_steps must be 0.',
                flow_encoder_mode='shared trains one encoder; frozen_initial gives the flow head a permanently fixed copy of the initial encoder.',
                flow_encoder_grad_scale=('Inactive with frozen_initial: flow gradients never reach the policy encoder.'
                    if c['flow_encoder_mode'] == 'frozen_initial' else 'Static multiplier on flow-to-shared-encoder gradients only.'),
                evaluation='Reserved for eval/eval.py --config; periodic training evaluation uses eval_* and terminal_eval_*.',
                breakpoint_mixture_layers='Number of hidden MLP layers on shared action/span features; no nucleotide CNN.',
                breakpoint_gap_layers='Additional parameter-head hidden layers, using breakpoint_gap_hidden_size.')


def parse_train_args(argv=None):
    parser = argparse.ArgumentParser(description='Train the infinite-sites GFlowNet')
    parser.add_argument('--config')
    for key,default in {**DEFAULTS,**DEFAULT_MODEL}.items():
        if key in ('evaluation','epochs_num'):
            continue
        flag = '--'+key.replace('_','-')
        if isinstance(default,bool):
            # Works on Python 3.9+, and makes --no-wandb an explicit override.
            parser.add_argument(flag,action=argparse.BooleanOptionalAction,default=None)
        else:
            kind = type(default) if default is not None else (int if key in
                    ('breakpoint_mixture_hidden_dim','time_hidden_dim') else float if key in
                    ('effective_population_size','mutation_rate','recombination_rate','encoder_lr') else str)
            parser.add_argument(flag,type=kind,default=None)
    parser.add_argument('--epochs',type=int)
    args = vars(parser.parse_args(argv)); path = args.pop('config')
    value = yaml.safe_load(Path(path).read_text()) if path else {}
    if value is not None and not isinstance(value,dict):
        parser.error('Configuration must be a YAML mapping')
    options = {**(value or {}),**{k:v for k,v in args.items() if v is not None}}
    # Validation stays in train(), allowing resume to recover saved effective options.
    return options
