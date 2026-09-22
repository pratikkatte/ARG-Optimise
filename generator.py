"""Infinite-sites GFlowNet with shared or permanently frozen flow encoding."""
import math
import time
from dataclasses import replace
import numpy as np
import torch
from torch import nn
from policy.encoder import InfiniteSitesEncoder, mlp
from policy.models import ARGModel
from policy.observations import pack_states, STATE_DIM, FEATURE_VERSION, RawObservationCache
from gfn.subtb import geometric_subtb_loss
from gfn.flow_encoder import FrozenFlowEncoder

MODEL_VERSION = 'infinite-sites-shared-v1'
FLOW_VERSION = 6
DEFAULT_MODEL = dict(embedding_size=64, hidden_size=128, transformer_depth=6,
                     transformer_heads=4, breakpoint_mixture_components=4,
                     transformer_mlp_ratio=2.0, dropout=0.0, attention_dropout=0.0,
                     breakpoint_mixture_hidden_dim=None, breakpoint_mixture_layers=1,
                     breakpoint_gap_hidden_size=64, breakpoint_gap_layers=0, breakpoint_dropout=0.0,
                     continuous_time_head='gamma', time_hidden_dim=None, time_layers=2,
                     time_mixture_components=4, time_parameterization='bounded_v1',
                     state_feature_transform='signed_log', initial_recombination_bias=0.)


def checkpoint_model_config(saved):
    """Missing versioned transforms mean the original checkpoint semantics."""
    return {**DEFAULT_MODEL, 'time_parameterization':'legacy',
            'state_feature_transform':'identity', **saved}


def validate_model_config(cfg):
    if cfg['time_parameterization'] not in ('legacy', 'bounded_v1'):
        raise ValueError('Unknown time_parameterization')
    if cfg['state_feature_transform'] not in ('identity', 'signed_log'):
        raise ValueError('Unknown state_feature_transform')
    if not math.isfinite(cfg['initial_recombination_bias']):
        raise ValueError('initial_recombination_bias must be finite')
    for key in ('embedding_size','hidden_size','transformer_depth','transformer_heads',
                'breakpoint_mixture_components','breakpoint_gap_hidden_size','time_mixture_components'):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(key+' must be a positive integer')
    for key in ('breakpoint_mixture_hidden_dim','time_hidden_dim'):
        if cfg[key] is not None and (isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1):
            raise ValueError(key+' must be a positive integer or null')
    for key in ('breakpoint_mixture_layers','breakpoint_gap_layers','time_layers'):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 0:
            raise ValueError(key+' must be a nonnegative integer')
    for key in ('dropout','attention_dropout','breakpoint_dropout'):
        if cfg[key] != 0:
            raise ValueError(key+' must be 0: exact policy replay and chunked gradients require deterministic scores')
    if cfg['continuous_time_head'] not in ('gamma','exponential','gamma_mixture'):
        raise ValueError('continuous_time_head must be gamma, exponential or gamma_mixture')
    if not math.isfinite(cfg['transformer_mlp_ratio']) or cfg['transformer_mlp_ratio'] <= 0:
        raise ValueError('transformer_mlp_ratio must be positive and finite')
    if cfg['embedding_size'] % cfg['transformer_heads']:
        raise ValueError('embedding_size must be divisible by transformer_heads')


class GFlowNetGenerator(nn.Module):
    def __init__(self, env, init_z_sample_count=8, *, device='cpu', model_kwargs=None,
                 policy_lr=1e-4, flow_lr=1e-3, grad_clip=10., subtb_lambda=.9,
                 initialize_z_from_policy=True, loss_type='subtb', flow_head_version=FLOW_VERSION,
                 flow_warmup_steps=0, verbose=False, encoder_lr=None,
                 flow_encoder_grad_scale=1., tb_loss_weight=0., flow_scale_mode='fixed',
                 flow_encoder_mode='shared'):
        super().__init__()
        if loss_type != 'subtb' or flow_head_version != FLOW_VERSION or flow_warmup_steps != 0:
            raise ValueError('Infinite sites requires SubTB v6 without head-only flow prefit')
        if flow_encoder_mode not in ('shared', 'frozen_initial'):
            raise ValueError('flow_encoder_mode must be shared or frozen_initial')
        self.flow_encoder_mode = flow_encoder_mode
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Float64 scoring requires CPU or CUDA; MPS is unsupported')
        if not math.isfinite(subtb_lambda) or subtb_lambda < 0:
            raise ValueError('subtb_lambda must be finite and nonnegative')
        if min(policy_lr, flow_lr, grad_clip) <= 0 or not all(map(math.isfinite, (policy_lr, flow_lr, grad_clip))):
            raise ValueError('Learning rates and gradient clipping must be finite and positive')
        self.env, self.loss_type, self.flow_head_version = env, loss_type, flow_head_version
        self.neural_source_flow = True
        self.subtb_lambda, self.grad_clip = float(subtb_lambda), float(grad_clip)
        self.policy_lr, self.flow_lr = float(policy_lr), float(flow_lr)
        if encoder_lr is not None and (not math.isfinite(encoder_lr) or encoder_lr <= 0):
            raise ValueError('encoder_lr must be positive and finite')
        if not math.isfinite(flow_encoder_grad_scale) or not 0 <= flow_encoder_grad_scale <= 1:
            raise ValueError('flow_encoder_grad_scale must be in [0,1]')
        if not math.isfinite(tb_loss_weight) or tb_loss_weight < 0:
            raise ValueError('tb_loss_weight must be finite and nonnegative')
        if flow_scale_mode not in ('fixed','empirical'):
            raise ValueError('flow_scale_mode must be fixed or empirical')
        self.encoder_lr = encoder_lr
        self.flow_encoder_grad_scale = float(flow_encoder_grad_scale)
        self.tb_loss_weight = float(tb_loss_weight)
        self.flow_scale_mode = flow_scale_mode
        self.init_z_sample_count = int(init_z_sample_count)
        self.model_kwargs = {**DEFAULT_MODEL, **(model_kwargs or {})}
        unknown = self.model_kwargs.keys()-DEFAULT_MODEL.keys()
        if unknown:
            raise ValueError('Unsupported or legacy model settings: '+', '.join(sorted(unknown)))
        cfg = self.model_kwargs
        validate_model_config(cfg)
        self.state_encoder = InfiniteSitesEncoder(env.num_sequences, **{k:cfg[k] for k in
                                ('embedding_size','hidden_size','transformer_depth','transformer_heads',
                                 'transformer_mlp_ratio','dropout','attention_dropout')})
        self.flow_encoder = (FrozenFlowEncoder(self.state_encoder)
                             if flow_encoder_mode == 'frozen_initial' else None)
        self.arg_model = ARGModel(**{k:cfg[k] for k in
                                 ('embedding_size','hidden_size','breakpoint_mixture_components',
                                  'breakpoint_mixture_hidden_dim','breakpoint_mixture_layers',
                                  'breakpoint_gap_hidden_size','breakpoint_gap_layers',
                                  'continuous_time_head','time_hidden_dim','time_layers','time_mixture_components',
                                  'time_parameterization')})
        # Initialization only; the normalized residual policy retains full
        # compatible support and the physical target prior is never modified.
        with torch.no_grad():
            self.arg_model.event_head[-1].bias[1] = cfg['initial_recombination_bias']
        self.flow_head = mlp(cfg['embedding_size']+STATE_DIM, cfg['hidden_size'], 1)
        nn.init.zeros_(self.flow_head[-1].weight); nn.init.zeros_(self.flow_head[-1].bias)
        self.register_buffer('flow_init_offset', torch.tensor(env.reward_fn.C, dtype=torch.float64))
        self.register_buffer('flow_output_scale', torch.tensor(1., dtype=torch.float64))
        self.to(self.device)
        groups = [
            {'params': (list(self.state_encoder.parameters()) if encoder_lr is None else [])+
                       list(self.arg_model.parameters()), 'lr':policy_lr},
            {'params': self.flow_head.parameters(), 'lr':flow_lr}]
        if encoder_lr is not None:
            groups.append({'params': self.state_encoder.parameters(), 'lr':encoder_lr})
        self.opt = torch.optim.Adam(groups)
        self.scheduler = None
        self._observation_cache = RawObservationCache()
        self.max_events = 10000
        if initialize_z_from_policy:
            self.initialize_flow_center(verbose=verbose)

    def encode(self, states, *, pooled_cache=None):
        return self._encode(states, self.state_encoder, pooled_cache=pooled_cache)

    def _encode(self, states, encoder, *, pooled_cache=None):
        # Each branch prepares missing raw rows against its own learned cache.
        # Only the nonlearned raw observation cache is shared between branches.
        nodes = [node for state in states for node in state.active_lineages] if pooled_cache is not None else None
        plan = pooled_cache.prepare(nodes) if pooled_cache is not None else None
        batch = pack_states(self.env, states, self.device, cache=self._observation_cache,
                            static_rows=None if plan is None else plan.missing)
        if self.model_kwargs['state_feature_transform'] == 'signed_log':
            scalars = batch.observations.state_scalars
            # These two potentially unbounded inputs previously overwhelmed
            # the summary/time networks on rare long histories. No scientific
            # likelihood, prior, target, or state is transformed.
            stable = torch.cat((scalars[:, :5],
                scalars[:, 5:7].sign()*scalars[:, 5:7].abs().log1p(), scalars[:, 7:]), -1)
            batch = replace(batch, observations=replace(batch.observations, state_scalars=stable))
        pooled = None
        if pooled_cache is not None:
            pooled = pooled_cache.get(encoder, batch.observations, nodes, plan=plan)
        lineage, summary = encoder(batch.observations, pooled_embeddings=pooled)
        return batch, lineage, summary

    def state_flows(self, states, summary=None, observations=None, *, flow_pooled_cache=None):
        if self.flow_encoder is not None:
            batch, _, summary = self._encode(states, self.flow_encoder, pooled_cache=flow_pooled_cache)
            observations = batch.observations
        else:
            if summary is None or observations is None:
                batch, _, summary = self.encode(states)
                observations = batch.observations
            # This control applies only to the shared trainable encoder.
            summary = summary.detach()+self.flow_encoder_grad_scale*(summary-summary.detach())
        residual = self.flow_head(torch.cat((summary, observations.state_scalars), -1)).squeeze(-1).double()
        prior = residual.new_tensor([s.accumulated_log_prior for s in states])
        potential = residual.new_tensor([s.partial_log_likelihood for s in states])
        remaining = residual.new_tensor([(s.total_active_blocks/self.env.sequence_length-1)/(self.env.num_sequences-1)
                                          for s in states])
        value = (self.env.reward_fn.C+prior+potential+
                 remaining*(self.flow_init_offset-self.env.reward_fn.C)+self.flow_output_scale*residual)
        terminal = torch.tensor([s.is_done for s in states], device=self.device)
        reward = value.new_tensor([s.log_reward if s.is_done else 0. for s in states])
        return torch.where(terminal, reward, value)

    def forward(self, states, *, forced_actions=None, return_flows=False, temperature=1.0,
                pooled_cache=None, flow_pooled_cache=None):
        if pooled_cache is not None and flow_pooled_cache is pooled_cache:
            raise ValueError('Policy and flow encoders require separate pooled caches')
        batch, lineages, summary = self.encode(states, pooled_cache=pooled_cache)
        log_pf, actions, factors = self.arg_model(self.env, states, batch, lineages, summary, forced_actions, temperature)
        flow = self.state_flows(states, summary, batch.observations,
                               flow_pooled_cache=flow_pooled_cache) if return_flows else None
        return dict(log_pf=log_pf, actions=actions, factors=factors, flows=flow)

    def compute_log_Z(self):
        states = [self.env.get_initial_state()]
        return self.state_flows(states)[0]

    def compute_event_probabilities(self, state):
        if state.is_done:
            return {'coal':0., 'recomb':0.}
        batch, _, summary = self.encode([state])
        values = self.arg_model.event_log_probs(batch, summary)[0].exp()
        return dict(zip(('coal','recomb'), values))

    @torch.no_grad()
    def initialize_flow_center(self, batch_size=32, *, verbose=False):
        """Estimate the center from every sampled ARG in bounded batches.

        Batch size changes RNG interleaving, but not the proposal distribution
        or the per-trajectory reward-minus-log-proposal estimator.
        """
        from gfn.rollout import RolloutWorker
        if self.init_z_sample_count < 1:
            raise ValueError('Flow initialization requires at least one trajectory')
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('Initialization batch_size must be a positive integer')
        targets = []
        worker = RolloutWorker(self.env, max_events=self.max_events)
        for index in range(0, self.init_z_sample_count, batch_size):
            started = time.perf_counter()
            count = min(batch_size, self.init_z_sample_count-index)
            outputs, _ = worker.rollout(self, episodes=count)
            targets.append(outputs['log_rewards']-outputs['log_paths_pf'].sum(-1))
            if verbose:
                mean_events = outputs['lengths'].double().mean().item()
                print(f'Z init {index+count}/{self.init_z_sample_count} | '
                      f'events/ARG={mean_events:.1f} | {time.perf_counter()-started:.1f}s', flush=True)
        values = torch.cat(targets)
        self.flow_init_offset.copy_(values.mean())
        self.flow_output_scale.copy_(values.std(unbiased=False).clamp_min(1.)
                                    if self.flow_scale_mode == 'empirical' else values.new_tensor(1.))

    def loss_components(self, outputs):
        subtb = geometric_subtb_loss(outputs['log_paths_pf'], outputs['log_paths_pb'],
                    outputs['state_flows'], outputs['lengths'], outputs['log_rewards'], self.subtb_lambda)
        mask = torch.arange(outputs['log_paths_pf'].shape[1],device=self.device)[None,:]<outputs['lengths'][:,None]
        forward = torch.where(mask,outputs['log_paths_pf'],0.).sum(-1)
        backward = torch.where(mask,outputs['log_paths_pb'],0.).sum(-1)
        residual = outputs['state_flows'][:, 0]+forward-backward-outputs['log_rewards']
        tb = residual.square().mean()
        return dict(subtb=subtb,tb=tb,total=subtb+self.tb_loss_weight*tb)

    def get_loss_from_rollout_outputs(self, outputs):
        return self.loss_components(outputs)['total']

    def count_backward_parents(self, state):
        if not state.actions:
            return 0
        latest = [node for node in state.all_nodes.values() if node.time == state.current_time]
        expected = 2 if state.actions[-1].event_type == 'recomb' else 1
        if len(latest) != expected or not all(n.node_id in {a.node_id for a in state.active_lineages} for n in latest):
            raise ValueError('State has no unique latest chronological event')
        return 1

    def predecessor(self, state):
        restored = self.env.restore_state(state)
        if not restored.actions:
            raise ValueError('Source has no predecessor')
        return self.env.replay(restored.actions[:-1]), restored.actions[-1]

    def save(self, path, metadata=None, trainer=None):
        from training.checkpoints import save_checkpoint
        return save_checkpoint(path, self, trainer=trainer, metadata=metadata)

    def load(self, path, load_optimizer=True, map_location=None):
        from training.checkpoints import load_checkpoint, restore_generator
        data = load_checkpoint(path) if not isinstance(path, dict) else path
        restore_generator(self, data, load_optimizer=load_optimizer)
        return data['metadata']


TBGFlowNetGenerator = GFlowNetGenerator
