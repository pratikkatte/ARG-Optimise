"""Shared-encoder infinite-sites GFlowNet. All scientific state lives in env."""
import math
import numpy as np
import torch
from torch import nn
from policy.encoder import InfiniteSitesEncoder, mlp
from policy.models import ARGModel
from policy.observations import pack_states, STATE_DIM, FEATURE_VERSION, RawObservationCache
from gfn.subtb import geometric_subtb_loss

MODEL_VERSION = 'infinite-sites-shared-v1'
FLOW_VERSION = 6
DEFAULT_MODEL = dict(embedding_size=64, hidden_size=128, transformer_depth=6,
                     transformer_heads=4, breakpoint_mixture_components=4,
                     transformer_mlp_ratio=2.0, dropout=0.0, attention_dropout=0.0,
                     breakpoint_mixture_hidden_dim=None, breakpoint_mixture_layers=1,
                     breakpoint_gap_hidden_size=64, breakpoint_gap_layers=0, breakpoint_dropout=0.0,
                     continuous_time_head='gamma', time_hidden_dim=None, time_layers=2)


def validate_model_config(cfg):
    for key in ('embedding_size','hidden_size','transformer_depth','transformer_heads',
                'breakpoint_mixture_components','breakpoint_gap_hidden_size'):
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
    if cfg['continuous_time_head'] not in ('gamma','exponential'):
        raise ValueError('continuous_time_head must be gamma or exponential')
    if not math.isfinite(cfg['transformer_mlp_ratio']) or cfg['transformer_mlp_ratio'] <= 0:
        raise ValueError('transformer_mlp_ratio must be positive and finite')
    if cfg['embedding_size'] % cfg['transformer_heads']:
        raise ValueError('embedding_size must be divisible by transformer_heads')


class GFlowNetGenerator(nn.Module):
    def __init__(self, env, init_z_sample_count=8, *, device='cpu', model_kwargs=None,
                 policy_lr=1e-4, flow_lr=1e-3, grad_clip=10., subtb_lambda=.9,
                 initialize_z_from_policy=True, loss_type='subtb', flow_head_version=FLOW_VERSION,
                 flow_warmup_steps=0, verbose=False):
        super().__init__()
        if loss_type != 'subtb' or flow_head_version != FLOW_VERSION or flow_warmup_steps != 0:
            raise ValueError('Infinite sites requires shared-encoder SubTB v6 without cached flow warm-up')
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
        self.arg_model = ARGModel(**{k:cfg[k] for k in
                                 ('embedding_size','hidden_size','breakpoint_mixture_components',
                                  'breakpoint_mixture_hidden_dim','breakpoint_mixture_layers',
                                  'breakpoint_gap_hidden_size','breakpoint_gap_layers',
                                  'continuous_time_head','time_hidden_dim','time_layers')})
        self.flow_head = mlp(cfg['embedding_size']+STATE_DIM, cfg['hidden_size'], 1)
        nn.init.zeros_(self.flow_head[-1].weight); nn.init.zeros_(self.flow_head[-1].bias)
        self.register_buffer('flow_init_offset', torch.tensor(env.reward_fn.C, dtype=torch.float64))
        self.register_buffer('flow_output_scale', torch.tensor(1., dtype=torch.float64))
        self.to(self.device)
        self.opt = torch.optim.Adam([
            {'params': list(self.state_encoder.parameters())+list(self.arg_model.parameters()), 'lr':policy_lr},
            {'params': self.flow_head.parameters(), 'lr':flow_lr}])
        self.scheduler = None
        self._observation_cache = RawObservationCache()
        self.max_events = 10000
        if initialize_z_from_policy:
            self.initialize_flow_center()

    def encode(self, states):
        batch = pack_states(self.env, states, self.device, cache=self._observation_cache)
        lineage, summary = self.state_encoder(batch.observations)
        return batch, lineage, summary

    def state_flows(self, states, summary, observations):
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

    def forward(self, states, *, forced_actions=None, return_flows=False, temperature=1.0):
        batch, lineages, summary = self.encode(states)
        log_pf, actions, factors = self.arg_model(self.env, states, batch, lineages, summary, forced_actions, temperature)
        flow = self.state_flows(states, summary, batch.observations) if return_flows else None
        return dict(log_pf=log_pf, actions=actions, factors=factors, flows=flow)

    def compute_log_Z(self):
        states = [self.env.get_initial_state()]
        batch, _, summary = self.encode(states)
        return self.state_flows(states, summary, batch.observations)[0]

    def compute_event_probabilities(self, state):
        if state.is_done:
            return {'coal':0., 'recomb':0.}
        batch, _, summary = self.encode([state])
        values = self.arg_model.event_log_probs(batch, summary)[0].exp()
        return dict(zip(('coal','recomb'), values))

    @torch.no_grad()
    def initialize_flow_center(self):
        from gfn.rollout import RolloutWorker
        if self.init_z_sample_count < 1:
            raise ValueError('Flow initialization requires at least one trajectory')
        progress = getattr(self, 'progress_reporter', None)
        if progress is not None:
            progress.begin('flow_initialization', initialized=0, initialization_total=self.init_z_sample_count,
                           neural_device=str(self.device), environment_device=self.env.device)
        targets = []
        for index in range(self.init_z_sample_count):
            if progress is not None:
                progress.update(current_arg=index+1, events_max=0, batch_completed=0, batch_total=1)
            outputs, paths = RolloutWorker(self.env, max_events=self.max_events).rollout(self)
            targets.append(outputs['log_rewards'][0]-outputs['log_paths_pf'][0].sum())
            if progress is not None:
                completed = index+1
                progress.update(initialized=completed, events_max=len(paths[0]), batch_completed=1,
                                active_lineages_max=0,
                                force=completed == 1 or completed % 10 == 0 or completed == self.init_z_sample_count)
        values = torch.stack(targets)
        self.flow_init_offset.copy_(values.mean())
        self.flow_output_scale.copy_(values.std(unbiased=False).clamp_min(1.))
        if progress is not None:
            progress.update(force=True, status='complete', initialized=self.init_z_sample_count)

    def get_loss_from_rollout_outputs(self, outputs):
        return geometric_subtb_loss(outputs['log_paths_pf'], outputs['log_paths_pb'],
                    outputs['state_flows'], outputs['lengths'], outputs['log_rewards'], self.subtb_lambda)

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
