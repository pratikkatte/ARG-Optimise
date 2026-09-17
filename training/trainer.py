"""Current-policy SubTB scoring for fresh, compatible-proposal, and replay paths."""
from dataclasses import dataclass, asdict
import math
import torch
from env.env import SimpleTrajectory
from .trajectories import DiverseTrajectoryBuffer


@dataclass(frozen=True)
class TrajectoryMixConfig:
    exploration_fraction: float = 0.
    replay_fraction: float = .25
    replay_capacity: int = 2048
    replay_grid_size: int = 16
    replay_per_topology: int = 4
    replay_min_size: int = 8

    def validate(self):
        if not all(math.isfinite(x) and 0 <= x < 1 for x in (self.exploration_fraction,self.replay_fraction)):
            raise ValueError('Invalid trajectory mixture fractions')
        if self.exploration_fraction+self.replay_fraction >= 1:
            raise ValueError('Keep a fresh policy fraction')
        if self.replay_capacity < 2 or not 1 <= self.replay_min_size <= self.replay_capacity:
            raise ValueError('Invalid replay capacity or threshold')


@torch.no_grad()
def sample_compatible_trajectories(env, episodes, max_events=10000):
    paths = []
    from gfn.rollout import RolloutFailure
    for _ in range(episodes):
        state, path = env.get_initial_state(), SimpleTrajectory()
        while not state.is_done:
            try:
                if len(path) >= max_events:
                    raise ValueError('Compatible proposal exceeded ARG event limit')
                step = env.sample_compatible_step(state)
                state = env.apply_action(state, step.action, step.log_prior)
                path.update(step.action, log_prior=step.log_prior, log_proposal=step.log_proposal,
                            log_reward=state.log_reward)
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                raise RolloutFailure(str(exc), paths+[path]) from exc
        paths.append(path)
    return paths


def trajectory_length_statistics(lengths, prefix=''):
    values = torch.as_tensor(lengths, dtype=torch.float64)
    return {prefix+'trajectory_length_'+key: float(value) for key,value in
            [('median',values.quantile(.5)),('p95',values.quantile(.95)),('max',values.max())]}


class Trainer:
    schema_version = 2

    def __init__(self, generator, worker, mix_config=None, seed=7, chunk_steps=16):
        self.generator, self.worker = generator, worker
        self.config = mix_config or TrajectoryMixConfig()
        self.config.validate()
        self.chunk_steps = chunk_steps
        self.buffer = (DiverseTrajectoryBuffer(generator.env, capacity=self.config.replay_capacity,
                    grid_size=self.config.replay_grid_size, per_topology=self.config.replay_per_topology,
                    seed=seed+700001) if self.config.replay_fraction else None)
        self.completed_updates = 0

    @staticmethod
    def allocated(batch_size, fraction, step):
        return round(batch_size*fraction*step)-round(batch_size*fraction*(step-1))

    def train_epoch(self, step=None, batch_size=2, grad_accum_steps=1):
        if grad_accum_steps != 1:
            raise ValueError('Use bounded score chunks; grad_accum_steps must be 1')
        step = self.completed_updates+1 if step is None else step
        if step != self.completed_updates+1:
            raise ValueError('Training updates must be consecutive')
        exploration = self.allocated(batch_size, self.config.exploration_fraction, step)
        replay = (self.allocated(batch_size,self.config.replay_fraction,step)
                  if self.buffer is not None and len(self.buffer) >= self.config.replay_min_size else 0)
        fresh = batch_size-exploration-replay
        if fresh < 1:
            raise ValueError('Batch allocation must retain a fresh policy trajectory')
        g = self.generator
        with torch.no_grad():
            _, paths = self.worker.rollout(g, fresh)
            sources = ['policy']*fresh
            if exploration:
                paths += sample_compatible_trajectories(g.env, exploration, self.worker.max_events)
                sources += ['compatible_proposal']*exploration
            if replay:
                for entry in self.buffer.sample(replay):
                    path = SimpleTrajectory(); path.actions = entry.actions()
                    paths.append(path); sources.append('replay')
            outputs, rescored = self.worker.replay(g, paths, return_states=True)
        # Differentiate the exact scalar loss with respect to its scores, then
        # recompute the neural score Jacobians in bounded-memory chunks.
        detached = dict(outputs)
        detached['log_paths_pf'] = outputs['log_paths_pf'].detach().requires_grad_()
        detached['state_flows'] = outputs['state_flows'].detach().requires_grad_()
        loss = g.get_loss_from_rollout_outputs(detached)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite SubTB loss')
        pf_weights, flow_weights = torch.autograd.grad(loss, (detached['log_paths_pf'],detached['state_flows']))
        g.opt.zero_grad(set_to_none=True)
        self.worker.backward_scores(g, paths, pf_weights, flow_weights, self.chunk_steps)
        norm = torch.nn.utils.clip_grad_norm_(g.parameters(), g.grad_clip, error_if_nonfinite=True)
        g.opt.step()
        if g.scheduler is not None:
            g.scheduler.step()
        if self.buffer is not None:
            for source, original, path, state in zip(sources, paths, rescored, outputs['states']):
                if source != 'replay':
                    path.log_proposals = original.log_proposals
                    self.buffer.add(g.env, path, state, source, step)
        self.completed_updates = step
        return dict(step=step, loss=float(loss.detach()), grad_norm=float(norm), fresh=fresh,
                    compatible_proposal=exploration, replay=replay,
                    log_reward_mean=float(outputs['log_rewards'].mean()),
                    mean_events=float(outputs['lengths'].float().mean()),
                    **trajectory_length_statistics(outputs['lengths']),
                    **(self.buffer.metrics() if self.buffer is not None else {}))

    def state_dict(self):
        return dict(schema_version=self.schema_version, config=asdict(self.config),
                    completed_updates=self.completed_updates, chunk_steps=self.chunk_steps,
                    replay=self.buffer.state_dict() if self.buffer is not None else None)

    def load_state_dict(self, data):
        if data.get('schema_version') != self.schema_version or data['config'] != asdict(self.config):
            raise ValueError('Incompatible infinite-sites trainer checkpoint')
        if data['chunk_steps'] != self.chunk_steps:
            raise ValueError('Score chunk size changed on resume')
        self.completed_updates = int(data['completed_updates'])
        if self.buffer is not None:
            self.buffer.load_state_dict(data['replay'])
