"""Current-policy SubTB scoring for fresh, compatible-proposal, and replay paths."""
from dataclasses import dataclass, asdict
import math
import torch
from env.env import SimpleTrajectory
from .trajectories import DiverseTrajectoryBuffer
from .schedules import PolicyTemperatureConfig


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
        for name in ('replay_capacity','replay_grid_size','replay_per_topology','replay_min_size'):
            if isinstance(getattr(self,name),bool) or not isinstance(getattr(self,name),int) or getattr(self,name)<1:
                raise ValueError(name+' must be a positive integer')
        if self.replay_capacity < 2 or not 1 <= self.replay_min_size <= self.replay_capacity:
            raise ValueError('Invalid replay capacity or threshold')


@torch.no_grad()
def sample_compatible_trajectories(env, episodes, max_events=10000, progress=None):
    paths = []
    from gfn.rollout import RolloutFailure
    for _ in range(episodes):
        state, path = env.get_initial_state(), SimpleTrajectory()
        if progress is not None:
            progress.update(proposal_completed=len(paths), proposal_total=episodes, events_max=0)
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
            if progress is not None and progress.due():
                progress.update(force=True, events_max=len(path),
                                active_lineages_max=len(state.active_lineages))
        paths.append(path)
        if progress is not None:
            progress.update(proposal_completed=len(paths), events_max=len(path))
    return paths


def trajectory_length_statistics(lengths, prefix=''):
    values = torch.as_tensor(lengths, dtype=torch.float64)
    return {prefix+'trajectory_length_'+key: float(value) for key,value in
            [('median',values.quantile(.5)),('p95',values.quantile(.95)),('max',values.max())]}


class Trainer:
    schema_version = 2

    def __init__(self, generator, worker, mix_config=None, seed=7, chunk_steps=16, temperature_config=None):
        self.generator, self.worker = generator, worker
        self.config = mix_config or TrajectoryMixConfig()
        self.config.validate()
        self.chunk_steps = chunk_steps
        self.buffer = (DiverseTrajectoryBuffer(generator.env, capacity=self.config.replay_capacity,
                    grid_size=self.config.replay_grid_size, per_topology=self.config.replay_per_topology,
                    seed=seed+700001) if self.config.replay_fraction else None)
        self.completed_updates = 0
        self.temperature_config = temperature_config or PolicyTemperatureConfig()
        self.temperature_config.validate_training('subtb', 'cwr_residual',
                    self.config.exploration_fraction, self.config.replay_fraction)

    @staticmethod
    def allocated(batch_size, fraction, step):
        return round(batch_size*fraction*step)-round(batch_size*fraction*(step-1))

    def train_epoch(self, step=None, batch_size=2, grad_accum_steps=1):
        if (isinstance(grad_accum_steps,bool) or not isinstance(grad_accum_steps,int)
                or not 1 <= grad_accum_steps <= batch_size):
            raise ValueError('grad_accum_steps must be an integer from 1 to batch_size')
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
        progress = getattr(g, 'progress_reporter', None)
        temperature = self.temperature_config.temperature(self.completed_updates)
        spec = self.temperature_config.random_spec(self.completed_updates)
        micro_size = math.ceil(batch_size/grad_accum_steps)
        if progress is not None:
            progress.begin('train_sampling', step=step, fresh_completed=0, fresh_total=fresh,
                           exploration=exploration, replay=replay, microbatch_size=micro_size)
        with torch.no_grad():
            paths = []
            for start in range(0,fresh,micro_size):
                if progress is not None:
                    progress.update(microbatch=start//micro_size+1,
                                    microbatches=math.ceil(fresh/micro_size))
                _, new = self.worker.rollout(g, min(micro_size,fresh-start), random_spec=spec)
                paths.extend(new)
                if progress is not None:
                    progress.update(force=True, fresh_completed=len(paths),
                                    batch_completed=len(new), batch_total=len(new),
                                    events_max=max(len(path) for path in new), active_lineages_max=0)
            sources = ['policy']*fresh
            if exploration:
                if progress is not None:
                    progress.begin('train_exploration', step=step, proposal_total=exploration)
                paths += sample_compatible_trajectories(g.env, exploration, self.worker.max_events, progress=progress)
                sources += ['compatible_proposal']*exploration
            if replay:
                if progress is not None:
                    progress.begin('replay_selection', step=step, trajectories=replay)
                for entry in self.buffer.sample(replay):
                    path = SimpleTrajectory(); path.actions = entry.actions()
                    paths.append(path); sources.append('replay')
        # Each microbatch contributes its fraction of the full-batch mean.
        # One clip, optimizer update, and scheduler step follow all microbatches.
        g.opt.zero_grad(set_to_none=True)
        loss_value, rewards, lengths, retained = 0., [], [], []
        for start in range(0,batch_size,micro_size):
            subset = paths[start:start+micro_size]
            context = dict(step=step, microbatch=start//micro_size+1,
                           microbatches=math.ceil(batch_size/micro_size), trajectories=len(subset))
            if progress is not None:
                progress.begin('train_scoring', **context)
            with torch.no_grad():
                outputs, rescored = self.worker.replay(g, subset, return_states=True)
            detached = dict(outputs)
            detached['log_paths_pf'] = outputs['log_paths_pf'].detach().requires_grad_()
            detached['state_flows'] = outputs['state_flows'].detach().requires_grad_()
            loss = g.get_loss_from_rollout_outputs(detached)*(len(subset)/batch_size)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite SubTB loss')
            weights = torch.autograd.grad(loss, (detached['log_paths_pf'],detached['state_flows']))
            if progress is not None:
                progress.begin('train_backward', **context)
            self.worker.backward_scores(g, subset, *weights, self.chunk_steps)
            loss_value += float(loss.detach())
            rewards.extend(outputs['log_rewards'].tolist()); lengths.extend(outputs['lengths'].tolist())
            retained.extend(zip(sources[start:start+micro_size], subset, rescored, outputs['states']))
        if progress is not None:
            progress.begin('optimizer_update', step=step)
        group_norms = {}
        for name, parameters in (('encoder',g.state_encoder.parameters()),
                                 ('policy',g.arg_model.parameters()),('flow',g.flow_head.parameters())):
            squares = [p.grad.detach().double().square().sum() for p in parameters if p.grad is not None]
            group_norms[name+'_grad_norm'] = float(torch.stack(squares).sum().sqrt()) if squares else 0.
        norm = torch.nn.utils.clip_grad_norm_(g.parameters(), g.grad_clip, error_if_nonfinite=True)
        used_lrs = [group['lr'] for group in g.opt.param_groups]
        g.opt.step()
        if g.scheduler is not None:
            g.scheduler.step()
        if self.buffer is not None:
            if progress is not None:
                progress.begin('replay_admission', step=step, completed=0, total=len(retained))
            for index, (source, original, path, state) in enumerate(retained):
                if source != 'replay':
                    path.log_proposals = original.log_proposals
                    self.buffer.add(g.env, path, state, source, step)
                if progress is not None:
                    progress.update(completed=index+1)
        self.completed_updates = step
        if progress is not None:
            progress.begin('update_complete', step=step, trajectories=batch_size, loss=loss_value)
        return dict(step=step, loss=loss_value, grad_norm=float(norm), fresh=fresh,
                    compatible_proposal=exploration, replay=replay,
                    policy_temperature=temperature, grad_accum_steps=grad_accum_steps,
                    policy_lr=used_lrs[0], flow_lr=used_lrs[1],
                    next_policy_lr=g.opt.param_groups[0]['lr'], next_flow_lr=g.opt.param_groups[1]['lr'],
                    gradient_clipped=bool(norm>g.grad_clip), **group_norms,
                    log_reward_mean=float(torch.tensor(rewards,dtype=torch.float64).mean()),
                    mean_events=float(torch.tensor(lengths,dtype=torch.float32).mean()),
                    **trajectory_length_statistics(lengths),
                    **(self.buffer.metrics() if self.buffer is not None else {}))

    def state_dict(self):
        return dict(schema_version=self.schema_version, config=asdict(self.config),
                    completed_updates=self.completed_updates, chunk_steps=self.chunk_steps,
                    temperature=self.temperature_config.state_dict(self.completed_updates),
                    replay=self.buffer.state_dict() if self.buffer is not None else None)

    def load_state_dict(self, data):
        if data.get('schema_version') != self.schema_version or data['config'] != asdict(self.config):
            raise ValueError('Incompatible infinite-sites trainer checkpoint')
        if data['chunk_steps'] != self.chunk_steps:
            raise ValueError('Score chunk size changed on resume')
        self.completed_updates = int(data['completed_updates'])
        self.temperature_config.validate_resume(data.get('temperature'),self.completed_updates)
        if self.buffer is not None:
            self.buffer.load_state_dict(data['replay'])
