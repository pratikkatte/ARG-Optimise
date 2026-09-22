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
                # This path retains actions and scalar scores, not past states.
                # Keep the supplied-prior check without cloning the whole ARG.
                state, _ = env.step_owned_state(state, step.action, step.log_prior)
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

    def __init__(self, generator, worker, mix_config=None, seed=7, chunk_steps=16, temperature_config=None):
        self.generator, self.worker = generator, worker
        self.config = mix_config or TrajectoryMixConfig()
        self.config.validate()
        self.chunk_steps = chunk_steps  # Accepted for legacy configs/checkpoints; inactive.
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

    def _score_subset(self, subset, sampled=None):
        """Keep the fresh prefix's graph and score any off-policy suffix once.

        Sampling and training use the same microbatch boundaries. Only the last
        fresh microbatch can need a suffix of compatible-proposal/replay paths.
        Activations are released after this microbatch's loss.backward().
        """
        if sampled is None:
            return self.worker.replay(self.generator, subset, return_states=True)
        fresh = len(sampled['lengths'])
        if fresh == len(subset):
            return sampled, subset
        scored, paths = self.worker.replay(self.generator, subset[fresh:], return_states=True)
        steps = max(sampled['log_paths_pf'].shape[1], scored['log_paths_pf'].shape[1])
        outputs = {}
        for key in ('log_paths_pf', 'log_paths_pb', 'log_factors', 'state_flows'):
            width = steps + (key == 'state_flows')
            def padded(value):
                padding = (0, width-value.shape[1])
                if value.ndim == 3:  # Preserve the policy-factor dimension.
                    padding = (0, 0) + padding
                return torch.nn.functional.pad(value, padding)
            outputs[key] = torch.cat((padded(sampled[key]), padded(scored[key])))
        for key in ('lengths', 'log_rewards'):
            outputs[key] = torch.cat((sampled[key], scored[key]))
        outputs['states'] = sampled['states'] + scored['states']
        return outputs, subset[:fresh] + paths

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
        temperature = self.temperature_config.temperature(self.completed_updates)
        spec = self.temperature_config.random_spec(self.completed_updates)
        reuse_fresh = temperature == 1.0
        micro_size = math.ceil(batch_size/grad_accum_steps)
        paths = []
        sources = ['policy']*fresh + ['compatible_proposal']*exploration + ['replay']*replay
        extras_prepared = False
        # Each microbatch contributes its fraction of the full-batch mean.
        # One clip, optimizer update, and scheduler step follow all microbatches.
        g.opt.zero_grad(set_to_none=True)
        loss_value, rewards, lengths, retained = 0., [], [], []
        component_values = dict(subtb_loss=0.,tb_loss=0.)
        for start in range(0,batch_size,micro_size):
            sampled = None
            if start < fresh:
                # At T=1, sampling itself supplies the differentiable training scores.
                # Tempered proposals need a separate T=1 scoring pass below.
                with torch.set_grad_enabled(reuse_fresh):
                    sampled, new = self.worker.rollout(g, min(micro_size,fresh-start), random_spec=spec,
                        collect_flows=reuse_fresh, return_states=reuse_fresh)
                paths.extend(new)
                if not reuse_fresh:
                    sampled = None
            if len(paths) == fresh and not extras_prepared:
                # Preserve fresh/proposal/replay sampling order and RNG consumption.
                with torch.no_grad():
                    if exploration:
                        paths += sample_compatible_trajectories(g.env, exploration, self.worker.max_events)
                    if replay:
                        for entry in self.buffer.sample(replay):
                            path = SimpleTrajectory(); path.actions = entry.actions()
                            paths.append(path)
                extras_prepared = True
            subset = paths[start:start+micro_size]
            outputs, rescored = self._score_subset(subset, sampled)
            del sampled
            components = g.loss_components(outputs)
            weight = len(subset)/batch_size
            loss = components['total']*weight
            for name in ('subtb','tb'):
                component_values[name+'_loss'] += float(components[name].detach())*weight
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite SubTB loss')
            loss.backward()
            loss_value += float(loss.detach())
            rewards.extend(outputs['log_rewards'].tolist()); lengths.extend(outputs['lengths'].tolist())
            retained.extend(zip(sources[start:start+micro_size], subset, rescored, outputs['states']))
            del outputs, loss, rescored, components
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
            for source, original, path, state in retained:
                if source != 'replay':
                    path.log_proposals = original.log_proposals
                    self.buffer.add(g.env, path, state, source, step)
        self.completed_updates = step
        return dict(step=step, loss=loss_value, grad_norm=float(norm), fresh=fresh,
                    **component_values,gradient_clip_factor=min(1.,g.grad_clip/(float(norm)+1e-6)),
                    compatible_proposal=exploration, replay=replay,
                    policy_temperature=temperature, grad_accum_steps=grad_accum_steps,
                    policy_lr=used_lrs[0], flow_lr=used_lrs[1],
                    next_policy_lr=g.opt.param_groups[0]['lr'], next_flow_lr=g.opt.param_groups[1]['lr'],
                    encoder_lr=used_lrs[2] if len(used_lrs)>2 else used_lrs[0],
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
        self.completed_updates = int(data['completed_updates'])
        self.temperature_config.validate_resume(data.get('temperature'),self.completed_updates)
        if self.buffer is not None:
            self.buffer.load_state_dict(data['replay'])
