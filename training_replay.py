"""Optional equal-budget prior exploration and diverse ARG replay for SubTB."""
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from training_exploration import sample_prior_trajectories
from trajectory_buffer import DiverseTrajectoryBuffer, action_fingerprint


@dataclass(frozen=True)
class ReplayTrainingConfig:
    exploration_fraction: float = 0.
    replay_fraction: float = 0.
    replay_capacity: int = 2048
    replay_grid_size: int = 16
    replay_per_topology: int = 4
    replay_min_size: int = 128

    @classmethod
    def from_namespace(cls, args):
        return cls(**{name: getattr(args, name) for name in cls.__dataclass_fields__})

    def validate(self):
        if not all(math.isfinite(x) and 0 <= x < 1 for x in (
                self.exploration_fraction, self.replay_fraction)):
            raise ValueError('Exploration and replay fractions must lie in [0,1)')
        if self.exploration_fraction+self.replay_fraction >= 1:
            raise ValueError('Keep a positive fresh on-policy training fraction')
        if self.replay_capacity < 2 or min(self.replay_grid_size, self.replay_per_topology, self.replay_min_size) < 1:
            raise ValueError('Invalid replay capacity, grid, quota, or minimum size')
        if self.replay_min_size > self.replay_capacity:
            raise ValueError('Replay minimum size exceeds capacity')

    @property
    def enabled(self):
        return bool(self.exploration_fraction or self.replay_fraction)


def allocated_count(batch_size, fraction, microstep):
    """Deterministic rounding preserves the requested fraction over updates."""
    return round(batch_size*fraction*microstep)-round(batch_size*fraction*(microstep-1))


class ReplayTrainer:
    schema_version = 1

    def __init__(self, generator, config, seed=7, forbidden_actions=()):
        config.validate()
        if generator.loss_type != 'subtb':
            raise ValueError('Exploration/replay training requires SubTB')
        if generator.arg_model.event_policy != 'cwr_residual':
            raise ValueError('Current-policy rescoring requires cwr_residual')
        self.config = config
        self.forbidden = frozenset(action_fingerprint(a) for a in forbidden_actions)
        self.buffer = (DiverseTrajectoryBuffer(generator.env, capacity=config.replay_capacity,
            grid_size=config.replay_grid_size, per_topology=config.replay_per_topology,
            seed=seed+700001, forbidden_actions=forbidden_actions) if config.replay_fraction else None)
        self.completed_updates = 0
        self.total_scored = 0
        self.total_generated = 0
        self.total_replayed = 0
        self.total_prior = 0
        self.total_transitions = 0

    def _check_training_paths(self, paths):
        if any(action_fingerprint(p.actions) in self.forbidden for p in paths):
            raise ValueError('Held-out evaluation paths cannot be used for training')

    def train_epoch(self, epoch_id, worker, generator, batch_size, grad_accum_steps=1):
        from train import length_statistics, train_epoch
        if not self.config.enabled:
            # Preserve the existing computation, RNG consumption, and outputs.
            return train_epoch(epoch_id, worker, generator, batch_size, grad_accum_steps)
        if epoch_id != self.completed_updates+1:
            raise ValueError('Replay training update differs from its saved state')
        if min(batch_size, grad_accum_steps) < 1:
            raise ValueError('Positive batch and accumulation sizes are required')
        counts = dict(policy=0, prior=0, replay=0)
        lengths = {key: [] for key in counts}
        losses = {key: [] for key in counts}
        chosen_replay = []
        new_entries = []
        transitions = 0
        for micro in range(grad_accum_steps):
            index = (epoch_id-1)*grad_accum_steps+micro+1
            prior_count = allocated_count(batch_size, self.config.exploration_fraction, index)
            replay_count = (allocated_count(batch_size, self.config.replay_fraction, index)
                            if self.buffer is not None and len(self.buffer) >= self.config.replay_min_size else 0)
            policy_count = batch_size-prior_count-replay_count
            if policy_count < 1:
                raise ValueError('Rounding left no fresh policy trajectories; increase batch size or reduce fractions')
            # Draw only from earlier training updates. New discoveries enter
            # the buffer after this update, not before its replay draw.
            replay_entries = self.buffer.sample(replay_count) if replay_count else []
            chosen_replay.extend(replay_entries)
            for source, count in (('policy', policy_count), ('prior', prior_count), ('replay', replay_count)):
                if not count:
                    continue
                if source == 'policy':
                    outputs, paths = worker.rollout(generator, count, collect_flows=True,
                        return_states=self.buffer is not None)
                elif source == 'prior':
                    fresh_prior = sample_prior_trajectories(worker.env, count)
                    self._check_training_paths(fresh_prior)
                    transitions += sum(map(len, fresh_prior))
                    outputs, paths = worker.replay(generator, fresh_prior, collect_flows=True,
                        return_states=self.buffer is not None)
                    torch.testing.assert_close(outputs['log_rewards'], torch.tensor(
                        [p.log_reward for p in fresh_prior], dtype=torch.float64, device=generator.device),
                        rtol=0, atol=1e-8)
                else:
                    outputs, paths = worker.replay(generator, [e.actions() for e in replay_entries], collect_flows=True)
                    torch.testing.assert_close(outputs['log_rewards'], torch.tensor(
                        [e.log_reward for e in replay_entries], dtype=torch.float64, device=generator.device),
                        rtol=0, atol=1e-8)
                self._check_training_paths(paths)
                assert not outputs['log_rewards'].requires_grad
                loss = float(generator.get_loss_from_rollout_outputs(outputs).detach())
                counts[source] += count
                losses[source].append((count, loss))
                lengths[source].extend(map(len, paths))
                transitions += sum(map(len, paths))
                # Mean over exactly batch_size * accumulation scored paths;
                # the per-trajectory SubTB weights and exact reward stay fixed.
                generator.accumulate_loss(outputs, factor=batch_size*grad_accum_steps/count)
                if self.buffer is not None and source != 'replay':
                    new_entries.extend((path, state, source) for path, state in zip(paths, outputs['states']))
                del outputs
        info = generator.update_model()
        if self.buffer is not None:
            for path, state, source in new_entries:
                self.buffer.add(worker.env, path, state, source, epoch_id)
        self.completed_updates = epoch_id
        self.total_scored += sum(counts.values())
        self.total_generated += counts['policy']+counts['prior']
        self.total_replayed += counts['replay']
        self.total_prior += counts['prior']
        self.total_transitions += transitions
        info.update(length_statistics(lengths['policy']))
        for source in counts:
            if counts[source]:
                info[source+'_training_subtb_loss'] = sum(n*x for n, x in losses[source])/counts[source]
                if source != 'policy':
                    info.update(length_statistics(lengths[source], source+'_'))
        info.update(on_policy_training_episodes=counts['policy'], prior_training_episodes=counts['prior'],
                    replay_training_episodes=counts['replay'], scored_training_episodes=sum(counts.values()),
                    generated_training_episodes=counts['policy']+counts['prior'],
                    exploration_fraction=self.config.exploration_fraction,
                    replay_fraction=self.config.replay_fraction,
                    realized_exploration_fraction=counts['prior']/sum(counts.values()),
                    realized_replay_fraction=counts['replay']/sum(counts.values()),
                    training_environment_transitions=transitions,
                    total_scored_training_episodes=self.total_scored,
                    total_generated_training_episodes=self.total_generated,
                    total_replay_training_episodes=self.total_replayed,
                    total_prior_training_episodes=self.total_prior,
                    total_training_environment_transitions=self.total_transitions,
                    reward_temperature=1., sampling_policy_temperature=1.)
        if chosen_replay:
            info.update(replay_sample_unique_trajectories=len({e.key for e in chosen_replay}),
                        replay_sample_unique_grid_topologies=len({e.topology for e in chosen_replay}),
                        replay_sample_log_reward_mean=float(np.mean([e.log_reward for e in chosen_replay])))
        if self.buffer is not None:
            info.update(self.buffer.metrics())
        return info

    def state_dict(self):
        return dict(schema_version=self.schema_version, config=asdict(self.config),
                    completed_updates=self.completed_updates, forbidden=sorted(self.forbidden),
                    total_scored=self.total_scored, total_generated=self.total_generated,
                    total_replayed=self.total_replayed, total_prior=self.total_prior,
                    total_transitions=self.total_transitions,
                    buffer=self.buffer.state_dict() if self.buffer is not None else None)

    def load_state_dict(self, state):
        if state['schema_version'] != self.schema_version or state['config'] != asdict(self.config):
            raise ValueError('Replay training checkpoint/config mismatch')
        if frozenset(state['forbidden']) != self.forbidden:
            raise ValueError('Replay held-out exclusions changed')
        if (state['buffer'] is None) != (self.buffer is None):
            raise ValueError('Replay buffer unexpectedly missing or present')
        if self.buffer is not None:
            self.buffer.load_state_dict(state['buffer'])
        for key in ('completed_updates', 'total_scored', 'total_generated', 'total_replayed',
                    'total_prior', 'total_transitions'):
            setattr(self, key, int(state[key]))
