"""Unified on-policy, tempered, prior-exploration, and replay training."""
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from env.env import SimpleTrajectory
from training.schedules import PolicyTemperatureConfig
from training.trajectories import DiverseTrajectoryBuffer, action_fingerprint


@dataclass(frozen=True)
class TrajectoryMixConfig:
    """Configure the prior and replay shares of each trajectory batch."""
    exploration_fraction: float = 0.
    replay_fraction: float = 0.
    replay_capacity: int = 2048
    replay_grid_size: int = 16
    replay_per_topology: int = 4
    replay_min_size: int = 128

    @classmethod
    def from_namespace(cls, args):
        """Build a configuration from parsed command-line arguments."""
        return cls(**{name: getattr(args, name) for name in cls.__dataclass_fields__})

    def validate(self):
        """Reject invalid fractions and replay-buffer dimensions."""
        if not all(math.isfinite(x) and 0 <= x < 1 for x in (
                self.exploration_fraction, self.replay_fraction)):
            raise ValueError('Exploration and replay fractions must lie in [0,1)')
        if self.exploration_fraction + self.replay_fraction >= 1:
            raise ValueError('Keep a positive fresh on-policy training fraction')
        if self.replay_capacity < 2 or min(
                self.replay_grid_size, self.replay_per_topology, self.replay_min_size) < 1:
            raise ValueError('Invalid replay capacity, grid, quota, or minimum size')
        if self.replay_min_size > self.replay_capacity:
            raise ValueError('Replay minimum size exceeds capacity')

    @property
    def enabled(self):
        """Return whether prior exploration or replay is enabled."""
        return bool(self.exploration_fraction or self.replay_fraction)


def allocated_count(batch_size, fraction, microstep):
    """Round deterministically while preserving a fraction over many updates."""
    return round(batch_size * fraction * microstep) - round(batch_size * fraction * (microstep - 1))


def trajectory_length_statistics(lengths, prefix=''):
    """Return median, 95th percentile, and maximum trajectory lengths."""
    values = torch.as_tensor(lengths, dtype=torch.float64)
    return {prefix + 'trajectory_length_' + key: float(value) for key, value in (
        ('median', values.quantile(.5)), ('p95', values.quantile(.95)), ('max', values.max()))}


@torch.no_grad()
def sample_prior_trajectories(env, episodes):
    """Draw complete new trajectories from the environment's physical prior."""
    if episodes < 1:
        raise ValueError('Prior exploration requires positive episodes')
    trajectories = []
    for _ in range(episodes):
        state, trajectory = env.get_initial_state(), SimpleTrajectory()
        while not state.is_done:
            action, log_prior = env.sample_prior_step(state)
            state = env.apply_action(state, action, log_prior=log_prior)
            trajectory.update(action, log_prior=log_prior, log_reward=state.log_reward)
        trajectories.append(trajectory)
    return trajectories


class Trainer:
    """Apply one update from policy, prior, and replay trajectory sources.

    All sources share one scoring and accumulation path. New discoveries enter
    replay only after the update, and stored actions use current-model scores.
    """
    schema_version = 1

    def __init__(self, generator, worker, mix_config=None, temperature_config=None, seed=7,
                 forbidden_actions=()):
        """Initialize the unified trainer and optional replay buffer."""
        self.generator, self.worker = generator, worker
        self.config = mix_config or TrajectoryMixConfig()
        self.temperature_config = temperature_config or PolicyTemperatureConfig()
        self.config.validate()
        self.temperature_config.validate_training(generator.loss_type, generator.arg_model.event_policy,
            self.config.exploration_fraction, self.config.replay_fraction)
        if self.config.enabled and generator.loss_type != 'subtb':
            raise ValueError('Exploration/replay training requires SubTB')
        if self.config.enabled and generator.arg_model.event_policy != 'cwr_residual':
            raise ValueError('Current-policy rescoring requires cwr_residual')
        self.forbidden = frozenset(action_fingerprint(a) for a in forbidden_actions)
        self.buffer = (DiverseTrajectoryBuffer(generator.env, capacity=self.config.replay_capacity,
            grid_size=self.config.replay_grid_size, per_topology=self.config.replay_per_topology,
            seed=seed + 700001, forbidden_actions=forbidden_actions)
            if self.config.replay_fraction else None)
        self.completed_updates = self.total_scored = self.total_generated = 0
        self.total_replayed = self.total_prior = self.total_transitions = 0

    def _check_training_paths(self, paths):
        """Reject any training path reserved for held-out evaluation."""
        if any(action_fingerprint(path.actions) in self.forbidden for path in paths):
            raise ValueError('Held-out evaluation paths cannot be used for training')

    def _source_counts(self, batch_size, index):
        """Allocate one microbatch across policy, prior, and replay paths."""
        prior = allocated_count(batch_size, self.config.exploration_fraction, index)
        replay = (allocated_count(batch_size, self.config.replay_fraction, index)
                  if self.buffer is not None and len(self.buffer) >= self.config.replay_min_size else 0)
        policy = batch_size - prior - replay
        if policy < 1:
            raise ValueError('Rounding left no fresh policy trajectories; increase batch size or reduce fractions')
        return dict(policy=policy, prior=prior, replay=replay)

    def _score_source(self, source, count, replay_entries, random_spec):
        """Generate or reconstruct one source batch under the current model."""
        if source == 'policy':
            options = {'collect_flows': True} if self.generator.loss_type == 'subtb' else {}
            if random_spec is not None:
                options['random_spec'] = random_spec
            if self.buffer is not None:
                options['return_states'] = True
            outputs, paths = self.worker.rollout(self.generator, episodes=count, **options)
            generated_transitions = 0
        elif source == 'prior':
            fresh = sample_prior_trajectories(self.worker.env, count)
            self._check_training_paths(fresh)
            outputs, paths = self.worker.replay(self.generator, fresh, collect_flows=True,
                                                return_states=self.buffer is not None)
            torch.testing.assert_close(outputs['log_rewards'], torch.tensor(
                [path.log_reward for path in fresh], dtype=torch.float64,
                device=self.generator.device), rtol=0, atol=1e-8)
            generated_transitions = sum(map(len, fresh))
        else:
            outputs, paths = self.worker.replay(self.generator,
                [entry.actions() for entry in replay_entries], collect_flows=True)
            torch.testing.assert_close(outputs['log_rewards'], torch.tensor(
                [entry.log_reward for entry in replay_entries], dtype=torch.float64,
                device=self.generator.device), rtol=0, atol=1e-8)
            generated_transitions = 0
        return outputs, paths, generated_transitions

    def train_epoch(self, step, batch_size, grad_accum_steps=1):
        """Score one accumulated trajectory batch and apply one optimizer update."""
        if step != self.completed_updates + 1:
            raise ValueError('Training update differs from its saved state')
        batch_size, grad_accum_steps = int(batch_size), max(int(grad_accum_steps), 1)
        if batch_size < 1:
            raise ValueError('Positive batch and accumulation sizes are required')
        counts = dict(policy=0, prior=0, replay=0)
        lengths, losses = ({source: [] for source in counts} for _ in range(2))
        chosen_replay, new_entries, recombinations = [], [], []
        transitions = 0
        temperature_enabled = self.temperature_config.schedule != 'constant'
        random_spec = self.temperature_config.random_spec(self.completed_updates)

        for microstep in range(grad_accum_steps):
            index = self.completed_updates * grad_accum_steps + microstep + 1
            source_counts = self._source_counts(batch_size, index)
            replay_entries = (self.buffer.sample(source_counts['replay'])
                              if source_counts['replay'] else [])
            chosen_replay.extend(replay_entries)
            for source, count in source_counts.items():
                if not count:
                    continue
                outputs, paths, generated = self._score_source(source, count, replay_entries, random_spec)
                self._check_training_paths(paths)
                if self.config.enabled:
                    assert not outputs['log_rewards'].requires_grad
                    loss = float(self.generator.get_loss_from_rollout_outputs(outputs).detach())
                    losses[source].append((count, loss))
                counts[source] += count
                lengths[source].extend(map(len, paths))
                transitions += generated + sum(map(len, paths))
                if temperature_enabled and source == 'policy':
                    recombinations.extend(sum(a.event_type == 'recomb' for a in path.actions) for path in paths)
                self.generator.accumulate_loss(outputs, factor=batch_size * grad_accum_steps / count)
                if self.buffer is not None and source != 'replay':
                    new_entries.extend((path, state, source)
                                       for path, state in zip(paths, outputs['states']))

        info = self.generator.update_model()
        if self.buffer is not None:
            for path, state, source in new_entries:
                self.buffer.add(self.worker.env, path, state, source, step)
        self.completed_updates = step
        self.total_scored += sum(counts.values())
        self.total_generated += counts['policy'] + counts['prior']
        self.total_replayed += counts['replay']
        self.total_prior += counts['prior']
        self.total_transitions += transitions
        info.update(trajectory_length_statistics(lengths['policy']))
        if self.config.enabled:
            self._add_mixture_metrics(info, counts, lengths, losses, transitions, chosen_replay)
        elif temperature_enabled:
            self._add_temperature_metrics(info, lengths['policy'], recombinations)
        return info

    def _add_mixture_metrics(self, info, counts, lengths, losses, transitions, chosen_replay):
        """Add source, budget, reward, and diversity metrics for mixed training."""
        scored = sum(counts.values())
        for source, count in counts.items():
            if count:
                info[source + '_training_subtb_loss'] = sum(n * loss for n, loss in losses[source]) / count
                if source != 'policy':
                    info.update(trajectory_length_statistics(lengths[source], source + '_'))
        info.update(on_policy_training_episodes=counts['policy'], prior_training_episodes=counts['prior'],
            replay_training_episodes=counts['replay'], scored_training_episodes=scored,
            generated_training_episodes=counts['policy'] + counts['prior'],
            exploration_fraction=self.config.exploration_fraction, replay_fraction=self.config.replay_fraction,
            realized_exploration_fraction=counts['prior'] / scored,
            realized_replay_fraction=counts['replay'] / scored,
            training_environment_transitions=transitions, total_scored_training_episodes=self.total_scored,
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

    def _add_temperature_metrics(self, info, lengths, recombinations):
        """Add temperature and generated-trajectory accounting."""
        temperature, episodes = self.temperature_config.temperature(self.completed_updates - 1), len(lengths)
        info.update(sampling_policy_temperature=temperature, sampling_time_temperature=1., reward_temperature=1.,
            policy_temperature_completed_updates=self.completed_updates,
            training_sample_source='tempered_policy' if temperature > 1 else 'policy',
            tempered_policy_training_episodes=episodes if temperature > 1 else 0,
            on_policy_training_episodes=episodes if temperature == 1 else 0,
            scored_training_episodes=episodes, generated_training_episodes=episodes,
            total_scored_training_episodes=self.completed_updates * episodes,
            total_generated_training_episodes=self.completed_updates * episodes,
            prior_training_episodes=0, replay_training_episodes=0,
            training_event_count_mean=float(np.mean(lengths)),
            training_recombination_count_mean=float(np.mean(recombinations)))

    def state_dict(self):
        """Serialize mixture progress using the existing replay-state schema."""
        return dict(schema_version=self.schema_version, config=asdict(self.config),
                    completed_updates=self.completed_updates, forbidden=sorted(self.forbidden),
                    total_scored=self.total_scored, total_generated=self.total_generated,
                    total_replayed=self.total_replayed, total_prior=self.total_prior,
                    total_transitions=self.total_transitions,
                    buffer=self.buffer.state_dict() if self.buffer is not None else None)

    def load_state_dict(self, state):
        """Restore and validate an existing replay-training state dictionary."""
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
