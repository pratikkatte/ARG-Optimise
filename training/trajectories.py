"""Serializable trajectories and bounded topology-diverse replay storage."""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random

import numpy as np

from eval.posterior_summary import mean_pairwise_rf, topology_signature
from utils import action_as_dict, action_from_dict


def action_text(actions):
    """Serialize an action sequence to deterministic compact JSON."""
    return json.dumps([action_as_dict(a) for a in actions], sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def action_fingerprint(actions):
    """Return a stable SHA-256 identity for an action sequence."""
    return hashlib.sha256(action_text(actions).encode()).hexdigest()


def environment_fingerprint(env):
    """Return a stable identity for replay-relevant environment settings."""
    return env.dataset_fingerprint



@dataclass(frozen=True)
class ReplayEntry:
    """Store one replayable terminal trajectory without model tensors."""
    key: str
    actions_json: str
    log_reward: float
    topology: tuple
    source: str
    added_step: int
    log_prior: float
    log_proposal: object = None

    def actions(self):
        """Deserialize the stored action sequence."""
        return [action_from_dict(a) for a in json.loads(self.actions_json)]


class DiverseTrajectoryBuffer:
    """Keep uniform-history and topology-capped high-reward replay samples.

    Half the capacity is a reservoir over fresh insertions. The other half
    retains high-reward entries with a quota per canonical grid topology.
    """
    schema_version = 2

    def __init__(self, env, capacity=2048, grid_size=16, per_topology=4, seed=7,
                 forbidden_actions=()):
        """Initialize an empty buffer tied to one training environment."""
        if capacity < 2 or grid_size < 1 or per_topology < 1:
            raise ValueError('Replay capacity >= 2 and positive grid/quota are required')
        self.capacity, self.grid_size, self.per_topology = map(int, (capacity, grid_size, per_topology))
        self.reservoir_capacity = self.capacity // 2
        self.elite_capacity = self.capacity - self.reservoir_capacity
        self.environment_sha256 = environment_fingerprint(env)
        self.sample_count = env.num_sequences
        self.grid = tuple((np.arange(self.grid_size) + .5) / self.grid_size * env.sequence_length)
        self.forbidden = frozenset(action_fingerprint(a) for a in forbidden_actions)
        self.rng = random.Random(seed)
        self.reservoir, self.elite = {}, {}
        self.seen = self.admissions = self.rejected_heldout = 0

    def __len__(self):
        """Return the number of unique retained trajectories."""
        return len(self.reservoir.keys() | self.elite.keys())

    def _topology(self, env, state):
        """Compute canonical local-tree signatures for a terminal ARG."""
        if not state.is_done:
            raise ValueError('Only complete terminal ARGs may enter replay')
        tree_sequence = env.save_to_tree_sequence(state)
        samples = tuple(map(int, tree_sequence.samples()))
        if samples != tuple(range(self.sample_count)):
            raise ValueError('Replay sample identities do not match the training alignment')
        signatures, cached = [], {}
        for position in self.grid:
            tree = tree_sequence.at(position)
            if tree.index not in cached:
                cached[tree.index] = topology_signature(tree, samples)
            signatures.append(cached[tree.index])
        return tuple(signatures)

    def add(self, env, trajectory, state, source, step):
        """Validate and conditionally retain a newly generated trajectory."""
        if source not in ('policy', 'compatible_proposal'):
            raise ValueError('Only newly generated training trajectories may enter replay')
        if environment_fingerprint(env) != self.environment_sha256:
            raise ValueError('Replay environment changed')
        if not state.is_done or not math.isfinite(trajectory.log_reward):
            raise ValueError('Replay requires a finite exact terminal reward')
        if abs(float(trajectory.log_reward) - float(state.log_reward)) > 1e-8:
            raise ValueError('Trajectory and terminal-state rewards disagree')
        if source == 'compatible_proposal' and (len(trajectory.log_proposals) != len(trajectory.actions)
                or any(p is None or not math.isfinite(p) for p in trajectory.log_proposals)):
            raise ValueError('Compatible exploration requires its actual proposal log densities')
        text = action_text(trajectory.actions)
        key = hashlib.sha256(text.encode()).hexdigest()
        if key in self.forbidden:
            self.rejected_heldout += 1
            raise ValueError('Held-out evaluation trajectory cannot enter training replay')
        existing = self.reservoir.get(key) or self.elite.get(key)
        if existing is not None:
            if abs(existing.log_reward - float(trajectory.log_reward)) > 1e-8:
                raise ValueError('The same replay trajectory has changed reward')
            return False
        entry = ReplayEntry(key, text, float(trajectory.log_reward), self._topology(env, state),
                            source, int(step), float(state.accumulated_log_prior),
                            sum(trajectory.log_proposals) if trajectory.log_proposals and
                            all(p is not None for p in trajectory.log_proposals) else None)
        self._admit(entry)
        return key in self.reservoir or key in self.elite

    def _admit(self, entry):
        """Apply reservoir sampling and elite topology quotas to one entry."""
        self.seen += 1
        if len(self.reservoir) < self.reservoir_capacity:
            self.reservoir[entry.key] = entry
        else:
            index = self.rng.randrange(self.seen)
            if index < self.reservoir_capacity:
                del self.reservoir[tuple(self.reservoir)[index]]
                self.reservoir[entry.key] = entry
        same = [retained for retained in self.elite.values() if retained.topology == entry.topology]
        if len(same) >= self.per_topology:
            worst = min(same, key=lambda retained: (retained.log_reward, retained.key))
            if entry.log_reward > worst.log_reward:
                del self.elite[worst.key]
                self.elite[entry.key] = entry
        elif len(self.elite) < self.elite_capacity:
            self.elite[entry.key] = entry
        else:
            worst = min(self.elite.values(), key=lambda retained: (retained.log_reward, retained.key))
            if entry.log_reward > worst.log_reward:
                del self.elite[worst.key]
                self.elite[entry.key] = entry
        self.admissions += int(entry.key in self.reservoir or entry.key in self.elite)

    def sample(self, count):
        """Sample entries with equal reservoir and elite probability."""
        if count < 1 or not len(self):
            raise ValueError('Replay needs a nonempty buffer and positive sample count')
        reservoir, elite = tuple(self.reservoir.values()), tuple(self.elite.values())
        buckets = [defaultdict(list) for _ in self.grid]
        for entry in elite:
            for index, signature in enumerate(entry.topology):
                buckets[index][signature].append(entry)
        samples = []
        for _ in range(count):
            if reservoir and (not elite or self.rng.random() < .5):
                entry = self.rng.choice(reservoir)
            else:
                local = self.rng.choice(buckets)
                entry = self.rng.choice(local[self.rng.choice(tuple(local))])
            samples.append(entry)
        return samples

    def metrics(self):
        """Summarize buffer size, reward quality, and topology diversity."""
        entries = list({**self.reservoir, **self.elite}.values())
        result = dict(replay_buffer_size=len(entries), replay_reservoir_size=len(self.reservoir),
                      replay_elite_size=len(self.elite), replay_buffer_seen=self.seen,
                      replay_buffer_admissions=self.admissions,
                      replay_heldout_rejections=self.rejected_heldout)
        if not entries:
            return result
        local = [Counter(entry.topology[i] for entry in entries) for i in range(self.grid_size)]
        result.update(replay_buffer_unique_grid_topologies=len({e.topology for e in entries}),
            replay_buffer_unique_local_mean=float(np.mean([len(c) for c in local])),
            replay_buffer_modal_local_frequency_mean=float(np.mean([max(c.values()) / len(entries) for c in local])),
            replay_buffer_log_reward_mean=float(np.mean([e.log_reward for e in entries])),
            replay_buffer_log_reward_max=max(e.log_reward for e in entries),
            replay_buffer_compatible_proposal_fraction=sum(e.source == 'compatible_proposal' for e in entries) / len(entries))
        if len(entries) > 1:
            result['replay_buffer_pairwise_local_rf_mean'] = float(np.mean([
                mean_pairwise_rf([entry.topology[i] for entry in entries], self.sample_count)
                for i in range(self.grid_size)]))
        return result

    def state_dict(self):
        """Serialize entries, counters, exclusions, and RNG state."""
        entries = {**self.reservoir, **self.elite}
        return dict(schema_version=self.schema_version, capacity=self.capacity, grid_size=self.grid_size,
                    per_topology=self.per_topology, environment_sha256=self.environment_sha256,
                    sample_count=self.sample_count, grid=self.grid, forbidden=sorted(self.forbidden),
                    rng_state=self.rng.getstate(), seen=self.seen, admissions=self.admissions,
                    rejected_heldout=self.rejected_heldout,
                    entries={key: asdict(value) for key, value in entries.items()},
                    reservoir=list(self.reservoir), elite=list(self.elite))

    def load_state_dict(self, state):
        """Restore and validate a schema-version-2 replay checkpoint."""
        for key in ('schema_version', 'capacity', 'grid_size', 'per_topology',
                    'environment_sha256', 'sample_count'):
            if state[key] != getattr(self, key):
                raise ValueError('Incompatible replay checkpoint: ' + key)
        if tuple(state['grid']) != self.grid or frozenset(state['forbidden']) != self.forbidden:
            raise ValueError('Replay topology grid or held-out exclusions changed')
        entries = {key: ReplayEntry(**{**value,
            'topology': tuple(tuple(signature) for signature in value['topology'])})
            for key, value in state['entries'].items()}
        for key, entry in entries.items():
            if entry.key != key or hashlib.sha256(entry.actions_json.encode()).hexdigest() != key:
                raise ValueError('Replay trajectory fingerprint mismatch')
            if (key in self.forbidden or not math.isfinite(entry.log_reward)
                    or not math.isfinite(entry.log_prior)
                    or (entry.log_proposal is not None and not math.isfinite(entry.log_proposal))
                    or (entry.source == 'compatible_proposal' and entry.log_proposal is None)):
                raise ValueError('Invalid replay entry')
            if entry.source not in ('policy', 'compatible_proposal') or len(entry.topology) != self.grid_size or entry.added_step < 1:
                raise ValueError('Invalid replay entry provenance or topology')
        reservoir = {key: entries[key] for key in state['reservoir']}
        elite = {key: entries[key] for key in state['elite']}
        if len(reservoir) > self.reservoir_capacity or len(elite) > self.elite_capacity:
            raise ValueError('Replay checkpoint exceeds capacity')
        if max(Counter(entry.topology for entry in elite.values()).values(), default=0) > self.per_topology:
            raise ValueError('Replay checkpoint violates topology quota')
        self.reservoir, self.elite = reservoir, elite
        self.rng.setstate(state['rng_state'])
        self.seen, self.admissions = int(state['seen']), int(state['admissions'])
        self.rejected_heldout = int(state['rejected_heldout'])
