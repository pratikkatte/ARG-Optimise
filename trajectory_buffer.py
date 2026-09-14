"""Bounded training-only ARG replay with reward and local-topology diversity.

Store action records, rewards for consistency checks, and canonical topologies.
Never store policy probabilities, flows, tensors, or autograd graphs. Replaying
an entry must reconstruct its states and rescore the current policy and flows.
"""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random

import numpy as np

from utils import action_as_dict, action_from_dict

from eval.posterior_summary import topology_signature, mean_pairwise_rf


def action_text(actions):
    return json.dumps([action_as_dict(a) for a in actions], sort_keys=True, separators=(',', ':'), allow_nan=False)


def action_fingerprint(actions):
    return hashlib.sha256(action_text(actions).encode()).hexdigest()


def environment_fingerprint(env):
    fields = dict(sequences=env.sequences, num_blocks=env.num_blocks,
                  sequence_length=env.sequence_length, mutation_rate=env.mutation_rate,
                  recombination_rate=env.recombination_rate, population_size=env.population_size,
                  arg_prior=env.arg_prior, rho=env.rho, time_policy=env.time_policy, reward_C=env.reward_fn.C)
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ReplayEntry:
    key: str
    actions_json: str
    log_reward: float
    topology: tuple
    source: str
    added_step: int

    def actions(self):
        return [action_from_dict(a) for a in json.loads(self.actions_json)]


class DiverseTrajectoryBuffer:
    """Half admission reservoir, half topology-capped high-reward archive.

    The reservoir samples fresh training insertions; exact duplicates already
    retained in either archive are skipped. The elite archive keeps the best rewards with
    a cap per canonical grid topology. Replay draws equally from the reservoir
    and from elite entries stratified by local topology at a random grid site.
    Thus balancing uses local trees, not only exact ARG/grid-path uniqueness.
    """
    schema_version = 1

    def __init__(self, env, capacity=2048, grid_size=16, per_topology=4, seed=7,
                 forbidden_actions=()):
        if capacity < 2 or grid_size < 1 or per_topology < 1:
            raise ValueError('Replay capacity >= 2 and positive grid/quota are required')
        self.capacity = int(capacity)
        self.grid_size = int(grid_size)
        self.per_topology = int(per_topology)
        self.reservoir_capacity = self.capacity // 2
        self.elite_capacity = self.capacity - self.reservoir_capacity
        self.environment_sha256 = environment_fingerprint(env)
        self.sample_count = env.num_sequences
        self.grid = tuple((np.arange(self.grid_size)+.5)/self.grid_size*env.sequence_length)
        self.forbidden = frozenset(action_fingerprint(a) for a in forbidden_actions)
        self.rng = random.Random(seed)
        self.reservoir = {}
        self.elite = {}
        self.seen = 0
        self.admissions = 0
        self.rejected_heldout = 0

    def __len__(self):
        return len(self.reservoir.keys() | self.elite.keys())

    def _topology(self, env, state):
        if not state.is_done:
            raise ValueError('Only complete terminal ARGs may enter replay')
        ts = env.save_to_tree_sequence(state)
        samples = tuple(map(int, ts.samples()))
        if samples != tuple(range(self.sample_count)):
            raise ValueError('Replay sample identities do not match the training alignment')
        signatures, cached = [], {}
        for position in self.grid:
            tree = ts.at(position)
            if tree.index not in cached:
                cached[tree.index] = topology_signature(tree, samples)
            signatures.append(cached[tree.index])
        return tuple(signatures)

    def add(self, env, trajectory, state, source, step):
        if source not in ('policy', 'prior'):
            raise ValueError('Only newly generated training trajectories may enter replay')
        if environment_fingerprint(env) != self.environment_sha256:
            raise ValueError('Replay environment changed')
        if not state.is_done or not math.isfinite(trajectory.log_reward):
            raise ValueError('Replay requires a finite exact terminal reward')
        if abs(float(trajectory.log_reward)-float(state.log_reward)) > 1e-8:
            raise ValueError('Trajectory and terminal-state rewards disagree')
        text = action_text(trajectory.actions)
        key = hashlib.sha256(text.encode()).hexdigest()
        if key in self.forbidden:
            self.rejected_heldout += 1
            raise ValueError('Held-out evaluation trajectory cannot enter training replay')
        existing = self.reservoir.get(key) or self.elite.get(key)
        if existing is not None:
            if abs(existing.log_reward-float(trajectory.log_reward)) > 1e-8:
                raise ValueError('The same replay trajectory has changed reward')
            return False
        entry = ReplayEntry(key, text, float(trajectory.log_reward),
                            self._topology(env, state), source, int(step))
        self._admit(entry)
        return key in self.reservoir or key in self.elite

    def _admit(self, entry):
        # Admission stream counts currently unretained fresh trajectories. It
        # does not keep an unbounded set of historical trajectory hashes.
        self.seen += 1
        if len(self.reservoir) < self.reservoir_capacity:
            self.reservoir[entry.key] = entry
        else:
            index = self.rng.randrange(self.seen)
            if index < self.reservoir_capacity:
                old = tuple(self.reservoir)[index]
                del self.reservoir[old]
                self.reservoir[entry.key] = entry
        same = [r for r in self.elite.values() if r.topology == entry.topology]
        if len(same) >= self.per_topology:
            worst = min(same, key=lambda r: (r.log_reward, r.key))
            if entry.log_reward > worst.log_reward:
                del self.elite[worst.key]
                self.elite[entry.key] = entry
        elif len(self.elite) < self.elite_capacity:
            self.elite[entry.key] = entry
        else:
            worst = min(self.elite.values(), key=lambda r: (r.log_reward, r.key))
            if entry.log_reward > worst.log_reward:
                del self.elite[worst.key]
                self.elite[entry.key] = entry
        self.admissions += int(entry.key in self.reservoir or entry.key in self.elite)

    def sample(self, count):
        if count < 1 or not len(self):
            raise ValueError('Replay needs a nonempty buffer and positive sample count')
        reservoir = tuple(self.reservoir.values())
        elite = tuple(self.elite.values())
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
        entries = list({**self.reservoir, **self.elite}.values())
        result = dict(replay_buffer_size=len(entries), replay_reservoir_size=len(self.reservoir),
                      replay_elite_size=len(self.elite), replay_buffer_seen=self.seen,
                      replay_buffer_admissions=self.admissions, replay_heldout_rejections=self.rejected_heldout)
        if not entries:
            return result
        local = [Counter(e.topology[i] for e in entries) for i in range(self.grid_size)]
        result.update(replay_buffer_unique_grid_topologies=len({e.topology for e in entries}),
            replay_buffer_unique_local_mean=float(np.mean([len(c) for c in local])),
            replay_buffer_modal_local_frequency_mean=float(np.mean([max(c.values())/len(entries) for c in local])),
            replay_buffer_log_reward_mean=float(np.mean([e.log_reward for e in entries])),
            replay_buffer_log_reward_max=max(e.log_reward for e in entries),
            replay_buffer_prior_fraction=sum(e.source == 'prior' for e in entries)/len(entries))
        if len(entries) > 1:
            result['replay_buffer_pairwise_local_rf_mean'] = float(np.mean([
                mean_pairwise_rf([e.topology[i] for e in entries], self.sample_count)
                for i in range(self.grid_size)]))
        return result

    def state_dict(self):
        entries = {**self.reservoir, **self.elite}
        return dict(schema_version=self.schema_version, capacity=self.capacity, grid_size=self.grid_size,
                    per_topology=self.per_topology, environment_sha256=self.environment_sha256,
                    sample_count=self.sample_count, grid=self.grid, forbidden=sorted(self.forbidden),
                    rng_state=self.rng.getstate(), seen=self.seen, admissions=self.admissions,
                    rejected_heldout=self.rejected_heldout,
                    entries={key: asdict(value) for key, value in entries.items()},
                    reservoir=list(self.reservoir), elite=list(self.elite))

    def load_state_dict(self, state):
        for key in ('schema_version', 'capacity', 'grid_size', 'per_topology',
                    'environment_sha256', 'sample_count'):
            if state[key] != getattr(self, key):
                raise ValueError('Incompatible replay checkpoint: '+key)
        if tuple(state['grid']) != self.grid or frozenset(state['forbidden']) != self.forbidden:
            raise ValueError('Replay topology grid or held-out exclusions changed')
        entries = {key: ReplayEntry(**{**value, 'topology': tuple(tuple(s) for s in value['topology'])})
                   for key, value in state['entries'].items()}
        for key, entry in entries.items():
            if entry.key != key or hashlib.sha256(entry.actions_json.encode()).hexdigest() != key:
                raise ValueError('Replay trajectory fingerprint mismatch')
            if key in self.forbidden or not math.isfinite(entry.log_reward):
                raise ValueError('Invalid replay entry')
            if entry.source not in ('policy', 'prior') or len(entry.topology) != self.grid_size or entry.added_step < 1:
                raise ValueError('Invalid replay entry provenance or topology')
        reservoir = {key: entries[key] for key in state['reservoir']}
        elite = {key: entries[key] for key in state['elite']}
        if len(reservoir) > self.reservoir_capacity or len(elite) > self.elite_capacity:
            raise ValueError('Replay checkpoint exceeds capacity')
        if max(Counter(e.topology for e in elite.values()).values(), default=0) > self.per_topology:
            raise ValueError('Replay checkpoint violates topology quota')
        self.reservoir, self.elite = reservoir, elite
        self.rng.setstate(state['rng_state'])
        self.seen, self.admissions = int(state['seen']), int(state['admissions'])
        self.rejected_heldout = int(state['rejected_heldout'])
