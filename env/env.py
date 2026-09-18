"""Infinite-sites Hudson ARG environment; CPU float64 reference implementation."""
import hashlib
import math
import random
from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np

from . import priors
from .action_context import ActionContext, update_compatibility
from .actions import CoalescenceChoice, PriorActionOptions, RecombinationChoice
from .infinite_sites import evaluate_infinite_sites
from .infinite_sites_tracker import InfiniteSitesTracker
from .snp_data import SNPData
from .states import ARGLineage, ARGState, MaterialSegments
from .time_env import TimeEnvCwrExponential


class IncompatibleActionError(ValueError):
    def __init__(self, site_ids):
        self.site_ids = tuple(site_ids)
        super().__init__(f'Coalescence contradicts infinite sites at SNP IDs {self.site_ids}')


class ARGReward:
    def __init__(self, C=3000):
        self.C = float(C)
        if not math.isfinite(self.C):
            raise ValueError('reward offset must be finite')

    def __call__(self, log_likelihood, accumulated_log_prior):
        if math.isnan(log_likelihood) or log_likelihood == math.inf or not math.isfinite(accumulated_log_prior):
            raise FloatingPointError('invalid likelihood or prior')
        score = float(self.C + log_likelihood + accumulated_log_prior)
        if log_likelihood != -math.inf and not math.isfinite(score):
            raise FloatingPointError('reward overflowed float64')
        return score


class SimpleTrajectory:
    def __init__(self):
        self.actions, self.log_priors, self.log_proposals, self.records = [], [], [], []
        self.log_reward = None

    def update(self, action, log_prior=None, log_reward=None, record=None, active_lineages=None,
               log_proposal=None):
        self.actions.append(action)
        self.log_priors.append(log_prior)
        self.log_proposals.append(log_proposal)
        self.log_reward = log_reward
        if record is not None:
            self.records.append(record)

    def __len__(self):
        return len(self.actions)


@dataclass(frozen=True)
class CompatibleStep:
    action: object
    log_proposal: float
    log_prior: float


class SimpleARGEnvironment:
    """Discrete genomic links, continuous 2Ne waits, and polarized SNP observations.

    Physical action enumeration and prior rates never depend on compatibility.
    Resolved material is retained until every physical position has one ancestor.
    """
    mutation_model = 'infinite_sites'

    def __init__(self, *, snp_data=None, population_size=10000.0, effective_population_size=None,
                 mutation_rate=2e-8, recombination_rate=2e-8, seed=7, reward_C=3000,
                 bp_per_blocks=1, device='cpu', arg_prior='hudson', time_policy='cwr_exponential',
                 sequences=None, num_sequences=None, sequence_length=None, num_blocks=None, rho=None):
        if sequences is not None:
            raise ValueError('FASTA/JC69 inputs are retired; pass snp_data=SNPData')
        if not isinstance(snp_data, SNPData):
            raise ValueError('snp_data must be SNPData from load_snp_dataset')
        if str(device) != 'cpu':
            raise ValueError('Phase 1 infinite-sites environment requires CPU float64')
        if arg_prior != 'hudson' or time_policy != 'cwr_exponential':
            raise ValueError('Infinite-sites environment requires Hudson with continuous 2Ne waits')
        if bp_per_blocks != 1 or not float(snp_data.sequence_length).is_integer():
            raise ValueError('Phase 1 requires an integer physical length and one-base blocks')
        self.snp_data = snp_data
        self.num_sequences, self.num_variants = snp_data.genotypes.shape
        self.sequence_length = self.num_blocks = int(snp_data.sequence_length)
        for supplied, expected in ((num_sequences, self.num_sequences),
                                   (sequence_length, self.sequence_length), (num_blocks, self.num_blocks)):
            if supplied is not None and supplied != expected:
                raise ValueError('dimensions must match SNPData and one-base physical blocks')
        self.population_size = self._rate(population_size if effective_population_size is None
                                          else effective_population_size, 'population_size', positive=True)
        self.mutation_rate = self._rate(mutation_rate, 'mutation_rate')
        self.recombination_rate = self._rate(recombination_rate, 'recombination_rate')
        self.kappa = 2 * self.population_size * self.mutation_rate
        self.rho = 4 * self.population_size * self.recombination_rate * self.sequence_length
        if not math.isfinite(self.kappa) or not math.isfinite(self.rho):
            raise ValueError('scaled rates overflow float64')
        if rho is not None and not math.isclose(float(rho), self.rho, rel_tol=1e-12, abs_tol=0):
            raise ValueError('rho must equal 4Ne*r*physical sequence length')
        if self.num_variants and self.kappa == 0:
            raise ValueError('observed SNPs have zero support at zero mutation rate')
        self.arg_prior, self.time_policy, self.device = arg_prior, time_policy, 'cpu'
        self.time_env, self.reward_fn = TimeEnvCwrExponential(), ARGReward(reward_C)
        self.rng = random.Random(seed)
        self.all_samples = (1 << self.num_sequences) - 1
        self.derived_sets = tuple(sum(1 << int(i) for i in np.flatnonzero(snp_data.genotypes[:, j]))
                                  for j in range(self.num_variants))
        h = hashlib.sha256()
        h.update(snp_data.genotypes.tobytes())
        h.update(snp_data.positions.tobytes())
        h.update(repr((snp_data.haplotype_ids, snp_data.site_ids, self.sequence_length,
                       self.population_size, self.mutation_rate, self.recombination_rate,
                       self.reward_fn.C)).encode())
        self.dataset_fingerprint = h.hexdigest()
        self._validate_observation_support()
        self.likelihood_tracker = InfiniteSitesTracker(self)

    @staticmethod
    def _rate(value, name, positive=False):
        value = float(value)
        if not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
        return value

    def _validate_observation_support(self):
        groups = {}
        for i, x in enumerate(self.snp_data.positions):
            groups.setdefault(int(x) if self.recombination_rate > 0 else 0, []).append(i)
        for indices in groups.values():
            for offset, i in enumerate(indices):
                a = self.derived_sets[i]
                for j in indices[offset + 1:]:
                    b = self.derived_sets[j]
                    if a & b and a & ~b and b & ~a:
                        raise ValueError('No compatible ancestry within inseparable material: '
                                         f'SNP IDs {self.snp_data.site_ids[i]}, {self.snp_data.site_ids[j]}')

    @property
    def time_metadata(self):
        return self.time_env.metadata

    def _check_state(self, state):
        if state.dataset_fingerprint != self.dataset_fingerprint:
            raise ValueError('state belongs to different observations or environment parameters')

    def get_initial_state(self):
        nodes = {}
        for i in range(self.num_sequences):
            node = ARGLineage(i, MaterialSegments.full(self.num_blocks), self.num_blocks)
            self.likelihood_tracker.initialize_leaf(node)
            nodes[i] = node
        return ARGState(list(nodes.values()), nodes, self.num_sequences - 1,
                        np.full(self.num_variants, np.nan), self.dataset_fingerprint,
                        total_active_blocks=self.num_sequences * self.num_blocks)

    def get_active_counts(self, state):
        changes = np.zeros(self.num_blocks + 1, dtype=np.int64)
        for node in state.active_lineages:
            for l, r in node.material_segments.segments:
                changes[l] += 1
                changes[r] -= 1
        return changes.cumsum()[:-1]

    def is_terminal(self, state):
        # Total length alone cannot detect holes offset by multiply covered intervals.
        return bool(np.all(self.get_active_counts(state) == 1) and all(
            node.descendants is not None and all(bits == self.all_samples
                for _, _, bits in node.descendants.segments)
            for node in state.active_lineages))

    def _get_action_context(self, state):
        self._check_state(state)
        # Active indices can shift or be reordered by public callers. Retain the
        # immutable geometry/descendants in the signature so replacements also
        # invalidate the cache, even when action count and lineage IDs agree.
        lineages = tuple((n.node_id, n.material_segments, n.descendants) for n in state.active_lineages)
        signature = (self.dataset_fingerprint, state.is_done, self.rho, self.num_blocks, lineages)
        context = state._action_context
        if context is None or context.signature != signature:
            context = ActionContext.build(state, signature, self.compute_event_rates)
            state._action_context = context
        return context

    def enumerate_actions(self, state):
        context = self._get_action_context(state)
        # Public lists remain independently mutable; cached choices are frozen.
        return list(context.coal_actions), list(context.recomb_choices)

    def incompatible_sites(self, state, action):
        left, right = (state.active_lineages[i] for i in
                       (action.active_lineage_i, action.active_lineage_j))
        common = np.intersect1d(left.snp_indices, right.snp_indices, assume_unique=True)
        bad = []
        for i in common:
            x, target = self.snp_data.positions[i], self.derived_sets[i]
            bits = left.descendants.at(x) | right.descendants.at(x)
            if bits & target and bits & ~target and target & ~bits:
                bad.append(self.snp_data.site_ids[i])
        return tuple(bad)

    def enumerate_policy_actions(self, state):
        context = self._get_action_context(state)
        coal = context.coal_actions
        if coal and self.num_variants:
            cached = state._compatibility
            if (cached is None or cached.dataset_fingerprint != self.dataset_fingerprint
                    or cached.lineages != context.signature[-1]):
                cached = update_compatibility(self, context, cached)
                state._compatibility = cached
            coal = cached.coal_actions
        return list(coal), list(context.recomb_choices) if self.recombination_rate > 0 else []

    def compute_coalescence_actions(self, state):
        return list(self._get_action_context(state).coal_actions)

    def compute_recombination_actions(self, state):
        return list(self._get_action_context(state).recomb_choices)

    def compute_event_rates(self, actions):
        return priors.compute_event_rates(actions, rho=self.rho, num_blocks=self.num_blocks)

    def enumerate_prior_options(self, state):
        context = self._get_action_context(state)
        return PriorActionOptions(context.coal_actions, context.recomb_choices, context.rates.copy())

    def compute_event_probabilities(self, state, actions=None):
        # Deliberately ignore caller-provided filtered action lists.
        return priors.compute_event_probabilities(self.enumerate_prior_options(state).rates)

    def _validate_physical_action(self, state, action):
        self._check_state(state)
        if state.is_done:
            raise ValueError('terminal states have no forward actions')
        if not isinstance(action, (CoalescenceChoice, RecombinationChoice)):
            raise ValueError('expected a coalescence or recombination action')
        indices = [action.active_lineage_i]
        if isinstance(action, CoalescenceChoice):
            indices.append(action.active_lineage_j)
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral)
               or not 0 <= i < len(state.active_lineages) for i in indices):
            raise ValueError('invalid active lineage index')
        if len(indices) == 2 and indices[0] == indices[1]:
            raise ValueError('coalescence requires distinct lineages')
        if action.time_action is not None:
            raise ValueError('use continuous delta_t, not time_action')
        self.time_env.event_time(state.current_time, action.delta_t)
        if isinstance(action, RecombinationChoice):
            choice = self._get_action_context(state).recomb_by_lineage[action.active_lineage_i]
            if (self.recombination_rate == 0 or choice is None
                    or isinstance(action.breakpoint, (bool, np.bool_))
                    or not isinstance(action.breakpoint, Integral)
                    or not choice.span_start < action.breakpoint <= choice.span_end
                    or replace(action, breakpoint=None, delta_t=None) != choice):
                raise ValueError('invalid recombination span, integer breakpoint, or zero recombination rate')

    def compute_cwr_event_log_prior(self, state, combined_actions, action=None, rates=None):
        """Score from the environment's physical context, ignoring caller lists/rates."""
        if action is None:
            action = combined_actions
        self._validate_physical_action(state, action)
        context = self._get_action_context(state)
        physical = (context.coal_actions, context.recomb_choices)
        return priors.compute_cwr_event_log_prior(state.active_lineages, physical, action,
                    context.rates, time_env=self.time_env, time_policy=self.time_policy)

    def apply_action(self, state, action, log_prior=None):
        return self._apply_action(state, action, log_prior, inplace=False)[0]

    def step_owned_state(self, state, action):
        """Advance a rollout-owned state, returning its exact event prior.

        The caller must not retain earlier versions of this state. Public
        apply_action remains nonmutating for branching and external callers.
        """
        return self._apply_action(state, action, None, inplace=True)

    def _apply_action(self, state, action, log_prior, *, inplace):
        self._validate_physical_action(state, action)
        if isinstance(action, CoalescenceChoice):
            conflicts = self.incompatible_sites(state, action)
            if conflicts:
                raise IncompatibleActionError(conflicts)
        actual_prior = self.compute_cwr_event_log_prior(state, action)
        if log_prior is not None and (not math.isfinite(log_prior)
                                     or not math.isclose(log_prior, actual_prior, rel_tol=0, abs_tol=1e-10)):
            raise ValueError('supplied log_prior disagrees with the unmasked Hudson prior')
        result = state if inplace else state.clone()
        event_time = self.time_env.event_time(state.current_time, action.delta_t)
        indices = ([action.active_lineage_i, action.active_lineage_j]
                   if isinstance(action, CoalescenceChoice) else [action.active_lineage_i])
        children = [result.active_lineages[i] for i in indices]
        if len(children) == 2:
            material = children[0].material_segments.union(children[1].material_segments)
            parents = [ARGLineage(result.max_node_idx + 1, material, self.num_blocks,
                                 children=[c.node_id for c in children], event_type='coal', time=event_time)]
        else:
            materials = children[0].material_segments.split(action.breakpoint)
            parents = [ARGLineage(result.max_node_idx + offset + 1, material, self.num_blocks,
                                 children=[children[0].node_id], event_type='recomb', time=event_time,
                                 breakpoint=int(action.breakpoint), recombination_side=side)
                       for offset, (side, material) in enumerate(zip(('left', 'right'), materials))]
        for parent in parents:
            self.likelihood_tracker.parent(parent, children)
            self.likelihood_tracker.record(result, parent)
            result.all_nodes[parent.node_id] = parent
        for child in children:
            child.parents = [p.node_id for p in parents]
            child.messages = child.snp_indices = None
        result.active_lineages = [n for i, n in enumerate(result.active_lineages) if i not in indices] + parents
        result.max_node_idx = parents[-1].node_id
        result.current_time = event_time
        result.actions += (action,)
        result.accumulated_log_prior += actual_prior
        result.total_active_blocks = sum(n.material_count for n in result.active_lineages)
        result.partial_log_likelihood = self.likelihood_tracker.potential(result)
        result.is_done = self.is_terminal(result)
        result._action_context = None
        if result.is_done:
            result._compatibility = None
        if not math.isfinite(result.accumulated_log_prior):
            raise FloatingPointError('accumulated prior overflowed float64')
        result.log_reward = self.compute_terminal_log_reward(result) if result.is_done else None
        return result, actual_prior

    def apply_coalescence(self, state, action, log_prior=None):
        if not isinstance(action, CoalescenceChoice):
            raise ValueError('expected coalescence')
        return self.apply_action(state, action, log_prior)

    def apply_recombination(self, state, action, log_prior=None):
        if not isinstance(action, RecombinationChoice):
            raise ValueError('expected recombination')
        return self.apply_action(state, action, log_prior)

    def apply_actions(self, states, actions, log_priors=None):
        log_priors = [None] * len(states) if log_priors is None else log_priors
        if len(states) != len(actions) or len(states) != len(log_priors):
            raise ValueError('expected one action and prior per state')
        return [self.apply_action(s, a, p) for s, a, p in zip(states, actions, log_priors)]

    def compute_terminal_log_reward(self, state, log_likelihood=None):
        self._check_state(state)
        if not self.is_terminal(state):
            raise ValueError('terminal reward requires complete ancestry across the genome')
        if log_likelihood is None:
            if np.isnan(state.completed_site_lengths).any():
                raise ValueError('terminal state has unresolved SNP likelihoods; restore its caches')
            log_likelihood = self.likelihood_tracker.potential(state)
        return self.reward_fn(log_likelihood, state.accumulated_log_prior)

    def sample_compatible_step(self, state):
        physical = self.enumerate_prior_options(state)
        total = priors.total_event_rate(physical.rates)
        coal, recomb = self.enumerate_policy_actions(state)
        link_hazard = self.rho / (2 * self.num_blocks)
        recomb_weight = link_hazard * sum(a.breakpoint_count for a in recomb)
        allowed = len(coal) + recomb_weight
        if allowed <= 0:
            raise RuntimeError('no compatible action: cannot silently discard this state')
        if self.rng.random() * allowed < len(coal):
            action = self.rng.choice(coal)
            log_discrete = -math.log(allowed)
        else:
            action = priors.sample_recombination_prior_action(recomb, rng=self.rng)
            log_discrete = math.log(link_hazard) - math.log(allowed)
        action = replace(action, delta_t=self.time_env.sample_from_prior(total, self.rng))
        return CompatibleStep(action, log_discrete + self.time_env.log_density(action.delta_t, total),
                              self.compute_cwr_event_log_prior(state, action))

    def sample_prior_step(self, state):
        """Unconditioned physical proposal; apply_action may reject its coalescence."""
        physical = self.enumerate_prior_options(state)
        total = priors.total_event_rate(physical.rates)
        if self.rng.random() * total < physical.rates['lambda_coal']:
            action = self.rng.choice(physical.coal_actions)
        else:
            action = priors.sample_recombination_prior_action(physical.recomb_choices, rng=self.rng)
        action = replace(action, delta_t=self.time_env.sample_from_prior(total, self.rng))
        return action, self.compute_cwr_event_log_prior(state, action)

    def sample_compatible_trajectory(self, max_events=10000):
        if not isinstance(max_events, Integral) or max_events < 1:
            raise ValueError('max_events must be a positive integer')
        state, trajectory = self.get_initial_state(), SimpleTrajectory()
        for _ in range(max_events):
            step = self.sample_compatible_step(state)
            state = self.apply_action(state, step.action, step.log_prior)
            trajectory.update(step.action, step.log_prior, state.log_reward, log_proposal=step.log_proposal)
            if state.is_done:
                return state, trajectory
        raise RuntimeError(f'compatible rollout exceeded {max_events} events; no trajectory was discarded or retried')

    def replay(self, actions):
        state = self.get_initial_state()
        for action in actions:
            state = self.apply_action(state, action)
        return state

    def restore_state(self, state):
        self._check_state(state)
        result = state.clone()
        result._compatibility = None
        self.likelihood_tracker.restore(result)
        self._restore_history_and_prior(result)
        result.total_active_blocks = sum(n.material_count for n in result.active_lineages)
        result.is_done = self.is_terminal(result)
        result.log_reward = self.compute_terminal_log_reward(result) if result.is_done else None
        return result

    def _restore_history_and_prior(self, state):
        """Recover chronological events from stored ancestry, not cached action/prior fields."""
        from collections import defaultdict
        events = defaultdict(list)
        for node in state.all_nodes.values():
            if node.children:
                events[node.time].append(node)
        cursor = self.get_initial_state()
        history, scores = [], []
        for event_time, parents in sorted(events.items()):
            active = {n.node_id: i for i, n in enumerate(cursor.active_lineages)}
            dt = event_time - cursor.current_time
            if len(parents) == 1 and parents[0].event_type == 'coal' and len(parents[0].children) == 2:
                child_ids = parents[0].children
                i, j = sorted(active[c] for c in child_ids)
                action = CoalescenceChoice(i, j, delta_t=dt)
            elif len(parents) == 2 and all(p.event_type == 'recomb' for p in parents):
                parents = sorted(parents, key=lambda p: p.recombination_side)
                left, right = parents
                if (left.recombination_side != 'left' or right.recombination_side != 'right'
                        or left.children != right.children or len(left.children) != 1
                        or left.breakpoint != right.breakpoint):
                    raise ValueError('invalid paired recombination nodes')
                child_ids = left.children
                child = cursor.active_lineages[active[child_ids[0]]]
                if child.material_segments.split(left.breakpoint) != (left.material_segments, right.material_segments):
                    raise ValueError('recombination parents must partition child material')
                choice = next(a for a in self.enumerate_actions(cursor)[1]
                              if a.active_lineage_i == active[child_ids[0]])
                action = replace(choice, breakpoint=left.breakpoint, delta_t=dt)
            else:
                raise ValueError('stored graph must contain distinct timed coalescence or paired recombination events')
            scores.append(self.compute_cwr_event_log_prior(cursor, action))
            history.append(action)
            cursor.active_lineages = [n for n in cursor.active_lineages if n.node_id not in child_ids] + parents
            cursor.current_time = event_time
            cursor.is_done = self.is_terminal(cursor)
        if {n.node_id for n in cursor.active_lineages} != {n.node_id for n in state.active_lineages}:
            raise ValueError('stored active lineages disagree with ancestry events')
        state.actions = tuple(history)
        state.current_time = cursor.current_time
        # Chronological accumulation; recovered waits can differ by float64 roundoff.
        state.accumulated_log_prior = sum(scores)

    def _iter_arg_edge_intervals(self, state):
        for parent in state.all_nodes.values():
            for child_id in parent.children:
                child = state.all_nodes[child_id]
                for l, r in parent.material_segments.intersection(child.material_segments).segments:
                    yield parent.node_id, child_id, l, r

    def save_to_tree_sequence(self, state, output_path=None):
        import tskit
        self._check_state(state)
        if not self.is_terminal(state):
            raise ValueError('tree sequence export requires complete ancestry across the genome')
        tables = tskit.TableCollection(self.sequence_length)
        tables.time_units = 'generations'
        mapping = {}
        for key, node in sorted(state.all_nodes.items()):
            mapping[key] = tables.nodes.add_row(time=node.time * (2 * self.population_size),
                                flags=tskit.NODE_IS_SAMPLE if key < self.num_sequences else 0)
        for parent, child, left, right in self._iter_arg_edge_intervals(state):
            tables.edges.add_row(left, right, mapping[parent], mapping[child])
        tables.sort()
        ts = tables.tree_sequence()
        if output_path is not None:
            ts.dump(output_path)
        return ts

    def evaluate_terminal(self, state):
        return evaluate_infinite_sites(self.save_to_tree_sequence(state), self.snp_data,
                                       mutation_rate=self.mutation_rate)
