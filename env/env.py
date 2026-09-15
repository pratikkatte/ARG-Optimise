import math
import random
from typing import Any, Optional, Sequence
import torch
import numpy as np

from . import priors
from .evo import EvolutionModelTorch
from env.time_env import TimeEnvCwrExponential
from .actions import CoalescenceChoice, PriorActionOptions, RecombinationChoice
from .states import ARGLineage, ARGState, MaterialSegments


CHARACTERS_MAPS = {
    'DNA_WITH_GAP': {
        'A': [1., 0., 0., 0.],
        'C': [0., 1., 0., 0.],
        'G': [0., 0., 1., 0.],
        'T': [0., 0., 0., 1.],
        '-': [1., 1., 1., 1.],
        'N': [1., 1., 1., 1.]
    }
}

class SimpleTrajectory:
    """Compact trajectory history used when cloned ARG states are not needed."""

    def __init__(self):
        self.actions = []
        self.log_priors = []
        self.records = []
        self.log_reward = None

    def update(self, action, log_prior=None, log_reward=None, record=None, active_lineages=None):
        if not isinstance(action, (CoalescenceChoice, RecombinationChoice)):
            raise ValueError("Trajectory actions must be action dataclasses")
        self.actions.append(action)
        self.log_priors.append(log_prior)
        self.log_reward = log_reward
        if record is not None:
            self.records.append(record)

    def __len__(self):
        return len(self.actions)

class ARGReward:
    """
    Terminal reward helpers for constructed ARG states.
    """
    def __init__(self, C=3000):
        self.C = C

    def __call__(self, log_likelihood, accumulated_log_prior):
        return float(self.C + log_likelihood + accumulated_log_prior)

class SimpleARGEnvironment:
    """Hudson ARG environment with discrete genomic links and continuous waits.

    Terminal rewards combine the Hudson prior with the JC69 sequence likelihood.
    """

    def __init__(
        self,
        num_sequences: Optional[int] = None,
        sequence_length: Optional[int] = None,
        num_blocks: Optional[int] = None,
        population_size: float = 10000.0,
        effective_population_size: Optional[float] = None,
        mutation_rate: float = 2e-8,
        recombination_rate: float = 2e-8,
        rho: Optional[float] = None,
        sequences: Optional[Sequence[Any]] = None,
        seed: Optional[int] = 7,
        bp_per_blocks: int = 1,
        device: Optional[torch.device] = 'cpu',
        time_bins: Optional[int] = None,
        time_delta_bin_width: Optional[float] = None,
        time_policy: str = "cwr_exponential",
        arg_prior: str = "hudson",
    ):
        if arg_prior != 'hudson':
            raise ValueError("Only the Hudson ARG prior is supported; start a fresh Hudson run")
        if time_policy != 'cwr_exponential':
            raise ValueError('Hudson requires continuous waiting times')
        self.arg_prior = arg_prior
        self.sequences = list(sequences) if sequences is not None else None
        self.chars_dict = CHARACTERS_MAPS['DNA_WITH_GAP']
        self.event_types = ["coal", "recomb"]
        self.device = torch.device(device)
        self.flow_likelihood = None

        if self.sequences is not None:
            num_sequences = len(self.sequences)
            sequence_length = len(self.sequences[0])
            if any(len(sequence) != sequence_length for sequence in self.sequences):
                raise ValueError("all sequences must have length sequence_length")


        self.num_sequences = int(num_sequences)
        self.sequence_length = int(sequence_length)
        if num_blocks is None:
            self.num_blocks = int(sequence_length // bp_per_blocks)
        else:
            self.num_blocks = int(num_blocks)
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        ## Important parameters
        self.recombination_rate = float(recombination_rate)
        if effective_population_size is not None:
            population_size = effective_population_size
        self.population_size = float(population_size)
        self.mutation_rate = float(mutation_rate) ## where are we using this?

        self.rho = (
            float(rho)
            if rho is not None
            else 4 * self.population_size * self.recombination_rate * self.sequence_length
        )

        ## Time environment
        self.time_policy = time_policy
        self.time_env = TimeEnvCwrExponential()

        self.rng = random.Random(seed)

        ## Sequence arrays
        seq_arrays = np.array([self.seq2array(seq) for seq in self.sequences], dtype=np.float32)

        block_seq_arrays = np.empty(
            (self.num_sequences, self.num_blocks, seq_arrays.shape[-1]),
            dtype=np.float32,
        )
        for block_idx in range(self.num_blocks):
            site_start = int(round(block_idx * self.sequence_length / self.num_blocks))
            site_end = int(round((block_idx + 1) * self.sequence_length / self.num_blocks))
            if site_end <= site_start:
                raise ValueError(
                    "num_blocks must not create empty block intervals for sequence_length"
                )
            block_seq_arrays[:, block_idx, :] = seq_arrays[:, site_start:site_end, :].mean(axis=1)

        self.seq_arrays = torch.nn.Parameter(
            torch.tensor(seq_arrays, dtype=torch.float32, device=self.device),
            requires_grad=False,
        )
        self.block_seq_arrays = torch.nn.Parameter(
            torch.tensor(block_seq_arrays, dtype=torch.float32, device=self.device),
            requires_grad=False,
        )
        
        ## Evolution model
        self.evolution_model = EvolutionModelTorch(self)

        ## Reward function 
        self.reward_fn = ARGReward()

    @property
    def time_metadata(self):
        return self.time_env.metadata

    def seq2array(self, seq):
        seq = [self.chars_dict[x] for x in seq]
        data = np.array(seq)
        return data

    def _validate_timing(self, action):
        if action.time_action is not None or action.delta_t is None:
            raise ValueError("cwr_exponential actions require only delta_t timing")
        self.time_env.positive(action.delta_t, "wait")

    def resolve_event_time(self, state, action, rates):
        """Validate the pre-action rate and resolve the continuous wait."""
        self._validate_timing(action)
        priors.total_event_rate(rates)
        return self.time_env.event_time(state.current_time, action.delta_t)

    def timing_for_delta(self, delta_t, rates):
        return {"delta_t": self.time_env.positive(delta_t, "reconstructed wait")}

    def get_initial_state(self, track_likelihood=True):
        active_lineages = []
        all_nodes = {}
        material_segments = MaterialSegments.full(self.num_blocks)
        material_segments_list = [material_segments] * self.num_sequences
        partials_list = self._initial_lineages_partials_batch(material_segments_list)

        for node_id in range(self.num_sequences):
            # Here, each lineage starts at time 0.0
            lineage = ARGLineage(
                node_id=node_id,
                children=[],
                parents=[],
                material_segments=material_segments,
                num_blocks=self.num_blocks,
                partials=partials_list[node_id],
                sequences_indices=[node_id],
                time=0.0,
            )
            active_lineages.append(lineage)
            all_nodes[node_id] = lineage
     

        state = ARGState(
            active_lineages=active_lineages,
            all_nodes=all_nodes,
            max_node_idx=self.num_sequences - 1,
            log_reward=None,
            accumulated_log_prior=0.0,
            is_done=False,
            total_active_blocks=self.num_sequences * self.num_blocks,
            current_time=0.0,
        )
        state.is_done = self.is_terminal(state)
        if track_likelihood and self.flow_likelihood is not None:
            self.flow_likelihood.initialize(state)
        if state.is_done:
            log_likelihood = self.evolution_model.compute_arg_log_likelihood(state)
            state.log_reward = self.compute_terminal_log_reward(state, log_likelihood)
        return state

    def _initial_lineage_partials(self, node_id, material_segments):
        partials = self.block_seq_arrays[int(node_id)].detach().clone().float()
        return self.evolution_model.mask_partials(partials, material_segments)

    def _initial_lineages_partials_batch(self, material_segments_list):
        """Initialize each sequence's tip partials with its material mask."""
        num_lineages = len(material_segments_list)
        if num_lineages != self.num_sequences:
            raise ValueError(
                f"Expected {self.num_sequences} material segment sets, got {num_lineages}"
            )

        return [
            self._initial_lineage_partials(node_id, material_segments)
            for node_id, material_segments in enumerate(material_segments_list)
        ]

    def _require_lineage_partials(self, lineage):
        if lineage.partials is None:
            raise ValueError(f"ARG lineage {lineage.node_id} is missing partials")
        return self.evolution_model._as_partials_tensor(lineage.partials)

    def _transition_lineage_partials(self, lineage, parent_time):
        edge_time = float(parent_time) - float(lineage.time)
        if edge_time <= 0:
            raise ValueError(
                f"ARG node times must increase from child to parent: "
                f"parent_time={parent_time}, child={lineage.node_id} time={lineage.time}"
            )
        partials = self._require_lineage_partials(lineage)
        return self.evolution_model.transition_partials(partials, edge_time)

    def _coalesced_parent_partials(self, child_i, child_j, parent_segments, parent_time):
        reference = self._require_lineage_partials(child_i)
        combined = torch.ones_like(reference)
        has_material = torch.zeros(
            reference.shape[0],
            1,
            dtype=torch.bool,
            device=reference.device,
        )

        for child in (child_i, child_j):
            transitioned = self._transition_lineage_partials(child, parent_time)
            transitioned = self.evolution_model.normalize_partials(transitioned)
            weights = self.evolution_model.material_site_weights(
                child.material_segments,
                device=transitioned.device,
                dtype=transitioned.dtype,
            )
            child_has_material = weights[:, None] > 0
            child_partials = transitioned * weights[:, None]
            combined = torch.where(child_has_material, combined * child_partials, combined)
            has_material = has_material | child_has_material

        combined = torch.where(has_material, combined, torch.zeros_like(combined))
        combined = self.evolution_model.mask_partials(combined, parent_segments)
        return self.evolution_model.normalize_partials(combined)

    def _recombined_parent_partials(self, transitioned, parent_segments):
        """Mask transitioned child partials for one recombination parent."""
        masked = self.evolution_model.mask_partials(transitioned, parent_segments)
        return self.evolution_model.normalize_partials(masked)

    def get_active_counts(self, state):
        if not state.active_lineages:
            return np.zeros(self.num_blocks, dtype=int)
        counts = np.zeros(self.num_blocks, dtype=int)
        for lineage in state.active_lineages:
            for start, end in lineage.material_segments.segments:
                counts[start:end] += 1
        return counts

    def get_arg_sequence_segments(self, state):
        return self.evolution_model.get_arg_sequence_segments(state)

    def _iter_arg_edge_intervals(self, state):
        for parent_id in sorted(state.all_nodes):
            parent = state.all_nodes[parent_id]
            for child_id in parent.children:
                if child_id not in state.all_nodes:
                    raise ValueError(f"ARG node {parent_id} references missing child {child_id}")
                child = state.all_nodes[child_id]
                material_segments = parent.material_segments.intersection(child.material_segments)
                for left_block, right_block in material_segments.segments:
                    yield parent_id, child_id, left_block, right_block

    def _arg_edge_breakpoints(self, state):
        num_blocks = int(self.num_blocks)
        breakpoints = set()
        for _, _, left_block, right_block in self._iter_arg_edge_intervals(state):
            if 0 < left_block < num_blocks:
                breakpoints.add(int(left_block))
            if 0 < right_block < num_blocks:
                breakpoints.add(int(right_block))
        return breakpoints

    def _arg_recombination_events(self, state, breakpoints=None):
        num_blocks = int(self.num_blocks)
        if breakpoints is None:
            breakpoints = set()
        recomb_by_event = {}

        for node_id, lineage in state.all_nodes.items():
            if (
                lineage.event_type != "recomb"
                or lineage.breakpoint is None
                or not lineage.children
            ):
                continue

            breakpoint = int(lineage.breakpoint)
            if 0 < breakpoint < num_blocks:
                breakpoints.add(breakpoint)

            key = (int(lineage.children[0]), breakpoint)
            grouped = recomb_by_event.setdefault(
                key,
                {"left": None, "right": None, "other": []},
            )
            if lineage.recombination_side == "left":
                grouped["left"] = int(node_id)
            elif lineage.recombination_side == "right":
                grouped["right"] = int(node_id)
            else:
                grouped["other"].append(int(node_id))

        recombination_events = []
        for (child_id, breakpoint), grouped in sorted(
            recomb_by_event.items(),
            key=lambda item: (item[0][1], item[0][0]),
        ):
            parent_ids = []
            if grouped["left"] is not None:
                parent_ids.append(grouped["left"])
            if grouped["right"] is not None:
                parent_ids.append(grouped["right"])
            parent_ids.extend(sorted(grouped["other"]))
            recombination_events.append(
                {
                    "child_id": child_id,
                    "breakpoint": breakpoint,
                    "parent_ids": parent_ids,
                }
            )
        return recombination_events

    def save_to_tree_sequence(self, state, output_path=None):
        """Convert a terminal ARG state to a tskit TreeSequence.

        The exported topology contains ancestry edges only. Stored ARG node
        times are internal t/(2Ne) values and are exported in generations to
        match msprime tree sequences.
        """
        if not self.is_terminal(state):
            raise ValueError("terminal_state_to_tree_sequence requires a terminal ARGState")
        if self.num_blocks <= 0 or self.sequence_length <= 0:
            raise ValueError("sequence_length and num_blocks must be positive")

        try:
            import tskit
        except ImportError as exc:
            raise ImportError(
                "tskit is required to export ARG states to .trees files. "
                "Install it with `pip install tskit`."
            ) from exc

        node_times = self._tskit_node_times(state)
        tables = tskit.TableCollection(sequence_length=float(self.sequence_length))
        tables.time_units = "generations"
        sample_node_ids = set(range(self.num_sequences))
        tskit_node_ids = {}

        for node_id in sorted(state.all_nodes):
            flags = tskit.NODE_IS_SAMPLE if node_id in sample_node_ids else 0
            tskit_node_ids[node_id] = tables.nodes.add_row(
                flags=flags,
                time=node_times[node_id],
            )

        for parent_id, child_id, left_block, right_block in self._iter_arg_edge_intervals(state):
            left = self._block_to_sequence_coordinate(left_block)
            right = self._block_to_sequence_coordinate(right_block)
            if left < right:
                tables.edges.add_row(
                    left=left,
                    right=right,
                    parent=tskit_node_ids[parent_id],
                    child=tskit_node_ids[child_id],
                )

        tables.sort()
        tree_sequence = tables.tree_sequence()
        if output_path is not None:
            tree_sequence.dump(output_path)
        return tree_sequence

    def _tskit_node_times(self, state): 
        time_scale = 2.0 * self.population_size
        node_times = {
            node_id: float(node.time) * time_scale
            for node_id, node in state.all_nodes.items()
        }
        for parent_id, parent in state.all_nodes.items():
            for child_id in parent.children:
                if node_times[parent_id] <= node_times[child_id]:
                    raise ValueError(
                        f"learned ARG node times must satisfy parent > child: "
                        f"parent={parent_id} child={child_id}"
                    )
        return node_times

    def _block_to_sequence_coordinate(self, block_index):
        return float(block_index) * float(self.sequence_length) / float(self.num_blocks)

    def compute_terminal_log_reward(self, state, log_likelihood=None):
        """Return the posterior terminal target for a completed ARG."""
        if not self.is_terminal(state):
            raise ValueError("terminal reward requires a terminal ARGState")
        if log_likelihood is None:
            log_likelihood = self.evolution_model.compute_arg_log_likelihood(state)
        log_reward = self.reward_fn(log_likelihood, state.accumulated_log_prior)
        return log_reward

    def compute_coalescence_actions(self, state):
        return list(CoalescenceChoice.enumerate_from_active_lineages(
            state.active_lineages))

    def compute_recombination_actions(self, state):
        return list(RecombinationChoice.enumerate_from_active_lineages(state.active_lineages))

    def enumerate_prior_options(self, state):
        coal_actions, recomb_actions = self.enumerate_actions(state)
        rates = self.compute_event_rates((coal_actions, recomb_actions))
        state.rates = rates
        prior_options = PriorActionOptions(
            coal_actions=tuple(coal_actions),
            recomb_choices=tuple(recomb_actions),
            rates=rates,
        )
        state.prior_options = prior_options
        return prior_options

    def action_options_from_prior_options(self, prior_options):
        actions = []
        if prior_options.rates["lambda_coal"] > 0:
            actions.extend(prior_options.coal_actions)
        if prior_options.rates["lambda_recomb"] > 0:
            actions.extend(choice for choice in prior_options.recomb_choices if choice.breakpoint_count > 0)
        return actions


    def is_terminal(self, state):
        if state.total_active_blocks is None:
            raise ValueError("total_active_blocks is required for terminal check")
        else:
            result = int(state.total_active_blocks) == self.num_blocks
            # bool(np.all(self.get_active_counts(state) == 1)) ## another way, realtime compute. 
            return result

    def _finalize_transition_state(self, next_state, log_prior):
        if log_prior is not None:
            next_state.accumulated_log_prior += log_prior
        next_state.is_done = self.is_terminal(next_state)
        if next_state.is_done:
            log_likelihood = self.evolution_model.compute_arg_log_likelihood(next_state)
            next_state.log_reward = self.compute_terminal_log_reward(next_state, log_likelihood)
        else:
            next_state.log_reward = None
        if (
            not math.isfinite(next_state.accumulated_log_prior)
            or (next_state.log_reward is not None and not math.isfinite(next_state.log_reward))
        ):
            raise ValueError("non-finite continuous accumulated prior or reward")
        return next_state

    def apply_coalescence(self, state, action, log_prior=None):

        rates = self._get_state_rates(state)

        next_state = state.clone(copy_partials=False)
        i = action.active_lineage_i
        j = action.active_lineage_j

        child_i = next_state.active_lineages[i].clone(copy_partials=False, copy_mask=False)
        child_j = next_state.active_lineages[j].clone(copy_partials=False, copy_mask=False)

        parent_id = next_state.max_node_idx + 1
        parent_segments = child_i.material_segments.union(child_j.material_segments)
        overlap_count = child_i.material_segments.intersection_count(child_j.material_segments)
        parent_time = self.resolve_event_time(state, action, rates)
        next_state.current_time = parent_time
        parent_partials = self._coalesced_parent_partials(
            child_i,
            child_j,
            parent_segments,
            parent_time,
        )
        parent = ARGLineage(
            node_id=parent_id,
            children=[child_i.node_id, child_j.node_id],
            parents=[],
            material_segments=parent_segments,
            num_blocks=self.num_blocks,
            partials=parent_partials,
            sequences_indices=sorted(set(child_i.sequences_indices + child_j.sequences_indices)),
            event_type="coal",
            time=parent_time,
        )

        child_i.parents.append(parent.node_id)
        child_j.parents.append(parent.node_id)
        if self.flow_likelihood is not None and next_state.partial_log_likelihood is not None:
            next_state.partial_log_likelihood += self.flow_likelihood.parent(parent, [child_i, child_j])
        child_i.partials = None
        child_j.partials = None
        child_i.likelihood_partials = child_j.likelihood_partials = None
        next_state.active_lineages[i] = child_i
        next_state.active_lineages[j] = child_j
        next_state.all_nodes[child_i.node_id] = child_i
        next_state.all_nodes[child_j.node_id] = child_j
        next_state.all_nodes[parent.node_id] = parent
        next_state.active_lineages = [
            lineage for idx, lineage in enumerate(next_state.active_lineages) if idx not in (i, j)
        ]
        next_state.active_lineages.append(parent)
        next_state.max_node_idx = parent.node_id
        if next_state.total_active_blocks is not None:
            next_state.total_active_blocks = int(next_state.total_active_blocks) - overlap_count
        return self._finalize_transition_state(next_state, log_prior)

    def apply_recombination(self, state, action, log_prior=None):
        rates = self._get_state_rates(state)

        next_state = state.clone(copy_partials=False)
        current_lineage_idx = action.active_lineage_i
        breakpoint = action.breakpoint
        child = next_state.active_lineages[current_lineage_idx].clone(copy_partials=False, copy_mask=False)
        left_segments, right_segments = child.material_segments.split(breakpoint)

        left_parent_id = next_state.max_node_idx + 1
        right_parent_id = next_state.max_node_idx + 2
        event_time = self.resolve_event_time(state, action, rates)
        next_state.current_time = event_time
        transitioned = self._transition_lineage_partials(child, event_time)
        left_partials = self._recombined_parent_partials(transitioned, left_segments)
        right_partials = self._recombined_parent_partials(transitioned, right_segments)
        left_parent = ARGLineage(
            node_id=left_parent_id,
            children=[child.node_id],
            parents=[],
            material_segments=left_segments,
            num_blocks=self.num_blocks,
            partials=left_partials,
            sequences_indices=list(child.sequences_indices),
            event_type="recomb",
            breakpoint=breakpoint,
            recombination_side="left",
            time=event_time,
        )
        right_parent = ARGLineage(
            node_id=right_parent_id,
            children=[child.node_id],
            parents=[],
            material_segments=right_segments,
            num_blocks=self.num_blocks,
            partials=right_partials,
            sequences_indices=list(child.sequences_indices),
            event_type="recomb",
            breakpoint=breakpoint,
            recombination_side="right",
            time=event_time,
        )

        child.parents = [left_parent.node_id, right_parent.node_id]
        if self.flow_likelihood is not None and next_state.partial_log_likelihood is not None:
            next_state.partial_log_likelihood += self.flow_likelihood.parent(left_parent, [child])
            next_state.partial_log_likelihood += self.flow_likelihood.parent(right_parent, [child])
        child.partials = None
        child.likelihood_partials = None
        next_state.all_nodes[child.node_id] = child
        next_state.all_nodes[left_parent.node_id] = left_parent
        next_state.all_nodes[right_parent.node_id] = right_parent
        next_state.active_lineages = [
            lineage for idx, lineage in enumerate(next_state.active_lineages) if idx != current_lineage_idx
        ]
        next_state.active_lineages.extend([left_parent, right_parent])
        next_state.max_node_idx = right_parent.node_id
        return self._finalize_transition_state(next_state, log_prior)

    def apply_action(self, state, action, log_prior=None):
        if isinstance(action, RecombinationChoice):
            return self.apply_recombination(
                state,
                action,
                log_prior
            )
        elif isinstance(action, CoalescenceChoice):
            return self.apply_coalescence(
                state,
                action,
                log_prior
            )
        else:
            raise ValueError(f"Unknown action event_type: {action}")

    def _get_state_rates(self, state, actions=None):
        """Compute and cache rates when the state has none."""
        if state.rates is None:
            if actions is None:
                actions = self.enumerate_actions(state)
            state.rates = self.compute_event_rates(actions)
        return state.rates

    def compute_event_rates(self, actions):
        return priors.compute_event_rates(
            actions, rho=self.rho, num_blocks=self.num_blocks)

    def compute_event_probabilities(self, state, actions=None):
        if actions is None:
            actions = self.enumerate_actions(state)
        rates = self.compute_event_rates(actions)
        state.rates = rates
        return priors.compute_event_probabilities(rates)

    def enumerate_actions(self, state):
        coal_actions = self.compute_coalescence_actions(state)
        recomb_actions = self.compute_recombination_actions(state)
        return coal_actions, recomb_actions

    def sample_prior_step(self, state):
        """Sample a timed prior action using the shared prior implementation."""
        actions = self.enumerate_actions(state)
        rates = self.compute_event_rates(actions)
        state.rates = rates
        return priors.sample_prior_step(
            state.active_lineages, actions, rates,
            time_env=self.time_env, time_policy=self.time_policy,
            rng=self.rng, event_rng=np.random)

    def sample_log_rewards(self, num_trajs, verbose=True):
        """Sample prior rollouts sequentially and return terminal log rewards."""
        log_rewards = []
        for traj_idx in range(num_trajs):
            if verbose:
                print(
                    f"Sampling prior trajectory {traj_idx + 1}/{num_trajs} for log Z init..."
                )
            state = self.get_initial_state()
            while not state.is_done:
                action, log_prior = self.sample_prior_step(state)
                state = self.apply_action(state, action, log_prior=log_prior)
            log_rewards.append(state.log_reward)
        return log_rewards

    def compute_cwr_event_log_prior(self, state, combined_actions, action=None, rates=None):
        """Validate timing and resolve state rates before scoring in priors."""
        if action is None:
            action = combined_actions
            combined_actions = self.enumerate_actions(state)
        if not isinstance(action, (CoalescenceChoice, RecombinationChoice)):
            raise ValueError("Invalid ARG action")
        self._validate_timing(action)

        if rates is None:
            rates = self._get_state_rates(state, combined_actions)
        state.rates = rates
        
        return priors.compute_cwr_event_log_prior(
            state.active_lineages, combined_actions, action, rates,
            time_env=self.time_env, time_policy=self.time_policy)

    def prepare_state_rollout_inputs(
        self,
        states,
        random_spec=None,
        event_policy="cwr",
    ):
        batch_size = len(states)
        if batch_size == 0:
            raise ValueError("states must contain at least one ARGState")

        if event_policy == "cwr_residual":
            event_actions = [self.enumerate_actions(state) for state in states]
            prior_probs = [
                self.compute_event_probabilities(state, actions)
                for state, actions in zip(states, event_actions)
            ]
            return {
                "states": states,
                "event_actions": event_actions,
                "event_prior_probs": [[p[e] for e in self.event_types] for p in prior_probs],
                "random_spec": random_spec,
            }
        if event_policy != "cwr":
            raise ValueError(f"Unknown event_policy: {event_policy}")

        event = {}
        input_actions = []
        for idx, state in enumerate(states):
            coal_actions, recomb_actions = self.enumerate_actions(state)
            event_probs = self.compute_event_probabilities(state, (coal_actions, recomb_actions))
            chosen_event_type = priors.sample_event_type(event_probs, rng=np.random)
            if chosen_event_type == "coal":
                input_actions.append(coal_actions)
            else:
                input_actions.append(recomb_actions)

            event[idx] = {}
            event[idx]["event_type"] = chosen_event_type
            event[idx]["probability"] = event_probs[chosen_event_type]

        input_dict = {
            "states": states,
            "event": event,
            "input_actions": input_actions,
            "random_spec": random_spec,
        }

        return input_dict
