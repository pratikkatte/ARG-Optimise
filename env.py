import copy
import math
import numbers
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch

import numpy as np

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

@dataclass
class ARGLineage:
    node_id: int
    children: List[int] = field(default_factory=list)
    parents: List[int] = field(default_factory=list)
    material_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    partials: Optional[Any] = None
    sequences_indices: List[int] = field(default_factory=list)
    event_type: Optional[str] = None
    breakpoint: Optional[int] = None
    recombination_side: Optional[str] = None

    def __post_init__(self):
        self.children = list(self.children)
        self.parents = list(self.parents)
        self.material_mask = np.asarray(self.material_mask, dtype=bool).copy()
        self.sequences_indices = list(self.sequences_indices)

    def clone(self):
        return ARGLineage(
            node_id=self.node_id,
            children=list(self.children),
            parents=list(self.parents),
            material_mask=self.material_mask.copy(),
            partials=copy.deepcopy(self.partials),
            sequences_indices=list(self.sequences_indices),
            event_type=self.event_type,
            breakpoint=self.breakpoint,
            recombination_side=self.recombination_side,
        )

@dataclass
class ARGState:
    active_lineages: List[ARGLineage]
    all_nodes: Dict[int, ARGLineage]
    max_node_idx: int
    log_reward: Optional[float] = None
    accumulated_log_prior: float = 0.0
    is_done: bool = False
    action_options: Tuple[List[Dict[str, Any]], List[Tuple[int, int, List[int]]], List[Dict[str, Any]]] = None
    rates: Optional[Dict[str, float]] = None

    def clone(self):
        all_nodes = {node_id: lineage.clone() for node_id, lineage in self.all_nodes.items()}
        active_lineages = [all_nodes[lineage.node_id] for lineage in self.active_lineages]
        return ARGState(
            active_lineages=active_lineages,
            all_nodes=all_nodes,
            max_node_idx=self.max_node_idx,
            log_reward=self.log_reward,
            accumulated_log_prior=self.accumulated_log_prior,
            is_done=self.is_done,
        )

class EvolutionModelTorch(torch.nn.Module):
    """Fixed-edge JC69 likelihood model for constructed ARG states."""

    _PROB_FLOOR = 1e-300
    _NON_FINITE_LOG_LIKELIHOOD = -1e6

    def __init__(self, env):
        super().__init__()
        self.env = env

    def compute_arg_log_likelihood(self, state):
        """Compute the JC69 sequence log likelihood of a terminal ARG.

        Each marginal segment induced by recombination breakpoints is scored
        with Felsenstein pruning. All traversed ARG edges use the environment's
        fixed edge length.
        """

        self._require_terminal(state)

        if self.env.sequences is None:
            return 0.0

        if self.env.fixed_edge_length < 0:
            raise ValueError("fixed_edge_length must be non-negative for likelihood scoring")

        seq_arrays = self._seq_arrays_numpy()
        transition_matrix = self._jc69_transition_matrix(self.env.fixed_edge_length)
        log_likelihood = 0.0

        for block_start, block_end in self.env.get_arg_sequence_segments(state)["segments"]:
            site_start = self._block_to_site(block_start)
            site_end = self._block_to_site(block_end)
            if site_start >= site_end:
                continue

            root_id = self._segment_root_node_id(state, block_start, block_end)
            root_partials, root_log_scale = self._compute_segment_partials(
                state,
                root_id,
                block_start,
                block_end,
                site_start,
                site_end,
                seq_arrays,
                transition_matrix,
                memo={},
            )
            site_probs = np.sum(root_partials * 0.25, axis=1)
            site_probs = np.maximum(site_probs, self._PROB_FLOOR)
            with np.errstate(divide="ignore", invalid="ignore"):
                log_likelihood += float(np.log(site_probs).sum() + root_log_scale.sum())

        if not math.isfinite(log_likelihood):
            return self._NON_FINITE_LOG_LIKELIHOOD
        return float(log_likelihood)

    def _require_terminal(self, state):
        if not self.env.is_terminal(state):
            raise ValueError("ARG likelihood and posterior reward require a terminal ARGState")

    def _seq_arrays_numpy(self):
        return self.env.seq_arrays.detach().cpu().numpy().astype(float, copy=False)

    def _jc69_transition_matrix(self, edge_length):
        same_prob = 0.25 + 0.75 * math.exp(-4.0 * float(edge_length) / 3.0)
        diff_prob = 0.25 - 0.25 * math.exp(-4.0 * float(edge_length) / 3.0)
        transition_matrix = np.full((4, 4), diff_prob, dtype=float)
        np.fill_diagonal(transition_matrix, same_prob)
        return transition_matrix

    def _block_to_site(self, block_index):
        site_fraction = (
            float(block_index) * float(self.env.sequence_length) / float(self.env.num_blocks)
        )
        return int(round(site_fraction))

    def _segment_root_node_id(self, state, block_start, block_end):
        roots = [
            lineage.node_id
            for lineage in state.active_lineages
            if np.all(lineage.material_mask[block_start:block_end])
        ]
        if len(roots) != 1:
            raise ValueError(
                "terminal ARG must have exactly one active root covering each sequence segment"
            )
        return roots[0]

    def _normalize_leaf_partials(self, partials):
        """Normalize leaf partials per site (phylo NORMALIZE_LIKELIHOOD behavior)."""
        normalized = np.full_like(partials, 0.25)
        row_sums = partials.sum(axis=-1, keepdims=True)
        np.divide(partials, row_sums, out=normalized, where=row_sums > 0)
        return normalized

    def _rescale_partials(self, partials, log_scale):
        scale = partials.max(axis=1)
        scale = np.maximum(scale, self._PROB_FLOOR)
        log_scale = log_scale + np.log(scale)
        partials = partials / scale[:, np.newaxis]
        return partials, log_scale

    def _compute_segment_partials(
        self,
        state,
        node_id,
        block_start,
        block_end,
        site_start,
        site_end,
        seq_arrays,
        transition_matrix,
        memo,
    ):
        if node_id in memo:
            return memo[node_id]

        node = state.all_nodes[node_id]
        if node_id < self.env.num_sequences:
            partials = self._normalize_leaf_partials(
                seq_arrays[node_id, site_start:site_end].copy()
            )
            log_scale = np.zeros(site_end - site_start, dtype=float)
            partials, log_scale = self._rescale_partials(partials, log_scale)
            result = (partials, log_scale)
            memo[node_id] = result
            return result

        relevant_children = [
            child_id
            for child_id in node.children
            if self._edge_covers_segment(state, node_id, child_id, block_start, block_end)
        ]

        if not relevant_children:
            raise ValueError(f"ARG node {node_id} has no descendants for the requested segment")

        partials = np.ones((site_end - site_start, seq_arrays.shape[-1]), dtype=float)
        log_scale = np.zeros(site_end - site_start, dtype=float)
        for child_id in relevant_children:
            child_partials, child_log_scale = self._compute_segment_partials(
                state,
                child_id,
                block_start,
                block_end,
                site_start,
                site_end,
                seq_arrays,
                transition_matrix,
                memo,
            )
            child_partials = np.maximum(child_partials, self._PROB_FLOOR)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                partials *= child_partials @ transition_matrix.T
            log_scale += child_log_scale
            partials, log_scale = self._rescale_partials(partials, log_scale)
        result = (partials, log_scale)
        memo[node_id] = result
        return result

    def _edge_covers_segment(self, state, parent_id, child_id, block_start, block_end):
        parent = state.all_nodes[parent_id]
        child = state.all_nodes[child_id]
        edge_mask = parent.material_mask & child.material_mask
        return bool(np.any(edge_mask[block_start:block_end]))


class ARGReward:
    """
    Terminal reward helpers for constructed ARG states.
    """

    def __init__(self):
        pass

    def __call__(self, state):
        return self.compute_terminal_posterior_log_reward(state)

    def compute_terminal_posterior_log_reward(self, log_likelihood):
        return log_likelihood

class SimpleARGEnvironment:
    """
    Minimal discrete coalescent-with-recombination ARG prototype.

    This intentionally avoids eete3, continuous breakpoints, and full continuous
    coalescent-with-recombination simulation. Terminal states are rewarded by the
    sampled event prior plus a fixed-edge JC69 sequence likelihood.
    """

    def __init__(
        self,
        num_sequences: Optional[int] = None,
        sequence_length: Optional[int] = None,
        num_blocks: Optional[int] = None,
        rho: float = 1.0,
        fixed_edge_length: float = 0.02,
        sequences: Optional[Sequence[Any]] = None,
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
    ):
        self.sequences = list(sequences) if sequences is not None else None
        self.chars_dict = CHARACTERS_MAPS['DNA_WITH_GAP']

        if self.sequences is not None:
            if num_sequences is None:
                num_sequences = len(self.sequences)
            elif int(num_sequences) != len(self.sequences):
                raise ValueError("num_sequences must equal len(sequences)")
            if sequence_length is None and self.sequences:
                sequence_length = len(self.sequences[0])
            if any(len(sequence) != sequence_length for sequence in self.sequences):
                raise ValueError("all sequences must have length sequence_length")

        if sequence_length is None and num_blocks is not None:
            sequence_length = num_blocks
        if num_blocks is None:
            num_blocks = sequence_length
        if num_sequences is None:
            raise ValueError("num_sequences is required when sequences are not provided")
        if sequence_length is None:
            raise ValueError("sequence_length or num_blocks is required")
        if num_blocks > sequence_length:
            raise ValueError("num_blocks must be less than or equal to sequence_length")

        self.num_sequences = int(num_sequences)
        self.sequence_length = int(sequence_length)
        self.num_blocks = int(num_blocks)
        self.rho = float(rho)
        self.fixed_edge_length = float(fixed_edge_length)
        self.rng = rng if rng is not None else random.Random(seed)
        self.block_indices = np.arange(self.num_blocks)

        if self.sequences is None:
            num_chars = len(next(iter(self.chars_dict.values())))
            seq_arrays = np.zeros(
                (self.num_sequences, self.sequence_length, num_chars),
                dtype=float,
            )
        else:
            seq_arrays = np.array([self.seq2array(seq) for seq in self.sequences])
        self.seq_arrays = torch.nn.Parameter(torch.tensor(seq_arrays), requires_grad=False)
        self.evolution_model = EvolutionModelTorch(self)
        self.reward_fn = ARGReward()

    def seq2array(self, seq):
        seq = [self.chars_dict[x] for x in seq]
        data = np.array(seq)
        return data

    def get_initial_state(self):
        active_lineages = []
        all_nodes = {}
        for node_id in range(self.num_sequences):
            lineage = ARGLineage(
                node_id=node_id,
                children=[],
                parents=[],
                material_mask=np.ones(self.num_blocks, dtype=bool),
                partials=None,
                sequences_indices=[node_id],
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
        )
        state.is_done = self.is_terminal(state)
        if state.is_done:
            log_likelihood = self.evolution_model.compute_arg_log_likelihood(state)
            state.log_reward = self.reward_fn(log_likelihood)+state.accumulated_log_prior
        return state

    def get_active_counts(self, state):
        if not state.active_lineages:
            return np.zeros(self.num_blocks, dtype=int)
        masks = [lineage.material_mask.astype(int) for lineage in state.active_lineages]
        return np.sum(masks, axis=0)

    def get_recombination_breakpoints(self, state):
        """Return recombination split metadata recorded in an ARG state."""
        breakpoints_by_event = {}
        for lineage in sorted(state.all_nodes.values(), key=lambda node: node.node_id):
            if lineage.event_type != "recomb":
                continue
            child_node_id = lineage.children[0] if lineage.children else None
            key = (child_node_id, lineage.breakpoint)
            event = breakpoints_by_event.setdefault(
                key,
                {
                    "child_node_id": child_node_id,
                    "breakpoint": lineage.breakpoint,
                    "parent_node_ids": {},
                    "segments_by_side": {},
                    "blocks_by_side": {},
                    "sequences_indices": list(lineage.sequences_indices),
                },
            )
            side = lineage.recombination_side or f"parent_{lineage.node_id}"
            event["parent_node_ids"][side] = lineage.node_id
            event["segments_by_side"][side] = self.mask_to_segments(lineage.material_mask)
            event["blocks_by_side"][side] = np.flatnonzero(lineage.material_mask).tolist()
        return list(breakpoints_by_event.values())

    def get_arg_sequence_segments(self, state):
        """Return the sequence-wide partition induced by ARG recombinations.

        This reports unique genomic segments for the whole constructed ARG, not
        the currently active lineage masks. A recombination event creates left
        and right parent nodes with the same breakpoint, but that breakpoint is
        counted once in the sequence partition.
        """
        recombination_events = self.get_recombination_breakpoints(state)
        breakpoints = sorted(
            {
                event["breakpoint"]
                for event in recombination_events
                if event["breakpoint"] is not None
            }
        )
        boundaries = [0] + breakpoints + [self.num_blocks]
        segments = [
            (start, end)
            for start, end in zip(boundaries, boundaries[1:])
            if start < end
        ]
        return {
            "breakpoints": breakpoints,
            "segments": segments,
            "num_segments": len(segments),
            "recombination_events": recombination_events,
        }

    def mask_to_segments(self, material_mask):
        """Convert a boolean material mask into contiguous half-open segments."""
        mask = np.asarray(material_mask, dtype=bool)
        segments = []
        start = None
        for block_i, has_material in enumerate(mask):
            if has_material and start is None:
                start = block_i
            elif not has_material and start is not None:
                segments.append((start, block_i))
                start = None
        if start is not None:
            segments.append((start, len(mask)))
        return segments

    def save_to_tree_sequence(self, state, output_path=None):
        """Convert a terminal ARG state to a tskit TreeSequence.

        The exported topology contains ancestry edges only. Synthetic node times
        are derived from graph depth because ARG rollout states do not store
        continuous event times.
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

        node_times = self._synthetic_tskit_node_times(state)
        tables = tskit.TableCollection(sequence_length=float(self.sequence_length))
        sample_node_ids = set(range(self.num_sequences))
        tskit_node_ids = {}

        for node_id in sorted(state.all_nodes):
            flags = tskit.NODE_IS_SAMPLE if node_id in sample_node_ids else 0
            tskit_node_ids[node_id] = tables.nodes.add_row(
                flags=flags,
                time=node_times[node_id],
            )

        for parent_id in sorted(state.all_nodes):
            parent = state.all_nodes[parent_id]
            for child_id in parent.children:
                if child_id not in state.all_nodes:
                    raise ValueError(f"ARG node {parent_id} references missing child {child_id}")
                child = state.all_nodes[child_id]
                material_mask = parent.material_mask & child.material_mask
                for left_block, right_block in self.mask_to_segments(material_mask):
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

    def terminal_state_to_tree_sequence(self, state, output_path=None):
        """Compatibility wrapper for exporting a terminal ARG as a tree sequence."""
        return self.save_to_tree_sequence(state, output_path=output_path)

    def _synthetic_tskit_node_times(self, state):
        if self.fixed_edge_length <= 0:
            raise ValueError("fixed_edge_length must be positive to export a valid tree sequence")

        node_times = {}
        sample_node_ids = set(range(self.num_sequences))
        for node_id in sorted(state.all_nodes):
            node = state.all_nodes[node_id]
            if node_id in sample_node_ids:
                node_times[node_id] = 0.0
                continue
            if not node.children:
                raise ValueError(f"Non-sample ARG node {node_id} has no children")
            try:
                child_times = [node_times[child_id] for child_id in node.children]
            except KeyError as exc:
                raise ValueError(
                    "ARG node ids must be topologically ordered so children are created "
                    "before their parents"
                ) from exc
            node_times[node_id] = max(child_times) + self.fixed_edge_length
        return node_times

    def _block_to_sequence_coordinate(self, block_index):
        return float(block_index) * float(self.sequence_length) / float(self.num_blocks)

    def compute_arg_log_likelihood(self, state):
        """Compute the terminal ARG sequence log likelihood under fixed-edge JC69."""
        return self.evolution_model.compute_arg_log_likelihood(state)

    def is_terminal(self, state):
        return bool(np.all(self.get_active_counts(state) == 1))

    def _valid_breakpoints_for_lineage(self, state, active_lineage_i):
        span = self._material_span(state.active_lineages[active_lineage_i].material_mask)
        if span is None:
            return []
        first_block, last_block, _ = span
        return list(range(first_block + 1, last_block + 1))

    def is_valid_coalescent_action(self, state, action):
        if action.get("event_type") != "coal":
            return False
        i = action.get("active_lineage_i")
        j = action.get("active_lineage_j")
        if not self._is_active_index(state, i) or not self._is_active_index(state, j):
            return False
        if i == j:
            return False
        mask_i = state.active_lineages[i].material_mask
        mask_j = state.active_lineages[j].material_mask
        return bool(np.any(mask_i & mask_j))

    def enumerate_action_options(self, state):
        if state.action_options is not None:
            return state.action_options[0], state.action_options[1], state.action_options[2]
        coal_actions = []
        recomb_weights = []
        recomb_actions = []
        if self.is_terminal(state):
            state.action_options = (coal_actions, recomb_weights, recomb_actions)
            state.rates = self.compute_event_rates(state)
            return coal_actions, recomb_weights, recomb_actions
        for i, lineage in enumerate(state.active_lineages):
            valid_breakpoints = self._valid_breakpoints_for_lineage(state, i)
            if valid_breakpoints:
                recomb_weights.append((i, int(lineage.material_mask.sum()), valid_breakpoints))
                for breakpoint in valid_breakpoints:
                    recomb_actions.append({"event_type": "recomb", "active_lineage_i": i, "breakpoint": breakpoint})
            for j in range(i + 1, len(state.active_lineages)):
                action = {"event_type": "coal", "active_lineage_i": i, "active_lineage_j": j}
                if self.is_valid_coalescent_action(state, action):
                    coal_actions.append(action)

        state.action_options = (coal_actions, recomb_weights, recomb_actions)
        rates = self.compute_event_rates(state)
        state.rates = rates
        return coal_actions, recomb_weights, recomb_actions

    def enumerate_actions(self, state, separate_coal_and_recomb=False):
        coal_actions, _, recomb_actions = self.enumerate_action_options(state)
        if separate_coal_and_recomb:
            return coal_actions, recomb_actions
        else:
            return coal_actions + recomb_actions

    def enumerate_coalescent_actions(self, state):
        coal_actions, _, _ = self.enumerate_action_options(state)
        return coal_actions

    def enumerate_recombination_actions(self, state):
        _, _, recomb_actions = self.enumerate_action_options(state)
        return recomb_actions

    def is_coalescence_action_valid(self, state, action):
        return self.is_valid_coalescent_action(state, action)

    def is_valid_recombination_action(self, state, action):
        if action.get("event_type") != "recomb":
            return False
        i = action.get("active_lineage_i")
        breakpoint = action.get("breakpoint")
        if not self._is_active_index(state, i):
            return False
        if not isinstance(breakpoint, numbers.Integral) or breakpoint < 1 or breakpoint >= self.num_blocks:
            return False
        span = self._material_span(state.active_lineages[i].material_mask)
        if span is None:
            return False
        first_block, last_block, _ = span
        return bool(first_block < breakpoint <= last_block)


    def apply_coalescence(self, state, action, log_prior=None):
        if not self.is_valid_coalescent_action(state, action):
            raise ValueError(f"Invalid coalescence action: {action}")

        if log_prior is None:
            log_prior = self.compute_cwr_event_log_prior(state, action)
        next_state = state.clone()
        i = action["active_lineage_i"]
        j = action["active_lineage_j"]
        child_i = next_state.active_lineages[i]
        child_j = next_state.active_lineages[j]

        parent_id = next_state.max_node_idx + 1
        parent_mask = child_i.material_mask | child_j.material_mask
        parent = ARGLineage(
            node_id=parent_id,
            children=[child_i.node_id, child_j.node_id],
            parents=[],
            material_mask=parent_mask,
            partials=None,
            sequences_indices=sorted(set(child_i.sequences_indices + child_j.sequences_indices)),
            event_type="coal",
        )

        child_i.parents.append(parent.node_id)
        child_j.parents.append(parent.node_id)
        next_state.all_nodes[parent.node_id] = parent
        next_state.active_lineages = [
            lineage for idx, lineage in enumerate(next_state.active_lineages) if idx not in (i, j)
        ]
        next_state.active_lineages.append(parent)
        next_state.max_node_idx = parent.node_id
        next_state.accumulated_log_prior += log_prior
        next_state.is_done = self.is_terminal(next_state)
        if next_state.is_done:
            log_likelihood = self.evolution_model.compute_arg_log_likelihood(next_state)
            next_state.log_reward = self.reward_fn(log_likelihood)+next_state.accumulated_log_prior
        else:
            next_state.log_reward = None
        return next_state

    def apply_recombination(self, state, action, log_prior=None):
        if not self.is_valid_recombination_action(state, action):
            raise ValueError(f"Invalid recombination action: {action}")

        if log_prior is None:
            log_prior = self.compute_cwr_event_log_prior(state, action)
        next_state = state.clone()
        next_state.action_options = None
        next_state.rates = None
        i = action["active_lineage_i"]
        breakpoint = action["breakpoint"]
        child = next_state.active_lineages[i]
        original_mask = child.material_mask.copy()
        left_mask, right_mask = self._split_mask(original_mask, breakpoint)

        if not np.array_equal(left_mask | right_mask, original_mask):
            raise ValueError("Recombination masks must cover the original material")
        if np.any(left_mask & right_mask):
            raise ValueError("Recombination masks must be disjoint")

        left_parent_id = next_state.max_node_idx + 1
        right_parent_id = next_state.max_node_idx + 2
        left_parent = ARGLineage(
            node_id=left_parent_id,
            children=[child.node_id],
            parents=[],
            material_mask=left_mask,
            partials=None,
            sequences_indices=list(child.sequences_indices),
            event_type="recomb",
            breakpoint=breakpoint,
            recombination_side="left",
        )
        right_parent = ARGLineage(
            node_id=right_parent_id,
            children=[child.node_id],
            parents=[],
            material_mask=right_mask,
            partials=None,
            sequences_indices=list(child.sequences_indices),
            event_type="recomb",
            breakpoint=breakpoint,
            recombination_side="right",
        )

        child.parents = [left_parent.node_id, right_parent.node_id]
        next_state.all_nodes[left_parent.node_id] = left_parent
        next_state.all_nodes[right_parent.node_id] = right_parent
        next_state.active_lineages = [
            lineage for idx, lineage in enumerate(next_state.active_lineages) if idx != i
        ]
        next_state.active_lineages.extend([left_parent, right_parent])
        next_state.max_node_idx = right_parent.node_id
        next_state.accumulated_log_prior += log_prior
        next_state.is_done = self.is_terminal(next_state)
        if next_state.is_done:
            log_likelihood = self.evolution_model.compute_arg_log_likelihood(next_state)
            next_state.log_reward = self.reward_fn(log_likelihood)+next_state.accumulated_log_prior
        else:
            next_state.log_reward = None
        return next_state

    def apply_action(self, state, action, log_prior=None):
        if action.get("event_type") == "coal":
            return self.apply_coalescence(state, action, log_prior)
        if action.get("event_type") == "recomb":
            return self.apply_recombination(state, action, log_prior)
        raise ValueError(f"Unknown action event_type: {action}")

    def compute_event_rates(self, state, coal_actions=None, recomb_weights=None):
        if coal_actions is None or recomb_weights is None:
            if state.action_options is None:
                self.enumerate_action_options(state)
            coal_actions, recomb_weights, _ = state.action_options

        lambda_coal = float(len(coal_actions))
        if recomb_weights:
            total_blocks = sum(weight for _, weight, _ in recomb_weights)
            total_active_material_length = float(total_blocks) / float(self.num_blocks)
        else:
            total_active_material_length = self.total_active_material_length(state)
        lambda_recomb = 0.0
        if recomb_weights:
            lambda_recomb = self.rho / 2.0 * total_active_material_length
        return {
            "lambda_coal": lambda_coal,
            "lambda_recomb": lambda_recomb,
            "total_active_material_length": total_active_material_length,
        }

    def compute_event_probabilities(self, state):
        rates = self.compute_event_rates(state) if state.rates is None else state.rates
        denom = rates["lambda_coal"] + rates["lambda_recomb"]
        
        if denom <= 0:
            return {"coal": 0.0, "recomb": 0.0}
        return {
            "coal": rates["lambda_coal"] / denom,
            "recomb": rates["lambda_recomb"] / denom,
        }

    def sample(self, num_trajs):
        trajectories = []
        for _ in range(num_trajs):
            state = self.get_initial_state()
            while not state.is_done:
                action, log_prior = self.sample_action_from_prior(state)
                state = self.apply_action(state, action, log_prior)
            trajectories.append(state)
        return trajectories

    def sample_action_from_prior(self, state):
        coal_actions, recomb_weights, _ = self.enumerate_action_options(state) if state.action_options is None else state.action_options
        # rates = state.rates if state.rates is not None else self.compute_event_rates(
        #     state,
        #     coal_actions,
        #     recomb_weights,
        # )
        # denom = rates["lambda_coal"] + rates["lambda_recomb"]
        # if denom <= 0:
        #     return None

        # coal_prob = rates["lambda_coal"] / denom if coal_actions else 0.0
        # recomb_prob = rates["lambda_recomb"] / denom if recomb_weights else 0.0

        event_probs = self.compute_event_probabilities(state)

        coal_prob = event_probs["coal"]
        recomb_prob = event_probs["recomb"]

        event_types = ["coal", "recomb"]
        event_types_chosen = np.random.choice(event_types, p=[coal_prob, recomb_prob])
        if event_types_chosen == "coal":
            action = dict(coal_actions[self.rng.randrange(len(coal_actions))])
            return action, math.log(coal_prob) - math.log(len(coal_actions))
        else:
            sampled = self._sample_recombination_prior_action(recomb_weights)
            if sampled is None:
                return None
            action, lineage_weight, valid_breakpoints = sampled
            total_weight = sum(weight for _, weight, _ in recomb_weights)
            log_prior = (
                math.log(recomb_prob)
                + math.log(lineage_weight / total_weight)
                - math.log(len(valid_breakpoints))
            )
            return action, log_prior
        # if coal_prob > 0 and (recomb_prob <= 0 or self.rng.random() < coal_prob):
        #     action = dict(coal_actions[self.rng.randrange(len(coal_actions))])
        #     return action, math.log(coal_prob) - math.log(len(coal_actions))

        # if recomb_prob <= 0:
        #     return None
        # sampled = self._sample_recombination_prior_action(recomb_weights)
        # if sampled is None:
        #     return None
        # action, lineage_weight, valid_breakpoints = sampled
        # total_weight = sum(weight for _, weight, _ in recomb_weights)
        # log_prior = (
        #     math.log(recomb_prob)
        #     + math.log(lineage_weight / total_weight)
        #     - math.log(len(valid_breakpoints))
        # )
        # return action, log_prior


    def compute_action_prior_distribution(self, state, event_type=None):
        coal_actions, recomb_weights, recomb_actions = self.enumerate_action_options(state) if state.action_options is None else state.action_options
        rates = state.rates if state.rates is not None else self.compute_event_rates(state, coal_actions, recomb_weights)
        denom = rates["lambda_coal"] + rates["lambda_recomb"]
        if denom <= 0:
            return []
        distribution = []
        if event_type == "coal":
            if coal_actions and rates["lambda_coal"] > 0:
                log_prob = math.log(rates["lambda_coal"] / denom) - math.log(len(coal_actions))
                distribution.extend((dict(action), log_prob) for action in coal_actions)
        else:
            if recomb_actions and rates["lambda_recomb"] > 0:
                total_weight = sum(weight for _, weight, _ in recomb_weights)
                if total_weight > 0:
                    weight_by_lineage = {
                        lineage_i: (weight, valid_breakpoints)
                        for lineage_i, weight, valid_breakpoints in recomb_weights
                    }
                    log_recomb_event = math.log(rates["lambda_recomb"] / denom)
                    for action in recomb_actions:
                        weight, valid_breakpoints = weight_by_lineage[action["active_lineage_i"]]
                        log_prob = (
                            log_recomb_event
                            + math.log(weight / total_weight)
                            - math.log(len(valid_breakpoints))
                        )
                        distribution.append((dict(action), log_prob))

        return distribution

    def compute_cwr_event_log_prior(self, state, action, rates=None, recomb_weights=None):
        if state.action_options is None:
            self.enumerate_action_options(state)
        coal_actions = state.action_options[0]
        if recomb_weights is None:
            recomb_weights = state.action_options[1]
        if rates is None:
            rates = state.rates if state.rates is not None else self.compute_event_rates(state)

        denom = rates["lambda_coal"] + rates["lambda_recomb"]
        if denom <= 0:
            return -math.inf
        if action.get("event_type") == "coal":
            if not self.is_valid_coalescent_action(state, action) or not coal_actions:
                return -math.inf
            action_prob = (rates["lambda_coal"] / denom) / len(coal_actions)
            return math.log(action_prob) if action_prob > 0 else -math.inf
        if action.get("event_type") == "recomb":
            i = action.get("active_lineage_i")
            breakpoint = action.get("breakpoint")
            if not self._is_active_index(state, i):
                return -math.inf
            if not isinstance(breakpoint, numbers.Integral) or breakpoint < 1 or breakpoint >= self.num_blocks:
                return -math.inf
            total_weight = sum(weight for _, weight, _ in recomb_weights)
            if total_weight <= 0:
                return -math.inf
            for lineage_i, weight, valid_breakpoints in recomb_weights:
                if lineage_i != i:
                    continue
                if (
                    not valid_breakpoints
                    or breakpoint < valid_breakpoints[0]
                    or breakpoint > valid_breakpoints[-1]
                ):
                    return -math.inf
                action_prob = (
                    (rates["lambda_recomb"] / denom)
                    * (weight / total_weight)
                    / len(valid_breakpoints)
                )
                return math.log(action_prob) if action_prob > 0 else -math.inf
            return -math.inf
        return -math.inf

    def compute_smcprime_event_log_prior(self, state, action, rates=None, recomb_weights=None):
        return self.compute_cwr_event_log_prior(state, action, rates, recomb_weights)

    def prepare_rollout_inputs(self, tree_features, input_actions=None, random_spec=None, batch_nb_seq=None):
        if len(tree_features.shape) != 4:
            raise ValueError("tree_features must have shape (batch, active_lineages, sequence_length, channels)")

        inputs = tree_features.float()
        batch_size, active_lineages, _, _ = inputs.shape
        batch_input = inputs.reshape(batch_size, active_lineages, -1)
        if batch_nb_seq is None:
            batch_nb_seq = torch.full(
                (batch_size,),
                active_lineages,
                dtype=torch.long,
                device=inputs.device,
            )
        else:
            batch_nb_seq = torch.as_tensor(batch_nb_seq, dtype=torch.long, device=inputs.device)
            if batch_nb_seq.shape != (batch_size,):
                raise ValueError("batch_nb_seq must have shape (batch,)")
            if torch.any(batch_nb_seq < 0) or torch.any(batch_nb_seq > active_lineages):
                raise ValueError("batch_nb_seq entries must be between 0 and active_lineages")

        input_dict = {
            "batch_input": batch_input,
            "batch_seq_features": inputs,
            "batch_nb_seq": batch_nb_seq,
            "batch_size": batch_size,
            "batch_traj_idx": torch.arange(batch_size, device=inputs.device),
            "random_spec": random_spec,
        }

        if input_actions is not None:
            event_type_map = {"coal": 0, "recomb": 1}
            input_dict["input_actions"] = input_actions
            input_dict["input_event_types"] = torch.tensor(
                [event_type_map.get(action.get("event_type"), -1) for action in input_actions],
                dtype=torch.long,
                device=inputs.device,
            )
            input_dict["input_active_lineage_i"] = torch.tensor(
                [action.get("active_lineage_i", -1) for action in input_actions],
                dtype=torch.long,
                device=inputs.device,
            )
            input_dict["input_active_lineage_j"] = torch.tensor(
                [action.get("active_lineage_j", -1) for action in input_actions],
                dtype=torch.long,
                device=inputs.device,
            )
            input_dict["input_breakpoints"] = torch.tensor(
                [action.get("breakpoint", -1) for action in input_actions],
                dtype=torch.long,
                device=inputs.device,
            )

        return input_dict

    def total_active_material_length(self, state):
        total_blocks = sum(int(lineage.material_mask.sum()) for lineage in state.active_lineages)
        return float(total_blocks) / float(self.num_blocks)

    def _sample_recombination_prior_action(self, recomb_weights):
        total_weight = sum(weight for _, weight, _ in recomb_weights)
        if total_weight <= 0:
            return None

        target = self.rng.random() * total_weight
        cumulative = 0.0
        selected = recomb_weights[-1]
        for item in recomb_weights:
            cumulative += item[1]
            if target <= cumulative:
                selected = item
                break

        lineage_i, lineage_weight, valid_breakpoints = selected
        if not valid_breakpoints:
            return None
        breakpoint = valid_breakpoints[self.rng.randrange(len(valid_breakpoints))]
        action = {
            "event_type": "recomb",
            "active_lineage_i": lineage_i,
            "breakpoint": breakpoint,
        }
        return action, lineage_weight, valid_breakpoints

    def _material_span(self, material_mask):
        material_blocks = np.flatnonzero(np.asarray(material_mask, dtype=bool))
        if material_blocks.size < 2:
            return None
        return int(material_blocks[0]), int(material_blocks[-1]), int(material_blocks.size)

    def _split_mask(self, material_mask, breakpoint):
        mask = np.asarray(material_mask, dtype=bool)
        left_mask = mask & (self.block_indices < breakpoint)
        right_mask = mask & (self.block_indices >= breakpoint)
        return left_mask, right_mask

    def _is_active_index(self, state, idx):
        return isinstance(idx, numbers.Integral) and 0 <= idx < len(state.active_lineages)
