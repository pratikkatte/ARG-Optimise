import copy
import math
import numbers
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch

import numpy as np

from time_env import (
    DEFAULT_TIME_BINS,
    DEFAULT_TIME_MODEL,
    DEFAULT_TIME_TAIL_PROBABILITY,
    TimeEnvCategorical,
)

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


@dataclass(frozen=True)
class MaterialSegments:
    """Canonical half-open material intervals in ARG block coordinates."""

    segments: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    count: int = field(init=False)
    span_start: Optional[int] = field(init=False)
    span_end: Optional[int] = field(init=False)

    def __post_init__(self):
        canonical = self._canonicalize(self.segments)
        object.__setattr__(self, "segments", canonical)
        object.__setattr__(self, "count", sum(end - start for start, end in canonical))
        if canonical:
            object.__setattr__(self, "span_start", canonical[0][0])
            object.__setattr__(self, "span_end", canonical[-1][1] - 1)
        else:
            object.__setattr__(self, "span_start", None)
            object.__setattr__(self, "span_end", None)

    @classmethod
    def full(cls, num_blocks):
        num_blocks = int(num_blocks)
        return cls(((0, num_blocks),)) if num_blocks > 0 else cls(())

    @classmethod
    def from_mask(cls, material_mask):
        mask = np.asarray(material_mask, dtype=bool)
        if mask.size == 0:
            return cls(())
        padded = np.concatenate(([False], mask, [False]))
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        return cls(tuple((int(start), int(end)) for start, end in zip(changes[::2], changes[1::2])))

    @classmethod
    def from_segments(cls, segments):
        if isinstance(segments, MaterialSegments):
            return segments
        return cls(tuple(segments or ()))

    @staticmethod
    def _canonicalize(segments):
        cleaned = []
        for start, end in sorted((int(start), int(end)) for start, end in segments):
            if end <= start:
                continue
            if cleaned and start <= cleaned[-1][1]:
                prev_start, prev_end = cleaned[-1]
                cleaned[-1] = (prev_start, max(prev_end, end))
            else:
                cleaned.append((start, end))
        return tuple(cleaned)

    def to_mask(self, num_blocks):
        mask = np.zeros(int(num_blocks), dtype=bool)
        for start, end in self.segments:
            mask[start:end] = True
        return mask

    def to_block_list(self):
        blocks = []
        for start, end in self.segments:
            blocks.extend(range(start, end))
        return blocks

    def valid_breakpoint_count(self):
        if self.count < 2 or self.span_start is None or self.span_end is None:
            return 0
        return int(self.span_end - self.span_start)

    def split(self, breakpoint):
        breakpoint = int(breakpoint)
        left = []
        right = []
        for start, end in self.segments:
            if start < breakpoint:
                left.append((start, min(end, breakpoint)))
            if end > breakpoint:
                right.append((max(start, breakpoint), end))
        return MaterialSegments(left), MaterialSegments(right)

    def union(self, other):
        other = MaterialSegments.from_segments(other)
        return MaterialSegments(self.segments + other.segments)

    def intersection(self, other):
        other = MaterialSegments.from_segments(other)
        intersections = []
        i = j = 0
        while i < len(self.segments) and j < len(other.segments):
            left_start, left_end = self.segments[i]
            right_start, right_end = other.segments[j]
            start = max(left_start, right_start)
            end = min(left_end, right_end)
            if start < end:
                intersections.append((start, end))
            if left_end <= right_end:
                i += 1
            else:
                j += 1
        return MaterialSegments(intersections)

    def intersection_count(self, other, interval_start=None, interval_end=None):
        other = MaterialSegments.from_segments(other)
        total = 0
        i = j = 0
        while i < len(self.segments) and j < len(other.segments):
            left_start, left_end = self.segments[i]
            right_start, right_end = other.segments[j]
            start = max(left_start, right_start)
            end = min(left_end, right_end)
            if interval_start is not None:
                start = max(start, int(interval_start))
            if interval_end is not None:
                end = min(end, int(interval_end))
            if start < end:
                total += end - start
            if left_end <= right_end:
                i += 1
            else:
                j += 1
        return int(total)

    def overlaps(self, other):
        return self.intersection_count(other) > 0

    def covers_interval(self, start, end):
        start = int(start)
        end = int(end)
        if start >= end:
            return False
        return any(seg_start <= start and end <= seg_end for seg_start, seg_end in self.segments)


@dataclass(frozen=True)
class RecombinationChoice:
    active_lineage_i: int
    material_count: int
    span_start: int
    span_end: int

    @property
    def breakpoint_count(self):
        return int(self.span_end - self.span_start)

    def as_weight_tuple(self):
        return (
            self.active_lineage_i,
            self.material_count,
            list(range(self.span_start + 1, self.span_end + 1)),
        )


@dataclass(frozen=True)
class PriorActionOptions:
    coal_actions: Tuple[Dict[str, Any], ...]
    recomb_choices: Tuple[RecombinationChoice, ...]
    rates: Dict[str, float]

    @property
    def total_recomb_weight(self):
        return sum(choice.material_count for choice in self.recomb_choices)


class ARGLineage:
    def __init__(
        self,
        node_id: int,
        children: Optional[Sequence[int]] = None,
        parents: Optional[Sequence[int]] = None,
        material_mask: Optional[np.ndarray] = None,
        material_segments: Optional[MaterialSegments] = None,
        num_blocks: Optional[int] = None,
        partials: Optional[Any] = None,
        sequences_indices: Optional[Sequence[int]] = None,
        event_type: Optional[str] = None,
        breakpoint: Optional[int] = None,
        recombination_side: Optional[str] = None,
        time: float = 0.0,
    ):
        self.node_id = int(node_id)
        self.children = list(children or [])
        self.parents = list(parents or [])
        self.partials = partials
        self.sequences_indices = list(sequences_indices or [])
        self.event_type = event_type
        self.breakpoint = breakpoint
        self.recombination_side = recombination_side
        self.time = float(time)
        self._material_mask = None

        if material_segments is None:
            mask = np.asarray([] if material_mask is None else material_mask, dtype=bool)
            self._material_mask = mask.copy()
            self.material_segments = MaterialSegments.from_mask(self._material_mask)
            self.num_blocks = int(mask.size if num_blocks is None else num_blocks)
        else:
            self.material_segments = MaterialSegments.from_segments(material_segments)
            self.num_blocks = int(
                num_blocks
                if num_blocks is not None
                else max((end for _, end in self.material_segments.segments), default=0)
            )
            if material_mask is not None:
                self._material_mask = np.asarray(material_mask, dtype=bool).copy()
                self.num_blocks = int(self._material_mask.size)

    @property
    def material_mask(self):
        if self._material_mask is None:
            self._material_mask = self.material_segments.to_mask(self.num_blocks)
        return self._material_mask

    @material_mask.setter
    def material_mask(self, value):
        if value is None:
            self._material_mask = None
            return
        self._material_mask = np.asarray(value, dtype=bool).copy()
        self.num_blocks = int(self._material_mask.size)
        self.material_segments = MaterialSegments.from_mask(self._material_mask)

    @property
    def material_count(self):
        return self.material_segments.count

    @property
    def material_span(self):
        if self.material_segments.count < 2:
            return None
        return (
            self.material_segments.span_start,
            self.material_segments.span_end,
            self.material_segments.count,
        )

    def clone(self, copy_partials=True, copy_mask=True):
        clone = ARGLineage(
            node_id=self.node_id,
            children=list(self.children),
            parents=list(self.parents),
            material_segments=self.material_segments,
            num_blocks=self.num_blocks,
            partials=copy.deepcopy(self.partials) if copy_partials else self.partials,
            sequences_indices=list(self.sequences_indices),
            event_type=self.event_type,
            breakpoint=self.breakpoint,
            recombination_side=self.recombination_side,
            time=float(self.time),
        )
        if copy_mask and self._material_mask is not None:
            clone._material_mask = self._material_mask.copy()
        return clone

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
    prior_options: Optional[PriorActionOptions] = None
    total_active_blocks: Optional[int] = None
    current_time: float = 0.0

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
            total_active_blocks=self.total_active_blocks,
            current_time=float(self.current_time),
        )

class EvolutionModelTorch(torch.nn.Module):
    """JC69 likelihood model for constructed ARG states."""

    _PROB_FLOOR = 1e-300
    _NON_FINITE_LOG_LIKELIHOOD = -1e6

    def __init__(self, env):
        super().__init__()
        self.env = env

    def compute_arg_log_likelihood(self, state):
        """Compute the JC69 sequence log likelihood of a terminal ARG.

        Each marginal segment induced by recombination breakpoints is scored
        with Felsenstein pruning. Fixed-time mode uses ``fixed_edge_length`` as
        a JC69 branch length. Learnable-time mode stores node times as
        t/(2Ne), which are converted to substitutions/site before scoring.
        """

        self._require_terminal(state)

        if self.env.sequences is None:
            return 0.0

        if self.env.fixed_edge_length < 0:
            raise ValueError("fixed_edge_length must be non-negative for likelihood scoring")

        seq_arrays = self._seq_arrays_numpy()
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
            if lineage.material_segments.covers_interval(block_start, block_end)
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
                memo,
            )
            edge_time = self.env.edge_length_between(state, node_id, child_id)
            branch_length = self.env.branch_length_for_likelihood(edge_time)
            transition_matrix = self._jc69_transition_matrix(branch_length)
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
        return parent.material_segments.intersection_count(
            child.material_segments,
            interval_start=block_start,
            interval_end=block_end,
        ) > 0


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
        learn_times: bool = False,
        time_increments: Optional[Sequence[float]] = None,
        time_model: str = DEFAULT_TIME_MODEL,
        time_bins: int = DEFAULT_TIME_BINS,
        time_tail_probability: float = DEFAULT_TIME_TAIL_PROBABILITY,
        effective_population_size: float = 10000.0,
        mutation_rate: float = 2e-8,
        use_time_prior: bool = True,
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
        self.learn_times = bool(learn_times)
        self.use_time_prior = bool(use_time_prior)
        self.effective_population_size = float(effective_population_size)
        self.mutation_rate = float(mutation_rate)
        if self.effective_population_size <= 0:
            raise ValueError("effective_population_size must be positive")
        if self.mutation_rate < 0:
            raise ValueError("mutation_rate must be non-negative")
        self.time_model = time_model
        self.time_bins = int(time_bins)
        self.time_tail_probability = float(time_tail_probability)
        self.time_env = (
            TimeEnvCategorical(
                time_increments,
                bins=self.time_bins,
                tail_probability=self.time_tail_probability,
                time_model=self.time_model,
            )
            if self.learn_times
            else None
        )
        if self.time_env is not None:
            self.time_model = self.time_env.time_model
            self.time_bins = self.time_env.bins
            if self.time_env.tail_probability is not None:
                self.time_tail_probability = self.time_env.tail_probability
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

    def edge_length_between(self, state, parent_id, child_id):
        if not self.learn_times:
            return self.fixed_edge_length

        parent_time = float(state.all_nodes[parent_id].time)
        child_time = float(state.all_nodes[child_id].time)
        edge_length = parent_time - child_time
        if edge_length <= 0:
            raise ValueError(
                f"ARG node times must increase from child to parent: "
                f"parent={parent_id} time={parent_time}, child={child_id} time={child_time}"
            )
        return edge_length

    def branch_length_for_likelihood(self, edge_time):
        if not self.learn_times:
            return float(edge_time)
        return (
            float(edge_time)
            * 2.0
            * self.effective_population_size
            * self.mutation_rate
        )

    def _require_time_action(self, action):
        if not self.learn_times:
            return None
        if "time_action" not in action:
            raise ValueError("learnable ARG times require every action to include time_action")
        return int(action["time_action"])

    def _total_event_rate(self, rates):
        total_rate = float(rates["lambda_coal"] + rates["lambda_recomb"])
        if total_rate <= 0:
            raise ValueError("waiting-time rate must be positive")
        return total_rate

    def _delta_t_for_action(self, action, rates=None):
        time_action = self._require_time_action(action)
        if time_action is None:
            return self.fixed_edge_length
        if rates is None:
            raise ValueError("rates are required to map learnable time actions")
        return self.time_env.time_action_to_delta(
            time_action,
            self._total_event_rate(rates),
        )

    def _time_action_for_delta(self, delta_t, rates=None):
        if not self.learn_times:
            return None
        if rates is None:
            raise ValueError("rates are required to recover learnable time actions")
        return self.time_env.delta_to_time_action(
            delta_t,
            self._total_event_rate(rates),
        )

    def _with_random_time_action(self, action, rates=None):
        action = dict(action)
        if self.learn_times and "time_action" not in action:
            action["time_action"] = self._sample_time_action_from_prior(rates)

        return action

    def _sample_time_action_from_prior(self, rates=None):
        if not self.learn_times:
            return None
        if not self.use_time_prior:
            return self.time_env.generate_random_action()
        if rates is None:
            raise ValueError("rates are required to sample learnable ARG times from the prior")
        return self.time_env.sample_action_from_prior(self._total_event_rate(rates), self.rng)

    def _time_action_log_prior_distribution(self, rates):
        if not self.learn_times:
            return [(None, 0.0)]
        if not self.use_time_prior:
            log_prob = -math.log(self.time_env.bins)
            return [(time_action, log_prob) for time_action in range(self.time_env.bins)]

        total_rate = float(rates["lambda_coal"] + rates["lambda_recomb"])
        if total_rate <= 0:
            return []
        return [
            (
                time_action,
                self.time_env.time_action_log_probability(time_action, total_rate),
            )
            for time_action in range(self.time_env.bins)
        ]

    def compute_waiting_time_log_prior(self, state, action, rates=None):
        if not self.learn_times or not self.use_time_prior:
            return 0.0
        if rates is None:
            rates = state.rates if state.rates is not None else self.compute_event_rates(state)
        total_rate = float(rates["lambda_coal"] + rates["lambda_recomb"])
        if total_rate <= 0:
            return -math.inf
        time_action = self._require_time_action(action)
        return self.time_env.time_action_log_probability(time_action, total_rate)

    def get_initial_state(self):
        active_lineages = []
        all_nodes = {}
        for node_id in range(self.num_sequences):
            lineage = ARGLineage(
                node_id=node_id,
                children=[],
                parents=[],
                material_segments=MaterialSegments.full(self.num_blocks),
                num_blocks=self.num_blocks,
                partials=None,
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
        if state.is_done:
            log_likelihood = self.evolution_model.compute_arg_log_likelihood(state)
            state.log_reward = self.reward_fn(log_likelihood)+state.accumulated_log_prior
        return state

    def get_active_counts(self, state):
        if not state.active_lineages:
            return np.zeros(self.num_blocks, dtype=int)
        counts = np.zeros(self.num_blocks, dtype=int)
        for lineage in state.active_lineages:
            for start, end in lineage.material_segments.segments:
                counts[start:end] += 1
        return counts

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
            event["segments_by_side"][side] = self.mask_to_segments(lineage.material_segments)
            event["blocks_by_side"][side] = lineage.material_segments.to_block_list()
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
        if isinstance(material_mask, MaterialSegments):
            return list(material_mask.segments)
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

        The exported topology contains ancestry edges only. In fixed-edge mode,
        synthetic node times are derived from graph depth. In learnable-time
        mode, stored ARG node times are internal t/(2Ne) values and are exported
        in generations to match msprime tree sequences.
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
        if self.learn_times:
            tables.time_units = "generations"
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
                material_segments = parent.material_segments.intersection(child.material_segments)
                for left_block, right_block in material_segments.segments:
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
        if self.learn_times:
            time_scale = 2.0 * self.effective_population_size
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
        if state.total_active_blocks is not None:
            result = int(state.total_active_blocks) == self.num_blocks
            if self.num_blocks <= 10000:
                dense_result = bool(np.all(self.get_active_counts(state) == 1))
                if result != dense_result:
                    raise AssertionError(
                        "cached total_active_blocks disagrees with dense terminal check"
                    )
            return result
        return bool(np.all(self.get_active_counts(state) == 1))

    def _valid_breakpoints_for_lineage(self, state, active_lineage_i):
        span = state.active_lineages[active_lineage_i].material_span
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
        return state.active_lineages[i].material_segments.overlaps(
            state.active_lineages[j].material_segments
        )

    def enumerate_prior_options(self, state):
        if state.prior_options is not None:
            return state.prior_options

        coal_actions = []
        recomb_choices = []
        if not self.is_terminal(state):
            for i, lineage in enumerate(state.active_lineages):
                span = lineage.material_span
                if span is not None:
                    first_block, last_block, material_count = span
                    if first_block < last_block:
                        recomb_choices.append(
                            RecombinationChoice(
                                active_lineage_i=i,
                                material_count=int(material_count),
                                span_start=int(first_block),
                                span_end=int(last_block),
                            )
                        )
            coal_actions = self._enumerate_coalescent_prior_actions(state)

        total_blocks = sum(choice.material_count for choice in recomb_choices)
        if recomb_choices:
            total_active_material_length = float(total_blocks) / float(self.num_blocks)
        else:
            total_active_material_length = self.total_active_material_length(state)
        lambda_recomb = (
            self.rho / 2.0 * total_active_material_length
            if recomb_choices
            else 0.0
        )
        rates = {
            "lambda_coal": float(len(coal_actions)),
            "lambda_recomb": float(lambda_recomb),
            "total_active_material_length": total_active_material_length,
        }
        state.prior_options = PriorActionOptions(
            coal_actions=tuple(coal_actions),
            recomb_choices=tuple(recomb_choices),
            rates=rates,
        )
        state.rates = rates
        return state.prior_options

    def _enumerate_coalescent_prior_actions(self, state):
        events = []
        for active_idx, lineage in enumerate(state.active_lineages):
            for start, end in lineage.material_segments.segments:
                events.append((start, 1, active_idx))
                events.append((end, -1, active_idx))
        events.sort(key=lambda item: (item[0], item[1]))

        active = set()
        pairs = set()
        for _position, event_type, active_idx in events:
            if event_type < 0:
                active.discard(active_idx)
                continue
            for other_idx in active:
                if other_idx < active_idx:
                    pairs.add((other_idx, active_idx))
                else:
                    pairs.add((active_idx, other_idx))
            active.add(active_idx)

        return [
            {
                "event_type": "coal",
                "active_lineage_i": i,
                "active_lineage_j": j,
            }
            for i, j in sorted(pairs)
        ]

    def enumerate_action_options(self, state):
        if state.action_options is not None:
            return state.action_options[0], state.action_options[1], state.action_options[2]
        prior_options = self.enumerate_prior_options(state)
        coal_actions = [dict(action) for action in prior_options.coal_actions]
        recomb_weights = [
            choice.as_weight_tuple()
            for choice in prior_options.recomb_choices
        ]
        recomb_actions = []
        for choice in prior_options.recomb_choices:
            for breakpoint in range(choice.span_start + 1, choice.span_end + 1):
                recomb_actions.append(
                    {
                        "event_type": "recomb",
                        "active_lineage_i": choice.active_lineage_i,
                        "breakpoint": breakpoint,
                    }
                )

        state.action_options = (coal_actions, recomb_weights, recomb_actions)
        state.rates = prior_options.rates
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
        span = state.active_lineages[i].material_span
        if span is None:
            return False
        first_block, last_block, _ = span
        return bool(first_block < breakpoint <= last_block)

    def _copy_state_for_transition(self, state):
        return ARGState(
            active_lineages=list(state.active_lineages),
            all_nodes=dict(state.all_nodes),
            max_node_idx=state.max_node_idx,
            log_reward=None,
            accumulated_log_prior=state.accumulated_log_prior,
            is_done=False,
            total_active_blocks=state.total_active_blocks,
            current_time=float(state.current_time),
        )

    def _finalize_transition_state(self, next_state, log_prior, compute_reward=True):
        next_state.accumulated_log_prior += log_prior
        next_state.is_done = self.is_terminal(next_state)
        if next_state.is_done and compute_reward:
            log_likelihood = self.evolution_model.compute_arg_log_likelihood(next_state)
            next_state.log_reward = (
                self.reward_fn(log_likelihood) + next_state.accumulated_log_prior
            )
        else:
            next_state.log_reward = None
        return next_state

    def apply_coalescence(self, state, action, log_prior=None, compute_reward=True):
        if not self.is_valid_coalescent_action(state, action):
            raise ValueError(f"Invalid coalescence action: {action}")

        rates = self.enumerate_prior_options(state).rates
        if log_prior is None:
            log_prior = self.compute_cwr_event_log_prior(state, action, rates=rates)
        next_state = self._copy_state_for_transition(state)
        i = action["active_lineage_i"]
        j = action["active_lineage_j"]
        child_i = next_state.active_lineages[i].clone(copy_partials=False, copy_mask=False)
        child_j = next_state.active_lineages[j].clone(copy_partials=False, copy_mask=False)

        parent_id = next_state.max_node_idx + 1
        parent_segments = child_i.material_segments.union(child_j.material_segments)
        overlap_count = child_i.material_segments.intersection_count(child_j.material_segments)
        if self.learn_times:
            delta_t = self._delta_t_for_action(action, rates)
            parent_time = float(state.current_time) + delta_t
            next_state.current_time = parent_time
        else:
            parent_time = 0.0
        parent = ARGLineage(
            node_id=parent_id,
            children=[child_i.node_id, child_j.node_id],
            parents=[],
            material_segments=parent_segments,
            num_blocks=self.num_blocks,
            partials=None,
            sequences_indices=sorted(set(child_i.sequences_indices + child_j.sequences_indices)),
            event_type="coal",
            time=parent_time,
        )

        child_i.parents.append(parent.node_id)
        child_j.parents.append(parent.node_id)
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
        return self._finalize_transition_state(next_state, log_prior, compute_reward)

    def apply_recombination(self, state, action, log_prior=None, compute_reward=True):
        if not self.is_valid_recombination_action(state, action):
            raise ValueError(f"Invalid recombination action: {action}")

        rates = self.enumerate_prior_options(state).rates
        if log_prior is None:
            log_prior = self.compute_cwr_event_log_prior(state, action, rates=rates)
        next_state = self._copy_state_for_transition(state)
        i = action["active_lineage_i"]
        breakpoint = action["breakpoint"]
        child = next_state.active_lineages[i].clone(copy_partials=False, copy_mask=False)
        left_segments, right_segments = child.material_segments.split(breakpoint)

        left_parent_id = next_state.max_node_idx + 1
        right_parent_id = next_state.max_node_idx + 2
        if self.learn_times:
            delta_t = self._delta_t_for_action(action, rates)
            event_time = float(state.current_time) + delta_t
            next_state.current_time = event_time
        else:
            event_time = 0.0
        left_parent = ARGLineage(
            node_id=left_parent_id,
            children=[child.node_id],
            parents=[],
            material_segments=left_segments,
            num_blocks=self.num_blocks,
            partials=None,
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
            partials=None,
            sequences_indices=list(child.sequences_indices),
            event_type="recomb",
            breakpoint=breakpoint,
            recombination_side="right",
            time=event_time,
        )

        child.parents = [left_parent.node_id, right_parent.node_id]
        next_state.all_nodes[child.node_id] = child
        next_state.all_nodes[left_parent.node_id] = left_parent
        next_state.all_nodes[right_parent.node_id] = right_parent
        next_state.active_lineages = [
            lineage for idx, lineage in enumerate(next_state.active_lineages) if idx != i
        ]
        next_state.active_lineages.extend([left_parent, right_parent])
        next_state.max_node_idx = right_parent.node_id
        return self._finalize_transition_state(next_state, log_prior, compute_reward)

    def apply_action(self, state, action, log_prior=None, compute_reward=True):
        if action.get("event_type") == "coal":
            return self.apply_coalescence(state, action, log_prior, compute_reward)
        if action.get("event_type") == "recomb":
            return self.apply_recombination(state, action, log_prior, compute_reward)
        raise ValueError(f"Unknown action event_type: {action}")

    def compute_event_rates(self, state, coal_actions=None, recomb_weights=None):
        if coal_actions is None and recomb_weights is None:
            return dict(self.enumerate_prior_options(state).rates)
        if coal_actions is None or recomb_weights is None:
            if state.action_options is None:
                self.enumerate_action_options(state)
            coal_actions, recomb_weights, _ = state.action_options

        lambda_coal = float(len(coal_actions))
        if recomb_weights:
            total_blocks = sum(self._recomb_weight(item) for item in recomb_weights)
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
        rates = self.enumerate_prior_options(state).rates
        denom = rates["lambda_coal"] + rates["lambda_recomb"]
        
        if denom <= 0:
            return {"coal": 0.0, "recomb": 0.0}
        return {
            "coal": rates["lambda_coal"] / denom,
            "recomb": rates["lambda_recomb"] / denom,
        }

    def sample(self, num_trajs, compute_reward=True):
        trajectories = []
        for _ in range(num_trajs):
            state = self.get_initial_state()
            while not state.is_done:
                action, log_prior = self.sample_action_from_prior(state)
                state = self.apply_action(
                    state,
                    action,
                    log_prior,
                    compute_reward=compute_reward,
                )
            trajectories.append(state)
        return trajectories

    def sample_action_from_prior(self, state):
        prior_options = self.enumerate_prior_options(state)
        coal_actions = prior_options.coal_actions
        recomb_choices = prior_options.recomb_choices
        rates = prior_options.rates
        denom = rates["lambda_coal"] + rates["lambda_recomb"]
        if denom <= 0:
            return None

        coal_prob = rates["lambda_coal"] / denom if coal_actions else 0.0
        recomb_prob = rates["lambda_recomb"] / denom if recomb_choices else 0.0

        if coal_prob > 0 and (recomb_prob <= 0 or self.rng.random() < coal_prob):
            action = self._with_random_time_action(
                dict(coal_actions[self.rng.randrange(len(coal_actions))]),
                rates,
            )
            return action, self.compute_cwr_event_log_prior(state, action)

        if recomb_prob <= 0:
            return None
        sampled = self._sample_recombination_prior_action(recomb_choices)
        if sampled is None:
            return None
        action, _lineage_weight, _choice = sampled
        action = self._with_random_time_action(action, rates)
        return action, self.compute_cwr_event_log_prior(state, action)


    def compute_action_prior_distribution(self, state, event_type=None):
        coal_actions, recomb_weights, recomb_actions = self.enumerate_action_options(state) if state.action_options is None else state.action_options
        rates = state.rates if state.rates is not None else self.compute_event_rates(state, coal_actions, recomb_weights)
        denom = rates["lambda_coal"] + rates["lambda_recomb"]
        if denom <= 0:
            return []
        time_log_probs = self._time_action_log_prior_distribution(rates)
        distribution = []
        if event_type in (None, "coal") and coal_actions and rates["lambda_coal"] > 0:
            log_action_prob = math.log(rates["lambda_coal"] / denom) - math.log(len(coal_actions))
            for action in coal_actions:
                for time_action, log_time_prob in time_log_probs:
                    timed_action = dict(action)
                    if time_action is not None:
                        timed_action["time_action"] = time_action
                    distribution.append((timed_action, log_action_prob + log_time_prob))

        if event_type in (None, "recomb") and recomb_actions and rates["lambda_recomb"] > 0:
            total_weight = sum(weight for _, weight, _ in recomb_weights)
            if total_weight > 0:
                weight_by_lineage = {
                    lineage_i: (weight, valid_breakpoints)
                    for lineage_i, weight, valid_breakpoints in recomb_weights
                }
                log_recomb_event = math.log(rates["lambda_recomb"] / denom)
                for action in recomb_actions:
                    weight, valid_breakpoints = weight_by_lineage[action["active_lineage_i"]]
                    log_action_prob = (
                        log_recomb_event
                        + math.log(weight / total_weight)
                        - math.log(len(valid_breakpoints))
                    )
                    for time_action, log_time_prob in time_log_probs:
                        timed_action = dict(action)
                        if time_action is not None:
                            timed_action["time_action"] = time_action
                        distribution.append((timed_action, log_action_prob + log_time_prob))

        return distribution

    def compute_cwr_event_log_prior(self, state, action, rates=None, recomb_weights=None):
        prior_options = self.enumerate_prior_options(state)
        coal_actions = prior_options.coal_actions
        recomb_choices = (
            prior_options.recomb_choices
            if recomb_weights is None
            else tuple(self._choice_from_recomb_weight(item) for item in recomb_weights)
        )
        if rates is None:
            rates = prior_options.rates

        denom = rates["lambda_coal"] + rates["lambda_recomb"]
        if denom <= 0:
            return -math.inf
        wait_log_prior = self.compute_waiting_time_log_prior(state, action, rates)
        if not math.isfinite(wait_log_prior):
            return -math.inf
        if action.get("event_type") == "coal":
            if not self.is_valid_coalescent_action(state, action) or not coal_actions:
                return -math.inf
            action_prob = (rates["lambda_coal"] / denom) / len(coal_actions)
            return math.log(action_prob) + wait_log_prior if action_prob > 0 else -math.inf
        if action.get("event_type") == "recomb":
            i = action.get("active_lineage_i")
            breakpoint = action.get("breakpoint")
            if not self._is_active_index(state, i):
                return -math.inf
            if not isinstance(breakpoint, numbers.Integral) or breakpoint < 1 or breakpoint >= self.num_blocks:
                return -math.inf
            total_weight = sum(choice.material_count for choice in recomb_choices)
            if total_weight <= 0:
                return -math.inf
            for choice in recomb_choices:
                if choice.active_lineage_i != i:
                    continue
                if (
                    choice.breakpoint_count <= 0
                    or breakpoint <= choice.span_start
                    or breakpoint > choice.span_end
                ):
                    return -math.inf
                action_prob = (
                    (rates["lambda_recomb"] / denom)
                    * (choice.material_count / total_weight)
                    / choice.breakpoint_count
                )
                return math.log(action_prob) + wait_log_prior if action_prob > 0 else -math.inf
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
        if state.total_active_blocks is not None:
            total_blocks = int(state.total_active_blocks)
        else:
            total_blocks = sum(lineage.material_count for lineage in state.active_lineages)
        return float(total_blocks) / float(self.num_blocks)

    def _sample_recombination_prior_action(self, recomb_weights):
        total_weight = sum(self._recomb_weight(item) for item in recomb_weights)
        if total_weight <= 0:
            return None

        target = self.rng.random() * total_weight
        cumulative = 0.0
        selected = recomb_weights[-1]
        for item in recomb_weights:
            cumulative += self._recomb_weight(item)
            if target <= cumulative:
                selected = item
                break

        if isinstance(selected, RecombinationChoice):
            if selected.breakpoint_count <= 0:
                return None
            breakpoint = (
                selected.span_start
                + 1
                + self.rng.randrange(selected.breakpoint_count)
            )
            action = {
                "event_type": "recomb",
                "active_lineage_i": selected.active_lineage_i,
                "breakpoint": breakpoint,
            }
            return action, selected.material_count, selected

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

    def _recomb_weight(self, recomb_weight):
        if isinstance(recomb_weight, RecombinationChoice):
            return recomb_weight.material_count
        return recomb_weight[1]

    def _choice_from_recomb_weight(self, recomb_weight):
        if isinstance(recomb_weight, RecombinationChoice):
            return recomb_weight
        lineage_i, weight, valid_breakpoints = recomb_weight
        if not valid_breakpoints:
            return RecombinationChoice(lineage_i, weight, 0, 0)
        return RecombinationChoice(
            active_lineage_i=lineage_i,
            material_count=weight,
            span_start=int(valid_breakpoints[0]) - 1,
            span_end=int(valid_breakpoints[-1]),
        )

    def _material_span(self, material_mask):
        if isinstance(material_mask, MaterialSegments):
            if material_mask.count < 2:
                return None
            return material_mask.span_start, material_mask.span_end, material_mask.count
        material_blocks = np.flatnonzero(np.asarray(material_mask, dtype=bool))
        if material_blocks.size < 2:
            return None
        return int(material_blocks[0]), int(material_blocks[-1]), int(material_blocks.size)

    def _split_mask(self, material_mask, breakpoint):
        if isinstance(material_mask, MaterialSegments):
            left, right = material_mask.split(breakpoint)
            return left.to_mask(self.num_blocks), right.to_mask(self.num_blocks)
        mask = np.asarray(material_mask, dtype=bool)
        left_mask = mask & (self.block_indices < breakpoint)
        right_mask = mask & (self.block_indices >= breakpoint)
        return left_mask, right_mask

    def _is_active_index(self, state, idx):
        return isinstance(idx, numbers.Integral) and 0 <= idx < len(state.active_lineages)
