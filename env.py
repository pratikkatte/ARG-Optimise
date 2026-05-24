import copy
import math
import numbers
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from dataclasses import replace
from evo import EvolutionModelTorch

import numpy as np

from time_env import TimeEnvFixedDelta

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
    time_action: Optional[int] = None
    breakpoint: Optional[int] = None

    @property
    def breakpoint_count(self):
        return int(self.span_end - self.span_start)

    def as_weight_tuple(self):
        return (
            self.active_lineage_i,
            self.material_count,
            list(range(self.span_start + 1, self.span_end + 1)),
        )

    @classmethod
    def from_action(cls, action):
        if isinstance(action, cls):
            return action
        if not isinstance(action, dict) or action.get("event_type") != "recomb":
            return None
        active_lineage_i = action.get("active_lineage_i")
        material_count = action.get("material_count")
        span_start = action.get("span_start")
        span_end = action.get("span_end")
        if not isinstance(active_lineage_i, numbers.Integral) or not isinstance(material_count, numbers.Integral) or not isinstance(span_start, numbers.Integral) or not isinstance(span_end, numbers.Integral):
            return None
        time_action = action.get("time_action")
        if time_action is not None and not isinstance(time_action, numbers.Integral):
            return None
        breakpoint = action.get("breakpoint")
        if breakpoint is not None and not isinstance(breakpoint, numbers.Integral):
            return None
        return cls(
            active_lineage_i=int(active_lineage_i),
            material_count=int(material_count),
            span_start=int(span_start),
            span_end=int(span_end),
            time_action=int(time_action) if time_action is not None else None,
            breakpoint=int(breakpoint) if breakpoint is not None else None,
        )

    def is_valid_for(self, active_lineages):
        return self.active_lineage_i < len(active_lineages) and self.span_start < self.span_end

    @classmethod
    def enumerate_from_active_lineages(cls, active_lineages):
        choices = []
        for i, lineage in enumerate(active_lineages):
            span = lineage.material_span
            if span is None:
                continue
            first_block, last_block, material_count = span
            if first_block < last_block:
                choices.append(
                    cls(
                        active_lineage_i=i,
                        material_count=int(material_count),
                        span_start=int(first_block),
                        span_end=int(last_block),
                    )
                )
        return tuple(choices)


@dataclass(frozen=True)
class CoalescenceChoice:
    active_lineage_i: int
    active_lineage_j: int
    time_action: Optional[int] = None

    def as_dict(self):
        action = {
            "event_type": "coal",
            "active_lineage_i": self.active_lineage_i,
            "active_lineage_j": self.active_lineage_j,
        }
        if self.time_action is not None:
            action["time_action"] = self.time_action
        return action

    def is_valid_for(self, active_lineages):
        i = self.active_lineage_i
        j = self.active_lineage_j
        if i == j:
            return False
        if not (0 <= i < len(active_lineages) and 0 <= j < len(active_lineages)):
            return False
        return active_lineages[i].material_segments.overlaps(
            active_lineages[j].material_segments
        )

    @classmethod
    def from_action(cls, action):
        if isinstance(action, cls):
            return action
        if not isinstance(action, dict) or action.get("event_type") != "coal":
            return None
        i = action.get("active_lineage_i")
        j = action.get("active_lineage_j")
        if not isinstance(i, numbers.Integral) or not isinstance(j, numbers.Integral):
            return None
        time_action = action.get("time_action")
        if time_action is not None and not isinstance(time_action, numbers.Integral):
            return None
        return cls(
            active_lineage_i=int(i),
            active_lineage_j=int(j),
            time_action=int(time_action) if time_action is not None else None,
        )

    @classmethod
    def enumerate_from_active_lineages(cls, active_lineages):
        events = []
        for active_idx, lineage in enumerate(active_lineages):
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

        return tuple(
            cls(active_lineage_i=i, active_lineage_j=j)
            for i, j in sorted(pairs)
        )


@dataclass(frozen=True)
class PriorActionOptions:
    coal_actions: Tuple[CoalescenceChoice, ...]
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


class Trajectory:
    """Full rollout history with cloned ARG states at each step."""

    def __init__(self, initial_state):
        self.current_state = initial_state.clone()
        self.transitions = []
        self.actions = []
        self.log_reward = None
        self.done = False

    def update(self, next_state, action, log_prior, done):
        self.transitions.append(    
            [self.current_state, next_state.clone(),log_prior, done]
        )
        self.actions.append(action)
        self.current_state = next_state.clone()
        self.done = done
        if next_state.log_reward is not None:
            self.log_reward = next_state.log_reward

    def update_reward(self, log_reward):
        if self.transitions:
            self.transitions[-1][3] = log_reward
        self.log_reward = log_reward

    def __len__(self):
        return len(self.transitions)


class SimpleTrajectory:
    """Compact trajectory history used when cloned ARG states are not needed."""

    def __init__(self):
        self.actions = []
        self.log_priors = []
        self.records = []
        self.log_reward = None

    def update(self, action, log_prior=None, log_reward=None, record=None, active_lineages=None):
        self.actions.append(action_as_dict(action))
        self.log_priors.append(log_prior)
        self.log_reward = log_reward
        if record is not None:
            self.records.append(record)


def action_as_dict(action):
    if isinstance(action, dict):
        return dict(action)
    if isinstance(action, CoalescenceChoice):
        return action.as_dict()
    if isinstance(action, RecombinationChoice):
        result = {
            "event_type": "recomb",
            "active_lineage_i": int(action.active_lineage_i),
            "breakpoint": int(action.breakpoint) if action.breakpoint is not None else None,
            "material_count": int(action.material_count),
            "span_start": int(action.span_start),
            "span_end": int(action.span_end),
        }
        if action.time_action is not None:
            result["time_action"] = int(action.time_action)
        return result
    raise ValueError(f"Unknown ARG action: {action}")


def canonicalize_action(action, active_lineages=None):
    coal_action = CoalescenceChoice.from_action(action)
    if coal_action is not None:
        return coal_action

    recomb_action = RecombinationChoice.from_action(action)
    if recomb_action is not None:
        return recomb_action

    if not isinstance(action, dict) or action.get("event_type") != "recomb":
        raise ValueError(f"Unknown ARG action: {action}")

    active_lineage_i = action.get("active_lineage_i")
    breakpoint = action.get("breakpoint")
    if not isinstance(active_lineage_i, numbers.Integral) or not isinstance(breakpoint, numbers.Integral):
        raise ValueError(f"Invalid recombination action: {action}")

    if active_lineages is None:
        raise ValueError(
            "Recombination actions without material metadata require active_lineages"
        )
    active_lineage_i = int(active_lineage_i)
    if not 0 <= active_lineage_i < len(active_lineages):
        raise ValueError(f"Invalid active lineage index in recombination action: {action}")

    span = active_lineages[active_lineage_i].material_span
    if span is None:
        raise ValueError(f"Active lineage has no valid recombination span: {action}")
    span_start, span_end, material_count = span
    time_action = action.get("time_action")
    return RecombinationChoice(
        active_lineage_i=active_lineage_i,
        material_count=int(action.get("material_count", material_count)),
        span_start=int(action.get("span_start", span_start)),
        span_end=int(action.get("span_end", span_end)),
        breakpoint=int(breakpoint),
        time_action=int(time_action) if time_action is not None else None,
    )

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
    canonical CWR prior plus a learned-time JC69 sequence likelihood.
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
    ):
        self.sequences = list(sequences) if sequences is not None else None
        self.chars_dict = CHARACTERS_MAPS['DNA_WITH_GAP']
        self.event_types = ["coal", "recomb"]
        self.device = device

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
        self.time_env = TimeEnvFixedDelta()

        self.rng = random.Random(seed)

        ## Sequence arrays
        self.block_indices = np.arange(self.num_blocks)

        seq_arrays = np.array([self.seq2array(seq) for seq in self.sequences], dtype=np.float32)

        self.seq_arrays = torch.nn.Parameter(
            torch.tensor(seq_arrays, dtype=torch.float32),
            requires_grad=False,
        )
        
        ## Evolution model
        self.evolution_model = EvolutionModelTorch(self)

        ## Reward function 
        self.reward_fn = ARGReward()

    def seq2array(self, seq):
        seq = [self.chars_dict[x] for x in seq]
        data = np.array(seq)
        return data

    def _total_event_rate(self, rates):
        total_rate = float(rates["lambda_coal"] + rates["lambda_recomb"])
        if total_rate <= 0 or total_rate is None:
            raise ValueError("waiting-time rate must be positive")
        return total_rate

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
            state.log_reward = self.compute_terminal_log_reward(state, log_likelihood)
        return state

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
        log_likelihood = self.reward_fn(log_likelihood)
        return float(log_likelihood + state.accumulated_log_prior)

    def compute_coalescence_actions(self, state):
        return list(CoalescenceChoice.enumerate_from_active_lineages(state.active_lineages))

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
            actions.extend(choice.as_dict() for choice in prior_options.coal_actions)
        if prior_options.rates["lambda_recomb"] > 0:
            actions.extend(
                {
                    "event_type": "recomb",
                    "active_lineage_i": int(choice.active_lineage_i),
                    "material_count": int(choice.material_count),
                    "span_start": int(choice.span_start),
                    "span_end": int(choice.span_end),
                }
                for choice in prior_options.recomb_choices
                if choice.breakpoint_count > 0
            )
        return actions


    def is_terminal(self, state):
        if state.total_active_blocks is None:
            raise ValueError("total_active_blocks is required for terminal check")
        else:
            result = int(state.total_active_blocks) == self.num_blocks
            # bool(np.all(self.get_active_counts(state) == 1)) ## another way, realtime compute. 
            return result

    def _finalize_transition_state(self, next_state, log_prior, compute_reward=True):
        if log_prior is not None:
            next_state.accumulated_log_prior += log_prior
        next_state.is_done = self.is_terminal(next_state)
        if next_state.is_done and compute_reward:
            log_likelihood = self.evolution_model.compute_arg_log_likelihood(next_state)
            next_state.log_reward = self.compute_terminal_log_reward(next_state, log_likelihood)
        else:
            next_state.log_reward = None
        return next_state

    def apply_coalescence(self, state, action, log_prior=None, compute_reward=True):

        rates = state.rates

        next_state = state.clone()
        i = action.active_lineage_i
        j = action.active_lineage_j

        child_i = next_state.active_lineages[i].clone(copy_partials=False, copy_mask=False)
        child_j = next_state.active_lineages[j].clone(copy_partials=False, copy_mask=False)

        parent_id = next_state.max_node_idx + 1
        parent_segments = child_i.material_segments.union(child_j.material_segments)
        overlap_count = child_i.material_segments.intersection_count(child_j.material_segments)
        delta_t = self.time_env.time_action_to_delta(action.time_action, self._total_event_rate(rates))
        parent_time = float(state.current_time) + delta_t
        next_state.current_time = parent_time
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
        return self._finalize_transition_state(next_state, log_prior, compute_reward=compute_reward)

    def apply_recombination(self, state, action, log_prior=None, compute_reward=True):
        rates = state.rates

        # if log_prior is None:
        #     log_prior = self.compute_cwr_event_log_prior(state, action, rates=rates)
        next_state = state.clone()
        current_lineage_idx = action.active_lineage_i
        breakpoint = action.breakpoint
        child = next_state.active_lineages[current_lineage_idx].clone(copy_partials=False, copy_mask=False)
        left_segments, right_segments = child.material_segments.split(breakpoint)

        left_parent_id = next_state.max_node_idx + 1
        right_parent_id = next_state.max_node_idx + 2
        delta_t = self.time_env.time_action_to_delta(action.time_action, self._total_event_rate(rates))

        event_time = float(state.current_time) + delta_t
        next_state.current_time = event_time
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
            lineage for idx, lineage in enumerate(next_state.active_lineages) if idx != current_lineage_idx
        ]
        next_state.active_lineages.extend([left_parent, right_parent])
        next_state.max_node_idx = right_parent.node_id
        return self._finalize_transition_state(next_state, log_prior, compute_reward=compute_reward)

    def apply_action(self, state, action, log_prior=None, compute_reward=True):
        if isinstance(action, dict):
            action = canonicalize_action(action, active_lineages=state.active_lineages)
        if isinstance(action, RecombinationChoice):
            return self.apply_recombination(
                state,
                action,
                log_prior,
                compute_reward=compute_reward,
            )
        elif isinstance(action, CoalescenceChoice):
            return self.apply_coalescence(
                state,
                action,
                log_prior,
                compute_reward=compute_reward,
            )
        else:
            raise ValueError(f"Unknown action event_type: {action}")

    def compute_event_rates(self, actions):
        coal_actions, recomb_actions = actions

        lambda_coal = float(len(coal_actions))

        total_blocks = sum(choice.material_count for choice in recomb_actions)
        total_active_material_length = float(total_blocks) / float(self.num_blocks)
        lambda_recomb = self.rho / 2.0 * total_active_material_length
        
        return {
            "lambda_coal": lambda_coal,
            "lambda_recomb": lambda_recomb,
            "total_active_material_length": total_active_material_length,
        }

    def compute_event_probabilities(self, state, actions=None):
        if actions is None:
            actions = self.enumerate_actions(state)
        rates = self.compute_event_rates(actions)
        state.rates = rates
        denom = rates["lambda_coal"] + rates["lambda_recomb"]
        if denom <= 0:
            return {"coal": 0.0, "recomb": 0.0}
        return {
            "coal": rates["lambda_coal"] / denom,
            "recomb": rates["lambda_recomb"] / denom,
        }

    def enumerate_actions(self, state):

        coal_actions = self.compute_coalescence_actions(state)
        recomb_actions = self.compute_recombination_actions(state)

        return coal_actions, recomb_actions


    def sample(self,num_trajs):
        """Sample terminal ARG states.

        Without a generator this preserves the prior-only sampler. With a
        generator, event types are sampled from environment event rates and the
        model chooses the concrete action through RolloutWorker.
        """

        states = [self.get_initial_state() for _ in range(num_trajs)]
        trajectories = [Trajectory(x) for x in states]
        event_types = ["coal", "recomb"]
        unfinished = [idx for idx, state in enumerate(states) if not state.is_done]
        
        while unfinished:
            combined_actions = [[]]*num_trajs
            event_probs = [[]]*num_trajs
            for idx in unfinished:
                combined_actions[idx] = self.enumerate_actions(states[idx])
                event_probs[idx] = list(self.compute_event_probabilities(states[idx], combined_actions[idx]).values())

            choosen_event_types = [event_types[np.random.choice(2, p=event_probs[idx])] for idx in unfinished]
            choosen_actions = []
            for idx, choosen_event in zip(unfinished, choosen_event_types):
                coal_actions, recomb_actions = combined_actions[idx]
                if choosen_event == "coal":
                    choosen_action = self.rng.choice(coal_actions)
                else:
                    choosen_action = self.rng.choice(recomb_actions)
                    
                    ## choose breakpoint.
                    breakpoint = self.rng.choice(range(choosen_action.span_start + 1, choosen_action.span_end + 1))
                    choosen_action = replace(choosen_action, breakpoint=breakpoint)

                ## Choose Time.
                time_action = self.time_env.sample_action_from_prior(self._total_event_rate(states[idx].rates), self.rng)
                choosen_action = replace(choosen_action, time_action=time_action)
                    
                choosen_actions.append(choosen_action)
     
                log_prior = self.compute_cwr_event_log_prior(states[idx], combined_actions[idx], choosen_action)

                next_state = self.apply_action(
                    states[idx],
                    choosen_action,
                    log_prior=log_prior,
                )
                states[idx] = next_state
                trajectories[idx].update(next_state, choosen_action, log_prior=log_prior, done=next_state.is_done)
            unfinished = [idx for idx, state in enumerate(states) if not state.is_done]
        return trajectories

    # def sample_action_from_prior(self, state):
    #     coal_actions = self.compute_coalescence_actions(state)
    #     recomb_choices = self.compute_recombination_actions(state)
    #     rates = self.compute_event_rates((coal_actions, recomb_choices))
    #     denom = rates["lambda_coal"] + rates["lambda_recomb"]
    #     if denom <= 0:
    #         return None

    #     coal_prob = rates["lambda_coal"] / denom if coal_actions else 0.0
    #     recomb_prob = rates["lambda_recomb"] / denom if recomb_choices else 0.0

    #     if coal_prob > 0 and (recomb_prob <= 0 or self.rng.random() < coal_prob):
    #         action = self._with_random_time_action(
    #             coal_actions[self.rng.randrange(len(coal_actions))].as_dict(),
    #             rates,
    #         )
    #         action = canonicalize_action(action, active_lineages=state.active_lineages)
    #         return action, self.compute_cwr_event_log_prior(state, action)

    #     if recomb_prob <= 0:
    #         return None
    #     sampled = self._sample_recombination_prior_action(recomb_choices)
    #     if sampled is None:
    #         return None
    #     action, _lineage_weight, _choice = sampled
    #     action = self._with_random_time_action(action, rates)
    #     action = canonicalize_action(action, active_lineages=state.active_lineages)
    #     return action, self.compute_cwr_event_log_prior(state, action)

    def compute_cwr_event_log_prior(self, state, combined_actions, action=None, rates=None):
        if action is None:
            action = combined_actions
            combined_actions = self.enumerate_actions(state)
        coal_actions, recomb_actions = combined_actions
        if not isinstance(action, (CoalescenceChoice, RecombinationChoice)):
            action = canonicalize_action(action, active_lineages=state.active_lineages)
        if rates is None:
            rates = state.rates if state.rates is not None else self.compute_event_rates((coal_actions, recomb_actions))
        state.rates = rates
        
        total_rate = self._total_event_rate(rates)
        recomb_total_weight = sum(choice.material_count for choice in recomb_actions)

        wait_log_prior = self.time_env.time_action_log_probability(action.time_action, total_rate)

        if isinstance(action, CoalescenceChoice) and CoalescenceChoice.is_valid_for(action, state.active_lineages):
            action_log_prior = math.log((rates["lambda_coal"] / total_rate) / len(coal_actions))
            
        elif isinstance(action, RecombinationChoice) and RecombinationChoice.is_valid_for(action, state.active_lineages):
            action_log_prior = math.log((rates["lambda_recomb"] / total_rate) * (action.material_count / recomb_total_weight) / action.breakpoint_count)
        else:
            raise ValueError(f"Invalid action: {action}")

        return action_log_prior + wait_log_prior

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
            input_dict["input_actions"] = input_actions
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

    def prepare_state_rollout_inputs(
        self,
        states,
    ):
        batch_size = len(states)
        if batch_size == 0:
            raise ValueError("states must contain at least one ARGState")

        input_actions = []
        for state in states:
            coal_actions, recomb_actions = self.enumerate_actions(state)
            event_prob = list(self.compute_event_probabilities(state, (coal_actions, recomb_actions)).values())
            choosen_event_types = self.event_types[np.random.choice(2, p=event_prob)]
            if choosen_event_types == "coal":
                input_actions.append(coal_actions)
            else:
                input_actions.append(recomb_actions)
        input_dict = {
            "states": states,
            input_actions: input_actions,
        }

        return input_dict

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
