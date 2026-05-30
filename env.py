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

from time_env import TimeEnvFixedDelta, TimeEnvLogDelta

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
    exact_delta_t: Optional[float] = None

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
        exact_delta_t = action.get("exact_delta_t")
        if exact_delta_t is not None and not isinstance(exact_delta_t, (numbers.Real, float)):
            return None
        return cls(
            active_lineage_i=int(active_lineage_i),
            material_count=int(material_count),
            span_start=int(span_start),
            span_end=int(span_end),
            time_action=int(time_action) if time_action is not None else None,
            breakpoint=int(breakpoint) if breakpoint is not None else None,
            exact_delta_t=float(exact_delta_t) if exact_delta_t is not None else None,
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
    exact_delta_t: Optional[float] = None

    def as_dict(self):
        action = {
            "event_type": "coal",
            "active_lineage_i": self.active_lineage_i,
            "active_lineage_j": self.active_lineage_j,
        }
        if self.time_action is not None:
            action["time_action"] = self.time_action
        if self.exact_delta_t is not None:
            action["exact_delta_t"] = self.exact_delta_t
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
        exact_delta_t = action.get("exact_delta_t")
        if exact_delta_t is not None and not isinstance(exact_delta_t, (numbers.Real, float)):
            return None
        return cls(
            active_lineage_i=int(i),
            active_lineage_j=int(j),
            time_action=int(time_action) if time_action is not None else None,
            exact_delta_t=float(exact_delta_t) if exact_delta_t is not None else None,
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
        if not copy_partials:
            partials = self.partials
        elif torch.is_tensor(self.partials):
            partials = self.partials.clone()
        else:
            partials = copy.deepcopy(self.partials)

        clone = ARGLineage(
            node_id=self.node_id,
            children=list(self.children),
            parents=list(self.parents),
            material_segments=self.material_segments,
            num_blocks=self.num_blocks,
            partials=partials,
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

    def clone(self, copy_partials=False):
        all_nodes = {
            node_id: lineage.clone(copy_partials=copy_partials)
            for node_id, lineage in self.all_nodes.items()
        }
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

    def __len__(self):
        return len(self.actions)


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

class ARGReward:
    """
    Terminal reward helpers for constructed ARG states.
    """

    def __init__(self, C=30000):
        self.C = C

    def __call__(self, log_likelihood, accumulated_log_prior):
        return self.compute_terminal_posterior_log_reward(log_likelihood, accumulated_log_prior)

    def compute_terminal_posterior_log_reward(self, log_likelihood, accumulated_log_prior):
        return float(self.C + log_likelihood + accumulated_log_prior)

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
        time_bins: Optional[int] = None,
        time_delta_bin_width: Optional[float] = None,
        time_bin_scheme: Optional[str] = None,
        time_rate_dependent: bool = False,
    ):
        self.sequences = list(sequences) if sequences is not None else None
        self.chars_dict = CHARACTERS_MAPS['DNA_WITH_GAP']
        self.event_types = ["coal", "recomb"]
        self.device = torch.device(device)

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
        time_env_kwargs = {}
        if time_bins is not None:
            time_env_kwargs["bins"] = int(time_bins)
        
        scheme = time_bin_scheme if time_bin_scheme is not None else "TimeEnvFixedDelta"
        if scheme == "TimeEnvFixedDelta":
            if time_delta_bin_width is not None:
                time_env_kwargs["delta_bin_width"] = float(time_delta_bin_width)
            time_env_kwargs["rate_dependent"] = time_rate_dependent
            self.time_env = TimeEnvFixedDelta(**time_env_kwargs)
        elif scheme == "TimeEnvLogDelta":
            time_env_kwargs["rate_dependent"] = time_rate_dependent
            self.time_env = TimeEnvLogDelta(**time_env_kwargs)
        else:
            raise ValueError(f"Unknown time bin scheme: {scheme}")

        self.rng = random.Random(seed)

        ## Sequence arrays
        self.block_indices = np.arange(self.num_blocks)

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
        meta = {
            "time_bin_scheme": type(self.time_env).__name__,
            "time_bins": int(self.time_env.bins),
        }
        if hasattr(self.time_env, "delta_bin_width"):
            meta["time_delta_bin_width"] = float(self.time_env.delta_bin_width)
        else:
            meta["min_time"] = float(self.time_env.min_time)
            meta["max_time"] = float(self.time_env.max_time)
        return meta

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
        material_segments = MaterialSegments.full(self.num_blocks)
        material_segments_list = [material_segments] * self.num_sequences
        partials_list = self._initial_lineages_partials_batch(material_segments_list)

        total_time = 0.0
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
            total_time += lineage.time
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

    def _is_latest_time_event(self, state, *node_ids):
        current_time = float(state.current_time)
        return all(
            math.isclose(
                float(state.all_nodes[node_id].time),
                current_time,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for node_id in node_ids
        )
        
    def _enumerate_inverse_arg_actions(self, state):
        inverse_actions = []

        # Use one loop to collect both coal and recomb candidates efficiently
        # Prepare coal candidates in a single pass with a list comprehension
        coal_candidates = [
            (active_idx, lineage)
            for active_idx, lineage in enumerate(state.active_lineages)
            if (
                lineage.event_type == "coal"
                and len(lineage.children) == 2
                and self._is_latest_time_event(state, lineage.node_id)
                and lineage.children[0] in state.all_nodes
                and lineage.children[1] in state.all_nodes
                and lineage.node_id in state.all_nodes[lineage.children[0]].parents
                and lineage.node_id in state.all_nodes[lineage.children[1]].parents
            )
        ]
        for active_idx, lineage in coal_candidates:
            child_i, child_j = lineage.children
            inverse_actions.append(
                {
                    "event_type": "coal",
                    "active_idx": active_idx,
                    "parent_id": lineage.node_id,
                    "child_ids": (child_i, child_j),
                }
            )

        # Prepare recomb_by_event using a single pass with a dictionary
        recomb_by_event = {}
        for active_idx, lineage in enumerate(state.active_lineages):
            if (
                lineage.event_type == "recomb"
                and len(lineage.children) == 1
                and lineage.breakpoint is not None
                and lineage.recombination_side in ("left", "right")
            ):
                key = (lineage.children[0], lineage.breakpoint)
                recomb_by_event.setdefault(key, {})[lineage.recombination_side] = (active_idx, lineage.node_id)

        # We can iterate efficiently over recomb_by_event rather than collecting in a list
        for (child_id, breakpoint), sides in recomb_by_event.items():
            if "left" not in sides or "right" not in sides or child_id not in state.all_nodes:
                continue
            left_idx, left_id = sides["left"]
            right_idx, right_id = sides["right"]
            child = state.all_nodes[child_id]
            left_parent = state.all_nodes[left_id]
            right_parent = state.all_nodes[right_id]

            # Fast short-circuit checks, in a single conditional
            if (
                not self._is_latest_time_event(state, left_id, right_id)
                or set(child.parents) != {left_id, right_id}
                or left_parent.material_segments.intersection_count(right_parent.material_segments) > 0
                or left_parent.material_segments.union(right_parent.material_segments) != child.material_segments
            ):
                continue

            inverse_actions.append(
                {
                    "event_type": "recomb",
                    "active_indices": (left_idx, right_idx),
                    "parent_ids": (left_id, right_id),
                    "child_id": child_id,
                    "breakpoint": breakpoint,
                }
            )

        return inverse_actions
    
    def count_backward_parents(self, arg_state):
        return len(self._enumerate_inverse_arg_actions(arg_state))

    def _initial_lineage_partials(self, node_id, material_segments):
        partials = self.block_seq_arrays[int(node_id)].detach().clone().float()
        return self.evolution_model.mask_partials(partials, material_segments)

    def _initial_lineages_partials_batch(self, material_segments_list):
        """Initialize tip partials for all sequences in one vectorized pass."""
        num_lineages = len(material_segments_list)
        if num_lineages != self.num_sequences:
            raise ValueError(
                f"Expected {self.num_sequences} material segment sets, got {num_lineages}"
            )

        reference_segments = material_segments_list[0]
        segments_match = all(
            ms.segments == reference_segments.segments for ms in material_segments_list
        )
        if segments_match:
            return [
                self._initial_lineage_partials(node_id, reference_segments)
                for node_id in range(num_lineages)
            ]

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

    def _recombined_parent_partials(self, child, parent_segments, parent_time):
        transitioned = self._transition_lineage_partials(child, parent_time)
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
        if int(state.total_active_blocks) == self.num_blocks:
            return True
        # If no further actions can be taken, treat it as terminal
        coal_actions, recomb_actions = self.enumerate_actions(state)
        if len(coal_actions) == 0 and len(recomb_actions) == 0:
            return True
        return False

    def _finalize_transition_state(self, next_state, log_prior):
        if log_prior is not None:
            next_state.accumulated_log_prior += log_prior
        next_state.is_done = self.is_terminal(next_state)
        if next_state.is_done:
            log_likelihood = self.evolution_model.compute_arg_log_likelihood(next_state)
            next_state.log_reward = self.compute_terminal_log_reward(next_state, log_likelihood)
        else:
            next_state.log_reward = None
        return next_state

    def apply_coalescence(self, state, action, log_prior=None):

        rates = state.rates
        if rates is None:
            rates = self.compute_event_rates(self.enumerate_actions(state))
            state.rates = rates

        next_state = state.clone(copy_partials=False)
        i = action.active_lineage_i
        j = action.active_lineage_j

        child_i = next_state.active_lineages[i].clone(copy_partials=False, copy_mask=False)
        child_j = next_state.active_lineages[j].clone(copy_partials=False, copy_mask=False)

        parent_id = next_state.max_node_idx + 1
        parent_segments = child_i.material_segments.union(child_j.material_segments)
        overlap_count = child_i.material_segments.intersection_count(child_j.material_segments)
        if getattr(action, "exact_delta_t", None) is not None:
            delta_t = action.exact_delta_t
        else:
            delta_t = self.time_env.time_action_to_delta(action.time_action, self._total_event_rate(rates))
        parent_time = float(state.current_time) + delta_t
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
        child_i.partials = None
        child_j.partials = None
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
        rates = state.rates
        if rates is None:
            rates = self.compute_event_rates(self.enumerate_actions(state))
            state.rates = rates

        # if log_prior is None:
        #     log_prior = self.compute_cwr_event_log_prior(state, action, rates=rates)
        next_state = state.clone(copy_partials=False)
        current_lineage_idx = action.active_lineage_i
        breakpoint = action.breakpoint
        child = next_state.active_lineages[current_lineage_idx].clone(copy_partials=False, copy_mask=False)
        left_segments, right_segments = child.material_segments.split(breakpoint)

        left_parent_id = next_state.max_node_idx + 1
        right_parent_id = next_state.max_node_idx + 2
        if getattr(action, "exact_delta_t", None) is not None:
            delta_t = action.exact_delta_t
        else:
            delta_t = self.time_env.time_action_to_delta(action.time_action, self._total_event_rate(rates))

        event_time = float(state.current_time) + delta_t
        next_state.current_time = event_time
        left_partials = self._recombined_parent_partials(child, left_segments, event_time)
        right_partials = self._recombined_parent_partials(child, right_segments, event_time)
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
        child.partials = None
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


    def _sample_prior_step(self, state):
        """Sample one prior coalescence/recombination action and its log prior."""
        event_types = ["coal", "recomb"]
        combined_actions = self.enumerate_actions(state)
        event_probs = list(self.compute_event_probabilities(state, combined_actions).values())
        chosen_event = event_types[np.random.choice(2, p=event_probs)]

        coal_actions, recomb_actions = combined_actions
        if chosen_event == "coal":
            chosen_action = self.rng.choice(coal_actions)
        else:
            prior_result = self._sample_recombination_prior_action(recomb_actions)
            if prior_result is None:
                raise ValueError("No valid recombination actions to sample")
            action_dict, _, selected = prior_result
            chosen_action = replace(
                selected,
                breakpoint=action_dict["breakpoint"],
            )

        time_action = self.time_env.sample_action_from_prior(
            self._total_event_rate(state.rates), self.rng
        )
        chosen_action = replace(chosen_action, time_action=time_action)
        log_prior = self.compute_cwr_event_log_prior(state, combined_actions, chosen_action)
        return chosen_action, log_prior

    def evaluate_state_action_log_pf(self, state, action):

        coal_candidates, recomb_candidates = self.enumerate_actions(state)
        event_probs = list(self.compute_event_probabilities(state, (coal_candidates, recomb_candidates)).values())

        event_pf = float(event_probs[0]) if isinstance(action, CoalescenceChoice) else float(event_probs[1])
        event_log_pf = np.log(event_pf)

        coal_candidates_probs = np.array([act.probability(state) for act in coal_candidates])
        coal_probs = coal_candidates_probs / np.sum(coal_candidates_probs)

        recomb_candidates_probs = np.array([act.probability(state) for act in recomb_candidates])
        recomb_probs = recomb_candidates_probs / np.sum(recomb_candidates_probs)

        if isinstance(action, CoalescenceChoice):
            coal_action_idx = coal_candidates.index(action)
            action_prob = float(coal_probs[coal_action_idx])
            action_log_pf = np.log(action_prob)
            breakpoint_log_pf = 0.0
        else:
            recomb_action_idx = recomb_candidates.index(action)
            action_prob = float(recomb_probs[recomb_action_idx])
            action_log_pf = np.log(action_prob)

            breakpoint_prob = float(action.probability(state))
            breakpoint_log_pf = np.log(breakpoint_prob)

        time_action = action.time_action
        time_prob = self.time_env.time_action_probability(time_action, self._total_event_rate(state.rates))
        time_log_pf = np.log(time_prob)

        total_log_pf = event_log_pf + action_log_pf + breakpoint_log_pf + time_log_pf
        probs = np.exp(total_log_pf)
        return total_log_pf, probs

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
                action, log_prior = self._sample_prior_step(state)
                state = self.apply_action(state, action, log_prior=log_prior)
            log_rewards.append(state.log_reward)
        return log_rewards

    def compute_cwr_event_log_prior(self, state, combined_actions, action=None, rates=None):
        if action and state is None:
            action = combined_actions
            combined_actions = self.enumerate_actions(state)
        coal_actions, recomb_actions = combined_actions

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

    def prepare_state_rollout_inputs(
        self,
        states,
        random_spec=None,
        window_start=0,
        window_end=None,
    ):
        batch_size = len(states)
        if batch_size == 0:
            raise ValueError("states must contain at least one ARGState")

        event = {}
        input_actions = []
        for idx, state in enumerate(states):
            coal_actions, recomb_actions = self.enumerate_actions(state)
            event_prob = list(self.compute_event_probabilities(state, (coal_actions, recomb_actions)).values())
            event_idx = np.random.choice(2, p=event_prob)
            choosen_event_type = self.event_types[event_idx]
            if choosen_event_type == "coal":
                input_actions.append(coal_actions)
            else:
                input_actions.append(recomb_actions)

            event[idx] = {}
            event[idx]["event_type"] = choosen_event_type
            event[idx]["probability"] = event_prob[event_idx]

        if window_end is None:
            window_end = self.num_blocks

        if not isinstance(window_start, (list, tuple, np.ndarray, torch.Tensor)):
            window_starts = [window_start] * batch_size
        else:
            window_starts = window_start

        if not isinstance(window_end, (list, tuple, np.ndarray, torch.Tensor)):
            window_ends = [window_end] * batch_size
        else:
            window_ends = window_end

        region_contexts = torch.tensor(
            [[float(ws) / self.num_blocks, float(we) / self.num_blocks] for ws, we in zip(window_starts, window_ends)],
            dtype=torch.float32,
            device=self.device
        )

        input_dict = {
            "states": states,
            "event": event,
            "input_actions": input_actions,
            "random_spec": random_spec,
            "region_contexts": region_contexts,
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

    def delete_genomic_window(self, state, window_start, window_end):
        """
        Delete all ancestral nodes in state that carry material inside the window [window_start, window_end) (in blocks),
        update parent/child relationships, rebuild partials topologically, and return the updated ARGState.
        """
        window_start = int(window_start)
        window_end = int(window_end)
        if window_start >= window_end:
            raise ValueError("window_start must be less than window_end")
            
        # 1. Identify nodes to delete (nodes with material in window, excluding samples)
        deletion_set = set()
        window_seg = MaterialSegments(((window_start, window_end),))
        
        for node_id, lineage in state.all_nodes.items():
            if node_id < self.num_sequences:
                continue
            if lineage.material_segments.intersection_count(window_seg) > 0:
                deletion_set.add(node_id)
                
        # 2. Recursively find all ancestors of the deleted nodes to delete them too
        to_check = list(deletion_set)
        while to_check:
            node_id = to_check.pop()
            lineage = state.all_nodes[node_id]
            for p in lineage.parents:
                if p not in deletion_set:
                    deletion_set.add(p)
                    to_check.append(p)
                    
        # 3. Clone state and delete the nodes
        new_state = state.clone(copy_partials=True)
        for node_id in deletion_set:
            new_state.all_nodes.pop(node_id, None)
            
        # 4. Update parent and child lists for remaining nodes
        for lineage in new_state.all_nodes.values():
            lineage.parents = [p for p in lineage.parents if p not in deletion_set]
            lineage.children = [c for c in lineage.children if c not in deletion_set]
            
        # 5. Determine new active lineages (remaining nodes with no parents)
        new_state.active_lineages = [
            lineage for lineage in new_state.all_nodes.values()
            if len(lineage.parents) == 0
        ]
        
        # 6. Recompute partials bottom-up
        sorted_nodes = sorted(
            new_state.all_nodes.values(),
            key=lambda l: (l.time, l.node_id)
        )
        
        # Reset partials for all remaining nodes to None first, except samples
        for lineage in sorted_nodes:
            if lineage.node_id >= self.num_sequences:
                lineage.partials = None
                
        # Now recompute bottom-up
        for lineage in sorted_nodes:
            nid = lineage.node_id
            if nid < self.num_sequences:
                lineage.partials = self._initial_lineage_partials(nid, lineage.material_segments)
            else:
                if lineage.event_type == "coal":
                    child_i = new_state.all_nodes[lineage.children[0]]
                    child_j = new_state.all_nodes[lineage.children[1]]
                    lineage.partials = self._coalesced_parent_partials(
                        child_i,
                        child_j,
                        lineage.material_segments,
                        lineage.time
                    )
                elif lineage.event_type == "recomb":
                    child = new_state.all_nodes[lineage.children[0]]
                    lineage.partials = self._recombined_parent_partials(
                        child,
                        lineage.material_segments,
                        lineage.time
                    )
                else:
                    raise ValueError(f"Unknown event type: {lineage.event_type} for node {nid}")
                    
        # 7. Update other state fields
        new_state.max_node_idx = max(new_state.all_nodes.keys()) if new_state.all_nodes else -1
        new_state.current_time = max((l.time for l in new_state.active_lineages), default=0.0)
        new_state.total_active_blocks = sum(l.material_count for l in new_state.active_lineages)
        new_state.is_done = self.is_terminal(new_state)
        new_state.log_reward = None
        new_state.rates = None
        new_state.prior_options = None
        new_state.action_options = None
        
        if new_state.is_done:
            log_likelihood = self.evolution_model.compute_arg_log_likelihood(new_state)
            new_state.log_reward = self.compute_terminal_log_reward(new_state, log_likelihood)
            
        return new_state

    def reconstruct_prefix_trajectory(self, state):
        """
        Reconstruct the chronological sequence of states and actions that leads from
        get_initial_state() to the partial `state`.
        """
        recomb_events = []
        coal_events = []
        
        sorted_nodes = sorted(
            state.all_nodes.values(),
            key=lambda l: (l.time, l.node_id)
        )
        
        processed_recomb_children = set()
        for lineage in sorted_nodes:
            if lineage.node_id < self.num_sequences:
                continue
            if lineage.event_type == "recomb":
                child_id = lineage.children[0]
                if child_id not in processed_recomb_children:
                    processed_recomb_children.add(child_id)
                    recomb_events.append({
                        'event_type': 'recomb',
                        'child_id': child_id,
                        'breakpoint': lineage.breakpoint,
                        'time': lineage.time
                    })
            elif lineage.event_type == "coal":
                coal_events.append({
                    'event_type': 'coal',
                    'ts_node_id': lineage.node_id,
                    'children': lineage.children,
                    'time': lineage.time
                })
                
        events = recomb_events + coal_events
        events.sort(key=lambda x: x['time'])
        
        curr_state = self.get_initial_state()
        prefix_states = [curr_state.clone(copy_partials=True)]
        prefix_actions = []
        
        node_id_to_ts_node = {i: i for i in range(self.num_sequences)}
        
        for event in events:
            if event['event_type'] == 'recomb':
                bp = event['breakpoint']
                target_idx = None
                for idx, l in enumerate(curr_state.active_lineages):
                    if node_id_to_ts_node.get(l.node_id) == event['child_id']:
                        if l.material_segments.covers_interval(bp - 1, bp + 1):
                            target_idx = idx
                            break
                if target_idx is None:
                    for idx, l in enumerate(curr_state.active_lineages):
                        if node_id_to_ts_node.get(l.node_id) == event['child_id']:
                            target_idx = idx
                            break
                if target_idx is None:
                    raise ValueError(f"Prefix reconstruction failed: child_id={event['child_id']} not found in active lineages.")
                    
                lineage = curr_state.active_lineages[target_idx]
                rates = self.compute_event_rates(self.enumerate_actions(curr_state))
                curr_state.rates = rates
                delta_t = max(1e-10, event['time'] - curr_state.current_time)
                time_action = self.time_env.delta_to_time_action(delta_t, self._total_event_rate(rates))
                
                action = RecombinationChoice(
                    active_lineage_i=target_idx,
                    material_count=lineage.material_count,
                    span_start=lineage.material_segments.span_start,
                    span_end=lineage.material_segments.span_end,
                    time_action=time_action,
                    breakpoint=bp,
                    exact_delta_t=delta_t
                )
                
                combined_actions = self.enumerate_actions(curr_state)
                log_prior = self.compute_cwr_event_log_prior(curr_state, combined_actions, action)
                curr_state = self.apply_action(curr_state, action, log_prior=log_prior)
                
                prefix_states.append(curr_state.clone(copy_partials=True))
                prefix_actions.append(action)
                
                target_left_parent_id = None
                target_right_parent_id = None
                for n_id, n in state.all_nodes.items():
                    if n.event_type == "recomb" and n.children and n.children[0] == event['child_id'] and n.breakpoint == bp:
                        if n.recombination_side == "left":
                            target_left_parent_id = n_id
                        elif n.recombination_side == "right":
                            target_right_parent_id = n_id

                parent_id_1 = curr_state.max_node_idx - 1
                parent_id_2 = curr_state.max_node_idx
                node_id_to_ts_node[parent_id_1] = target_left_parent_id
                node_id_to_ts_node[parent_id_2] = target_right_parent_id
                
            elif event['event_type'] == 'coal':
                children_ts_ids = set(event['children'])
                active_indices = [
                    idx for idx, lineage in enumerate(curr_state.active_lineages)
                    if node_id_to_ts_node.get(lineage.node_id) in children_ts_ids
                ]
                
                while len(active_indices) > 1:
                    found_pair = False
                    for idx1 in range(len(active_indices)):
                        for idx2 in range(idx1 + 1, len(active_indices)):
                            i = active_indices[idx1]
                            j = active_indices[idx2]
                            if curr_state.active_lineages[i].material_segments.overlaps(curr_state.active_lineages[j].material_segments):
                                if i > j:
                                    i, j = j, i
                                
                                rates = self.compute_event_rates(self.enumerate_actions(curr_state))
                                curr_state.rates = rates
                                delta_t = max(1e-10, event['time'] - curr_state.current_time)
                                time_action = self.time_env.delta_to_time_action(delta_t, self._total_event_rate(rates))
                                
                                action = CoalescenceChoice(
                                    active_lineage_i=i,
                                    active_lineage_j=j,
                                    time_action=time_action,
                                    exact_delta_t=delta_t
                                )
                                
                                combined_actions = self.enumerate_actions(curr_state)
                                log_prior = self.compute_cwr_event_log_prior(curr_state, combined_actions, action)
                                curr_state = self.apply_action(curr_state, action, log_prior=log_prior)
                                
                                prefix_states.append(curr_state.clone(copy_partials=True))
                                prefix_actions.append(action)
                                
                                node_id_to_ts_node[curr_state.max_node_idx] = event['ts_node_id']
                                
                                active_indices = [
                                    idx for idx, lineage in enumerate(curr_state.active_lineages)
                                    if node_id_to_ts_node.get(lineage.node_id) in children_ts_ids
                                ]
                                found_pair = True
                                break
                        if found_pair:
                            break
                    if not found_pair:
                        break

        return prefix_states[:-1], prefix_actions
