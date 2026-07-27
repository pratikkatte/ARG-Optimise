import copy
import math
import numbers
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from dataclasses import replace

try:
    from .evo import EvolutionModelTorch
    from .time_env import TimeEnvFixedDelta
except ImportError:  # Support the repository's legacy script-style imports.
    from evo import EvolutionModelTorch
    from time_env import TimeEnvFixedDelta

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

    def to_block_tensor(self, device):
        device = torch.device(device)
        if not self.segments:
            return torch.empty(0, dtype=torch.long, device=device)
        chunks = [
            torch.arange(int(start), int(end), dtype=torch.long, device=device)
            for start, end in self.segments
            if int(end) > int(start)
        ]
        if not chunks:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.cat(chunks, dim=0)

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
    delta_time: Optional[float] = None

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
        delta_time = action.get("delta_time")
        if delta_time is not None and not isinstance(delta_time, numbers.Real):
            return None
        return cls(
            active_lineage_i=int(active_lineage_i),
            material_count=int(material_count),
            span_start=int(span_start),
            span_end=int(span_end),
            time_action=int(time_action) if time_action is not None else None,
            breakpoint=int(breakpoint) if breakpoint is not None else None,
            delta_time=(
                float(delta_time) if delta_time is not None else None
            ),
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
    delta_time: Optional[float] = None

    def as_dict(self):
        action = {
            "event_type": "coal",
            "active_lineage_i": self.active_lineage_i,
            "active_lineage_j": self.active_lineage_j,
        }
        if self.time_action is not None:
            action["time_action"] = self.time_action
        if self.delta_time is not None:
            action["delta_time"] = float(self.delta_time)
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
        delta_time = action.get("delta_time")
        if delta_time is not None and not isinstance(delta_time, numbers.Real):
            return None
        return cls(
            active_lineage_i=int(i),
            active_lineage_j=int(j),
            time_action=int(time_action) if time_action is not None else None,
            delta_time=(
                float(delta_time) if delta_time is not None else None
            ),
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
        variant_indices: Optional[Sequence[int]] = None,
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
        self._block_indices_cache = {}

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
        if variant_indices is None:
            self.variant_indices = (
                tuple(self.material_segments.to_block_list())
                if partials is not None
                else ()
            )
        else:
            self.variant_indices = tuple(int(value) for value in variant_indices)

    @property
    def material_mask(self):
        if self._material_mask is None:
            self._material_mask = self.material_segments.to_mask(self.num_blocks)
        return self._material_mask

    @material_mask.setter
    def material_mask(self, value):
        if value is None:
            self._material_mask = None
            self.clear_runtime_caches()
            return
        self._material_mask = np.asarray(value, dtype=bool).copy()
        self.num_blocks = int(self._material_mask.size)
        self.material_segments = MaterialSegments.from_mask(self._material_mask)
        self.clear_runtime_caches()

    @property
    def material_count(self):
        return self.material_segments.count

    def clear_runtime_caches(self):
        if hasattr(self, "_block_indices_cache"):
            self._block_indices_cache.clear()

    def block_indices_tensor(self, device):
        device = torch.device(device)
        key = str(device)
        cached = self._block_indices_cache.get(key)
        if cached is None:
            cached = self.material_segments.to_block_tensor(device)
            self._block_indices_cache[key] = cached
        return cached

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
            variant_indices=self.variant_indices,
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
    log_likelihood: Optional[float] = None
    accumulated_log_prior: float = 0.0
    accumulated_log_likelihood: float = 0.0
    outside_log_likelihood: float = 0.0
    partial_log_reward: float = 0.0
    terminal_partial_correction: float = 0.0
    is_done: bool = False
    action_options: Tuple[List[Dict[str, Any]], List[Tuple[int, int, List[int]]], List[Dict[str, Any]]] = None
    rates: Optional[Dict[str, float]] = None
    prior_options: Optional[PriorActionOptions] = None
    total_active_blocks: Optional[int] = None
    current_time: float = 0.0
    target_material: Optional[MaterialSegments] = None
    block_boundaries: Optional[Tuple[float, ...]] = None
    time_scale: float = 1.0
    generated_node_start: Optional[int] = None
    transition_records: List[Dict[str, Any]] = field(default_factory=list)
    fixed_ancestor_schedule: List[Dict[str, Any]] = field(
        default_factory=list
    )
    target_variant_indices: Tuple[int, ...] = field(default_factory=tuple)
    variant_block_indices: Dict[int, int] = field(default_factory=dict)
    local_breakpoint_weights: Dict[int, float] = field(default_factory=dict)
    vcf_alignment: Dict[str, Any] = field(default_factory=dict)
    likelihood_scope: str = "none"
    local_context_id: Optional[str] = None
    local_target_interval: Optional[Tuple[float, float]] = None
    local_cut_time: Optional[float] = None
    local_initial_time: float = 0.0

    def structural_identity(self):
        """Return a history-independent identity for an ARG construction state.

        Policy caches and transition records are deliberately excluded.  The
        latter are an undo/debugging aid and must not make two biologically
        identical local states different GFlowNet states.
        """

        nodes = tuple(
            (
                int(node_id),
                tuple(sorted(int(value) for value in lineage.children)),
                tuple(sorted(int(value) for value in lineage.parents)),
                tuple(
                    (int(left), int(right))
                    for left, right in lineage.material_segments.segments
                ),
                tuple(int(value) for value in lineage.variant_indices),
                str(lineage.event_type),
                (
                    None
                    if lineage.breakpoint is None
                    else int(lineage.breakpoint)
                ),
                lineage.recombination_side,
                float(lineage.time),
            )
            for node_id, lineage in sorted(self.all_nodes.items())
        )
        return (
            self.local_context_id,
            (
                None
                if self.local_target_interval is None
                else tuple(
                    float(value) for value in self.local_target_interval
                )
            ),
            (
                None
                if self.local_cut_time is None
                else float(self.local_cut_time)
            ),
            float(self.current_time),
            tuple(
                int(lineage.node_id)
                for lineage in self.active_lineages
            ),
            nodes,
        )

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
            log_likelihood=self.log_likelihood,
            accumulated_log_prior=self.accumulated_log_prior,
            accumulated_log_likelihood=self.accumulated_log_likelihood,
            outside_log_likelihood=self.outside_log_likelihood,
            partial_log_reward=self.partial_log_reward,
            terminal_partial_correction=self.terminal_partial_correction,
            is_done=self.is_done,
            total_active_blocks=self.total_active_blocks,
            current_time=float(self.current_time),
            target_material=self.target_material,
            block_boundaries=self.block_boundaries,
            time_scale=float(self.time_scale),
            generated_node_start=self.generated_node_start,
            transition_records=copy.deepcopy(self.transition_records),
            fixed_ancestor_schedule=copy.deepcopy(
                self.fixed_ancestor_schedule
            ),
            target_variant_indices=tuple(self.target_variant_indices),
            variant_block_indices=dict(self.variant_block_indices),
            local_breakpoint_weights=dict(self.local_breakpoint_weights),
            vcf_alignment=copy.deepcopy(self.vcf_alignment),
            likelihood_scope=str(self.likelihood_scope),
            local_context_id=self.local_context_id,
            local_target_interval=self.local_target_interval,
            local_cut_time=self.local_cut_time,
            local_initial_time=float(self.local_initial_time),
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
        if action.delta_time is not None:
            result["delta_time"] = float(action.delta_time)
        return result
    if isinstance(action, FixedAttachmentChoice):
        return {
            "event_type": "fixed_attachment",
            "event_time": float(action.event_time),
        }
    raise ValueError(f"Unknown ARG action: {action}")


@dataclass(frozen=True)
class FixedAttachmentChoice:
    """Deterministic installation of the next fixed local ancestor group."""

    event_time: float

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
        variant_data: Optional[Any] = None,
        seed: Optional[int] = 7,
        bp_per_blocks: int = 1,
        device: Optional[torch.device] = 'cpu',
        time_bins: Optional[int] = None,
        time_delta_bin_width: Optional[float] = None,
        reward_C: float = 30000,
        structural_only: bool = False,
    ):
        if sequences is not None and variant_data is not None:
            raise ValueError("Pass either dense sequences or variant_data, not both")
        self.structural_only = bool(structural_only)
        self.input_mode = (
            "vcf"
            if variant_data is not None
            else ("structural" if self.structural_only else "dense")
        )
        self.variant_data = variant_data
        self.sequences = list(sequences) if sequences is not None else None
        self.chars_dict = CHARACTERS_MAPS['DNA_WITH_GAP']
        self.event_types = ["coal", "recomb"]
        self.device = torch.device(device)

        if self.input_mode == "vcf":
            num_sequences = int(variant_data.num_haplotypes)
            sequence_length = int(variant_data.sequence_length)
            inferred_num_blocks = int(variant_data.num_variants)
            if num_blocks is not None and int(num_blocks) != inferred_num_blocks:
                raise ValueError(
                    "VCF mode requires num_blocks to match the retained variant count "
                    f"({inferred_num_blocks}), got {num_blocks}"
                )
            num_blocks = inferred_num_blocks
        elif self.sequences is not None:
            num_sequences = len(self.sequences)
            sequence_length = len(self.sequences[0])
            if any(len(sequence) != sequence_length for sequence in self.sequences):
                raise ValueError("all sequences must have length sequence_length")


        if num_sequences is None or sequence_length is None:
            raise ValueError(
                "num_sequences and sequence_length are required when no "
                "sequence or variant data is supplied"
            )
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
        if time_delta_bin_width is not None:
            time_env_kwargs["delta_bin_width"] = float(time_delta_bin_width)
        self.time_env = TimeEnvFixedDelta(**time_env_kwargs)

        self.rng = random.Random(seed)

        ## Sequence or sparse-variant arrays
        self.block_indices = np.arange(self.num_blocks)

        if self.input_mode == "vcf":
            self.sample_ids = list(variant_data.sample_ids)
            self.haplotype_ids = list(variant_data.haplotype_ids)
            self.variant_positions0 = np.asarray(variant_data.positions0, dtype=np.int64)
            self.variant_boundaries = self._build_variant_boundaries(self.variant_positions0)
            self.variant_gap_lengths = self._build_variant_gap_lengths(self.variant_positions0)
            if not self.structural_only:
                variant_partials = np.asarray(
                    variant_data.haplotype_partials,
                    dtype=np.float32,
                )
                expected_shape = (self.num_sequences, self.num_blocks, 4)
                if tuple(variant_partials.shape) != expected_shape:
                    raise ValueError(
                        "variant partials must have shape "
                        f"{expected_shape}, got {variant_partials.shape}"
                    )
                self.block_seq_arrays = torch.nn.Parameter(
                    torch.tensor(
                        variant_partials,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    requires_grad=False,
                )
                self.variant_position_tensor = torch.nn.Parameter(
                    torch.tensor(
                        self.variant_positions0,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    requires_grad=False,
                )
                self.variant_boundary_tensor = torch.nn.Parameter(
                    torch.tensor(
                        self.variant_boundaries,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    requires_grad=False,
                )
                self.variant_prev_gap_tensor = torch.nn.Parameter(
                    torch.tensor(
                        self.variant_gap_lengths["prev"],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    requires_grad=False,
                )
                self.variant_next_gap_tensor = torch.nn.Parameter(
                    torch.tensor(
                        self.variant_gap_lengths["next"],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    requires_grad=False,
                )
            else:
                self.block_seq_arrays = None
                self.variant_position_tensor = None
                self.variant_boundary_tensor = None
                self.variant_prev_gap_tensor = None
                self.variant_next_gap_tensor = None
        elif self.structural_only:
            self.seq_arrays = None
            self.block_seq_arrays = None
        else:
            if self.sequences is None:
                raise ValueError("dense mode requires sequences")
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
        self.evolution_model = (
            None if self.structural_only else EvolutionModelTorch(self)
        )

        ## Reward function 
        self.reward_fn = ARGReward(C=reward_C)

    @property
    def time_metadata(self):
        return {
            "time_bin_scheme": type(self.time_env).__name__,
            "time_bins": int(self.time_env.bins),
            "time_delta_bin_width": float(self.time_env.delta_bin_width),
        }

    @property
    def is_vcf_mode(self):
        return self.input_mode == "vcf"

    def seq2array(self, seq):
        seq = [self.chars_dict[x] for x in seq]
        data = np.array(seq)
        return data

    def _build_variant_boundaries(self, positions0):
        positions0 = np.asarray(positions0, dtype=np.float64)
        if positions0.ndim != 1 or positions0.size != int(self.num_blocks):
            raise ValueError("VCF positions must be a 1D array with num_blocks entries")
        if positions0.size == 0:
            raise ValueError("VCF mode requires at least one retained variant")
        if np.any(np.diff(positions0) <= 0):
            raise ValueError("VCF positions must be strictly increasing")
        boundaries = np.empty(positions0.size + 1, dtype=np.float64)
        boundaries[0] = 0.0
        boundaries[-1] = float(self.sequence_length)
        if positions0.size > 1:
            boundaries[1:-1] = (positions0[:-1] + positions0[1:]) / 2.0
        return boundaries

    def _build_variant_gap_lengths(self, positions0):
        positions0 = np.asarray(positions0, dtype=np.float64)
        prev_gaps = np.zeros_like(positions0, dtype=np.float32)
        next_gaps = np.zeros_like(positions0, dtype=np.float32)
        if positions0.size > 1:
            diffs = np.diff(positions0).astype(np.float32)
            prev_gaps[1:] = diffs
            next_gaps[:-1] = diffs
        return {"prev": prev_gaps, "next": next_gaps}

    def _lineage_block_indices(self, material_segments):
        return np.asarray(material_segments.to_block_list(), dtype=np.int64)

    def _select_compact_partials(self, partials, source_segments, target_segments):
        if not self.is_vcf_mode:
            return partials
        source_blocks = source_segments.to_block_tensor(partials.device)
        target_blocks = target_segments.to_block_tensor(partials.device)
        if target_blocks.numel() == 0:
            return partials.new_zeros((0, partials.shape[-1]))
        if source_blocks.numel() == 0:
            raise ValueError("Cannot select VCF partial rows from an empty source lineage")
        positions = torch.searchsorted(source_blocks, target_blocks)
        safe_positions = positions.clamp(max=source_blocks.numel() - 1)
        valid = (
            (positions >= 0)
            & (positions < source_blocks.numel())
            & (source_blocks.index_select(0, safe_positions) == target_blocks)
        )
        if not bool(valid.all().detach().cpu().item()):
            raise ValueError("target material segments are not contained in source material segments")
        return partials.index_select(0, positions)

    def _breakpoint_gap_length(self, breakpoint):
        if not self.is_vcf_mode:
            return 1.0
        breakpoint = int(breakpoint)
        if not (1 <= breakpoint < int(self.num_blocks)):
            return 0.0
        return max(
            float(self.variant_positions0[breakpoint] - self.variant_positions0[breakpoint - 1]),
            1.0,
        )

    def _recombination_breakpoint_weights(self, choice):
        breakpoints = range(int(choice.span_start) + 1, int(choice.span_end) + 1)
        weights = []
        for breakpoint in breakpoints:
            left_segments, right_segments = self._choice_lineage_segments(choice).split(breakpoint)
            if left_segments.count <= 0 or right_segments.count <= 0:
                continue
            weights.append((int(breakpoint), self._breakpoint_gap_length(breakpoint)))
        return weights

    def _choice_lineage_segments(self, choice):
        return MaterialSegments(((int(choice.span_start), int(choice.span_end) + 1),))

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
        partials_list = (
            [None] * self.num_sequences
            if self.structural_only
            else self._initial_lineages_partials_batch(material_segments_list)
        )

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
            partial_log_reward=0.0,
            terminal_partial_correction=0.0,
            is_done=False,
            total_active_blocks=self.num_sequences * self.num_blocks,
            current_time=0.0,
        )
        state.is_done = self.is_terminal(state)
        if state.is_done and not self.structural_only:
            log_likelihood = self.evolution_model.compute_arg_log_likelihood(state)
            state.log_reward = self.compute_terminal_log_reward(state, log_likelihood)
            state.terminal_partial_correction = float(state.log_reward - state.partial_log_reward)
            state.partial_log_reward = float(state.log_reward)
        return state

    def _initial_lineage_partials(self, node_id, material_segments):
        partials = self.block_seq_arrays[int(node_id)].detach().clone().float()
        if self.is_vcf_mode:
            return self._select_compact_partials(
                partials,
                MaterialSegments.full(self.num_blocks),
                material_segments,
            )
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
        partials = lineage.partials
        if torch.is_tensor(partials):
            tensor = partials.to(device=self.device)
            if not tensor.is_floating_point():
                tensor = tensor.to(dtype=torch.float32)
        else:
            tensor = torch.as_tensor(partials, device=self.device, dtype=torch.float32)
        expected_rows = (
            int(len(lineage.variant_indices))
            if self.is_vcf_mode
            else int(self.num_blocks)
        )
        expected_shape = (expected_rows, 4)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"ARG lineage {lineage.node_id} partials must have shape "
                f"{expected_shape}, got {tuple(tensor.shape)}"
            )
        return tensor

    def _transition_lineage_partials(self, lineage, parent_time):
        edge_time = float(parent_time) - float(lineage.time)
        if edge_time <= 0:
            raise ValueError(
                f"ARG node times must increase from child to parent: "
                f"parent_time={parent_time}, child={lineage.node_id} time={lineage.time}"
            )
        partials = self._require_lineage_partials(lineage)
        return self.evolution_model.transition_partials(partials, edge_time)

    def normalize_partials_with_log_scale(self, partials, carried_rows=None):
        row_sums = partials.sum(dim=-1, keepdim=True)
        normalized = torch.where(
            row_sums > 0,
            partials / row_sums.clamp_min(1e-12),
            torch.zeros_like(partials),
        )
        if carried_rows is None:
            carried = (row_sums.squeeze(-1) > 0)
        else:
            carried = torch.as_tensor(
                carried_rows,
                dtype=torch.bool,
                device=partials.device,
            )
        valid = carried & (row_sums.squeeze(-1) > 0)
        if not bool(valid.any().detach().cpu().item()):
            return normalized, 0.0
        log_scale = torch.log(row_sums.squeeze(-1)[valid]).sum()
        return normalized, float(log_scale.detach().cpu().item())

    def _coalesced_parent_partials(self, child_i, child_j, parent_segments, parent_time):
        if self.is_vcf_mode:
            return self._coalesced_parent_partials_sparse(
                child_i,
                child_j,
                parent_segments,
                parent_time,
            )

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
        carried_rows = has_material.squeeze(-1) & (
            self.evolution_model.material_site_weights(
                parent_segments,
                device=combined.device,
                dtype=combined.dtype,
            ) > 0
        )
        return self.normalize_partials_with_log_scale(combined, carried_rows)

    def _recombined_parent_partials(self, child, parent_segments, parent_time):
        transitioned = self._transition_lineage_partials(child, parent_time)
        if self.is_vcf_mode:
            selected = self._select_compact_partials(
                transitioned,
                child.material_segments,
                parent_segments,
            )
            return self.evolution_model.normalize_partials(selected)
        masked = self.evolution_model.mask_partials(transitioned, parent_segments)
        return self.evolution_model.normalize_partials(masked)

    def _recombined_parent_partials_indexed(
        self,
        child,
        parent_segments,
        parent_time,
        parent_variant_indices,
    ):
        """Transition and select compact VCF rows by global variant ID."""

        if not self.is_vcf_mode:
            return self._recombined_parent_partials(
                child,
                parent_segments,
                parent_time,
            )
        transitioned = self._transition_lineage_partials(child, parent_time)
        child_indices = torch.as_tensor(
            child.variant_indices,
            dtype=torch.long,
            device=transitioned.device,
        )
        target_indices = torch.as_tensor(
            tuple(int(value) for value in parent_variant_indices),
            dtype=torch.long,
            device=transitioned.device,
        )
        if target_indices.numel() == 0:
            return transitioned.new_zeros((0, 4))
        positions = torch.searchsorted(child_indices, target_indices)
        safe_positions = positions.clamp(max=child_indices.numel() - 1)
        valid = (
            (positions >= 0)
            & (positions < child_indices.numel())
            & (
                child_indices.index_select(0, safe_positions)
                == target_indices
            )
        )
        if not bool(valid.all().detach().cpu().item()):
            raise ValueError(
                "recombination target VCF rows are not contained in the child"
            )
        selected = transitioned.index_select(0, positions)
        return self.evolution_model.normalize_partials(selected)

    def _coalesced_parent_partials_sparse(self, child_i, child_j, parent_segments, parent_time):
        parent_blocks = torch.as_tensor(
            sorted(
                set(int(value) for value in child_i.variant_indices)
                | set(int(value) for value in child_j.variant_indices)
            ),
            dtype=torch.long,
            device=self.device,
        )
        reference_partials = self._require_lineage_partials(child_i)
        combined = torch.ones(
            (parent_blocks.numel(), 4),
            dtype=reference_partials.dtype,
            device=self.device,
        )
        has_material = torch.zeros(
            parent_blocks.numel(),
            dtype=torch.bool,
            device=self.device,
        )

        for child in (child_i, child_j):
            child_partials = self.evolution_model.normalize_partials(
                self._transition_lineage_partials(child, parent_time)
            )
            child_blocks = torch.as_tensor(
                child.variant_indices,
                dtype=torch.long,
                device=self.device,
            )
            if child_blocks.numel() == 0:
                continue
            parent_positions = torch.searchsorted(parent_blocks, child_blocks)
            safe_positions = parent_positions.clamp(max=parent_blocks.numel() - 1)
            valid = (
                (parent_positions >= 0)
                & (parent_positions < parent_blocks.numel())
                & (parent_blocks.index_select(0, safe_positions) == child_blocks)
            )
            if not bool(valid.all().detach().cpu().item()):
                raise ValueError("child material segments are not contained in coalesced parent segments")
            combined[parent_positions] = combined[parent_positions] * child_partials
            has_material[parent_positions] = True

        combined = torch.where(has_material[:, None], combined, torch.zeros_like(combined))
        return self.normalize_partials_with_log_scale(combined, has_material)

    def _lineage_variants_for_material(
        self,
        state,
        lineage,
        material_segments,
    ):
        if not self.is_vcf_mode:
            return ()
        if state.variant_block_indices:
            return tuple(
                int(variant_index)
                for variant_index in lineage.variant_indices
                if any(
                    int(start)
                    <= int(state.variant_block_indices[int(variant_index)])
                    < int(end)
                    for start, end in material_segments.segments
                )
            )
        return tuple(
            int(variant_index)
            for variant_index in lineage.variant_indices
            if any(
                int(start) <= int(variant_index) < int(end)
                for start, end in material_segments.segments
            )
        )

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
        if self.is_vcf_mode:
            block_index = min(max(int(block_index), 0), len(self.variant_boundaries) - 1)
            return float(self.variant_boundaries[block_index])
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
        else:
            result = int(state.total_active_blocks) == self.num_blocks
            # bool(np.all(self.get_active_counts(state) == 1)) ## another way, realtime compute. 
            return result

    def _finalize_transition_state(self, next_state, log_prior, partial_log_likelihood_increment=0.0):
        if log_prior is not None:
            next_state.accumulated_log_prior += log_prior
            next_state.partial_log_reward += log_prior
        next_state.accumulated_log_likelihood += float(
            partial_log_likelihood_increment
        )
        next_state.partial_log_reward += float(partial_log_likelihood_increment)
        if self.structural_only or next_state.target_material is not None:
            next_state.is_done = False
            next_state.log_reward = None
            next_state.log_likelihood = None
            next_state.terminal_partial_correction = 0.0
            next_state.rates = None
            next_state.prior_options = None
            return next_state
        next_state.is_done = self.is_terminal(next_state)
        if next_state.is_done:
            log_likelihood = self.evolution_model.compute_arg_log_likelihood(next_state)
            next_state.log_likelihood = float(log_likelihood)
            next_state.log_reward = self.compute_terminal_log_reward(next_state, log_likelihood)
            next_state.terminal_partial_correction = float(
                next_state.log_reward - next_state.partial_log_reward
            )
            next_state.partial_log_reward = float(next_state.log_reward)
        else:
            next_state.log_reward = None
            next_state.log_likelihood = None
            next_state.terminal_partial_correction = 0.0
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
        delta_t = (
            float(action.delta_time)
            if action.delta_time is not None
            else self.time_env.time_action_to_delta(
                action.time_action,
                self._total_event_rate(rates),
            )
        )
        parent_time = float(state.current_time) + delta_t
        next_state.current_time = parent_time
        if self.structural_only:
            parent_partials = None
            partial_log_likelihood_increment = 0.0
        else:
            (
                parent_partials,
                partial_log_likelihood_increment,
            ) = self._coalesced_parent_partials(
                child_i,
                child_j,
                parent_segments,
                parent_time,
            )
        parent_variant_indices = (
            tuple(
                sorted(
                    set(int(value) for value in child_i.variant_indices)
                    | set(int(value) for value in child_j.variant_indices)
                )
            )
            if self.is_vcf_mode
            else ()
        )
        parent = ARGLineage(
            node_id=parent_id,
            children=[child_i.node_id, child_j.node_id],
            parents=[],
            material_segments=parent_segments,
            num_blocks=max(child_i.num_blocks, child_j.num_blocks),
            partials=parent_partials,
            variant_indices=parent_variant_indices,
            sequences_indices=sorted(set(child_i.sequences_indices + child_j.sequences_indices)),
            event_type="coal",
            time=parent_time,
        )

        child_i.parents.append(parent.node_id)
        child_j.parents.append(parent.node_id)
        if self.structural_only:
            child_i.partials = None
            child_j.partials = None
        child_i.clear_runtime_caches()
        child_j.clear_runtime_caches()
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
        return self._finalize_transition_state(
            next_state,
            log_prior,
            partial_log_likelihood_increment=partial_log_likelihood_increment,
        )

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
        delta_t = (
            float(action.delta_time)
            if action.delta_time is not None
            else self.time_env.time_action_to_delta(
                action.time_action,
                self._total_event_rate(rates),
            )
        )

        event_time = float(state.current_time) + delta_t
        next_state.current_time = event_time
        if self.structural_only:
            left_partials = None
            right_partials = None
            left_variant_indices = ()
            right_variant_indices = ()
        else:
            left_variant_indices = self._lineage_variants_for_material(
                state,
                child,
                left_segments,
            )
            right_variant_indices = self._lineage_variants_for_material(
                state,
                child,
                right_segments,
            )
            left_partials = self._recombined_parent_partials_indexed(
                child,
                left_segments,
                event_time,
                left_variant_indices,
            )
            right_partials = self._recombined_parent_partials_indexed(
                child,
                right_segments,
                event_time,
                right_variant_indices,
            )
        left_parent = ARGLineage(
            node_id=left_parent_id,
            children=[child.node_id],
            parents=[],
            material_segments=left_segments,
            num_blocks=child.num_blocks,
            partials=left_partials,
            variant_indices=left_variant_indices,
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
            num_blocks=child.num_blocks,
            partials=right_partials,
            variant_indices=right_variant_indices,
            sequences_indices=list(child.sequences_indices),
            event_type="recomb",
            breakpoint=breakpoint,
            recombination_side="right",
            time=event_time,
        )

        child.parents = [left_parent.node_id, right_parent.node_id]
        if self.structural_only:
            child.partials = None
        child.clear_runtime_caches()
        next_state.all_nodes[child.node_id] = child
        next_state.all_nodes[left_parent.node_id] = left_parent
        next_state.all_nodes[right_parent.node_id] = right_parent
        next_state.active_lineages = [
            lineage for idx, lineage in enumerate(next_state.active_lineages) if idx != current_lineage_idx
        ]
        next_state.active_lineages.extend([left_parent, right_parent])
        next_state.max_node_idx = right_parent.node_id
        return self._finalize_transition_state(
            next_state,
            log_prior,
            partial_log_likelihood_increment=0.0,
        )

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

        total_recomb_weight = sum(self._recomb_weight(choice) for choice in recomb_actions)
        normalizer = float(self.sequence_length if self.is_vcf_mode else self.num_blocks)
        total_active_material_length = float(total_recomb_weight) / max(normalizer, 1.0)
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

    def enumerate_actions(self, state, action_filter=None):

        coal_actions = self.compute_coalescence_actions(state)
        recomb_actions = self.compute_recombination_actions(state)
        if action_filter is not None:
            coal_actions, recomb_actions = action_filter(
                state,
                coal_actions,
                recomb_actions,
            )

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
        if action is None:
            action = combined_actions
            combined_actions = self.enumerate_actions(state)
        coal_actions, recomb_actions = combined_actions

        if rates is None:
            rates = state.rates if state.rates is not None else self.compute_event_rates((coal_actions, recomb_actions))
        state.rates = rates
        
        total_rate = self._total_event_rate(rates)
        recomb_total_weight = sum(self._recomb_weight(choice) for choice in recomb_actions)

        wait_log_prior = self.time_env.time_action_log_probability(action.time_action, total_rate)

        if isinstance(action, CoalescenceChoice) and CoalescenceChoice.is_valid_for(action, state.active_lineages):
            action_log_prior = math.log((rates["lambda_coal"] / total_rate) / len(coal_actions))
            
        elif isinstance(action, RecombinationChoice) and RecombinationChoice.is_valid_for(action, state.active_lineages):
            breakpoint_probability = self._breakpoint_prior_probability(action)
            action_log_prior = math.log(
                (rates["lambda_recomb"] / total_rate)
                * (self._recomb_weight(action) / recomb_total_weight)
                * breakpoint_probability
            )
        else:
            raise ValueError(f"Invalid action: {action}")

        return action_log_prior + wait_log_prior

    def prepare_state_rollout_inputs(
        self,
        states,
        random_spec=None,
        action_filter=None,
    ):
        batch_size = len(states)
        if batch_size == 0:
            raise ValueError("states must contain at least one ARGState")

        event = {}
        input_actions = []
        for idx, state in enumerate(states):
            coal_actions, recomb_actions = self.enumerate_actions(
                state,
                action_filter=action_filter,
            )
            if not coal_actions and not recomb_actions:
                raise ValueError(
                    "No available ARG actions for non-terminal state; "
                    "local refinement action filter may be too restrictive."
                )
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

        input_dict = {
            "states": states,
            "event": event,
            "input_actions": input_actions,
            "random_spec": random_spec,
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
            breakpoint = self._sample_breakpoint_from_choice(selected)
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
            if self.is_vcf_mode:
                return sum(weight for _, weight in self._recombination_breakpoint_weights(recomb_weight))
            return recomb_weight.material_count
        return recomb_weight[1]

    def _sample_breakpoint_from_choice(self, choice):
        weighted_breakpoints = self._recombination_breakpoint_weights(choice)
        if not weighted_breakpoints:
            raise ValueError("No valid recombination breakpoints to sample")
        total = sum(weight for _, weight in weighted_breakpoints)
        target = self.rng.random() * total
        cumulative = 0.0
        for breakpoint, weight in weighted_breakpoints:
            cumulative += weight
            if target <= cumulative:
                return int(breakpoint)
        return int(weighted_breakpoints[-1][0])

    def _breakpoint_prior_probability(self, action):
        weighted_breakpoints = self._recombination_breakpoint_weights(action)
        if not weighted_breakpoints:
            raise ValueError("Recombination action has no valid breakpoints")
        total = sum(weight for _, weight in weighted_breakpoints)
        if total <= 0:
            raise ValueError("Recombination breakpoint prior weight must be positive")
        selected = int(action.breakpoint)
        for breakpoint, weight in weighted_breakpoints:
            if int(breakpoint) == selected:
                return float(weight) / float(total)
        raise ValueError(f"Breakpoint {selected} is not valid for action {action}")

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


class LocalARGEnvironment:
    """Context-bound local ARG environment used by GFlowNet rollouts.

    The biological implementation lives in the production ``refinement``
    package copied from the verified ``new_rl`` workflow.  This adapter owns
    one or more prepared interval/time contexts and exposes the same rollout
    surface as :class:`SimpleARGEnvironment`.
    """

    _DELEGATED_MUTABLE_ATTRIBUTES = {
        "device",
        "seq_arrays",
        "block_seq_arrays",
        "variant_position_tensor",
        "variant_boundary_tensor",
        "variant_prev_gap_tensor",
        "variant_next_gap_tensor",
    }

    def __init__(self, base_env, prepared_contexts, context_ids=None):
        if not isinstance(base_env, SimpleARGEnvironment):
            raise TypeError("base_env must be a SimpleARGEnvironment")
        if base_env.structural_only or not base_env.is_vcf_mode:
            raise ValueError(
                "LocalARGEnvironment requires a likelihood-enabled phased-VCF "
                "SimpleARGEnvironment"
            )
        object.__setattr__(self, "base_env", base_env)

        if isinstance(prepared_contexts, dict):
            contexts = {
                str(context_id): prepared
                for context_id, prepared in prepared_contexts.items()
            }
        else:
            prepared_contexts = list(prepared_contexts)
            if context_ids is None:
                context_ids = [
                    f"region_{index + 1:06d}"
                    for index in range(len(prepared_contexts))
                ]
            if len(context_ids) != len(prepared_contexts):
                raise ValueError(
                    "context_ids length must match prepared_contexts length"
                )
            contexts = {
                str(context_id): prepared
                for context_id, prepared in zip(
                    context_ids,
                    prepared_contexts,
                )
            }
        if not contexts:
            raise ValueError("LocalARGEnvironment requires at least one context")
        for context_id, prepared in contexts.items():
            if not prepared.context.is_valid:
                reasons = "; ".join(
                    diagnostic.message
                    for diagnostic in prepared.context.rejection_diagnostics
                )
                raise ValueError(
                    f"local context {context_id!r} is invalid: {reasons}"
                )
        object.__setattr__(self, "prepared_contexts", contexts)
        object.__setattr__(self, "context_ids", tuple(contexts))
        object.__setattr__(self, "is_local", True)

    def __getattr__(self, name):
        return getattr(self.base_env, name)

    def __setattr__(self, name, value):
        if (
            name in self._DELEGATED_MUTABLE_ATTRIBUTES
            and "base_env" in self.__dict__
        ):
            setattr(self.base_env, name, value)
            return
        object.__setattr__(self, name, value)

    @property
    def time_metadata(self):
        return self.base_env.time_metadata

    def context_for_state(self, state):
        context_id = getattr(state, "local_context_id", None)
        if context_id not in self.prepared_contexts:
            raise ValueError(
                f"ARGState has unknown local_context_id {context_id!r}"
            )
        return self.prepared_contexts[context_id]

    def get_initial_state(self, context_id=None):
        try:
            from .refinement.local_construction import initialize_local_arg_state
        except ImportError:
            from refinement.local_construction import initialize_local_arg_state

        if context_id is None:
            if len(self.context_ids) != 1:
                raise ValueError(
                    "context_id is required when the local environment owns "
                    "multiple refinement contexts"
                )
            context_id = self.context_ids[0]
        context_id = str(context_id)
        if context_id not in self.prepared_contexts:
            raise ValueError(f"unknown local context {context_id!r}")
        prepared = self.prepared_contexts[context_id]
        state = initialize_local_arg_state(prepared, self.base_env)
        state.local_context_id = context_id
        state.local_target_interval = tuple(
            float(value)
            for value in prepared.context.request.genomic_range
        )
        state.local_cut_time = float(state.current_time)
        state.local_initial_time = float(state.current_time)
        return state

    def initial_states(self):
        return [
            self.get_initial_state(context_id)
            for context_id in self.context_ids
        ]

    def clone_state_to_device(self, state):
        cloned = state.clone(copy_partials=False)
        for lineage in cloned.all_nodes.values():
            if lineage.partials is not None:
                lineage.partials = torch.as_tensor(
                    lineage.partials,
                    device=self.device,
                )
        return cloned

    def enumerate_actions(self, state, action_filter=None):
        try:
            from .refinement.local_construction import enumerate_local_prior_actions
        except ImportError:
            from refinement.local_construction import enumerate_local_prior_actions

        prepared = self.context_for_state(state)
        options = enumerate_local_prior_actions(
            state,
            prepared.context,
            self.base_env,
        )
        coal = list(options.coal_actions)
        recomb = list(options.recomb_choices)
        if action_filter is not None:
            coal, recomb = action_filter(state, coal, recomb)
        return coal, recomb

    def enumerate_prior_options(self, state):
        try:
            from .refinement.local_construction import enumerate_local_prior_actions
        except ImportError:
            from refinement.local_construction import enumerate_local_prior_actions

        prepared = self.context_for_state(state)
        return enumerate_local_prior_actions(
            state,
            prepared.context,
            self.base_env,
        )

    def compute_event_rates(self, actions, state=None):
        if state is None:
            return self.base_env.compute_event_rates(actions)
        options = self.enumerate_prior_options(state)
        return dict(options.rates)

    def compute_event_probabilities(self, state, actions=None):
        options = self.enumerate_prior_options(state)
        rates = options.rates
        total_rate = float(
            rates["lambda_coal"] + rates["lambda_recomb"]
        )
        if total_rate <= 0.0:
            return {"coal": 0.0, "recomb": 0.0}
        return {
            "coal": float(rates["lambda_coal"]) / total_rate,
            "recomb": float(rates["lambda_recomb"]) / total_rate,
        }

    def next_fixed_ancestor_time(self, state):
        try:
            from .refinement import local_construction as local
        except ImportError:
            from refinement import local_construction as local

        return local._next_fixed_ancestor_time(state)

    def valid_breakpoints(self, state, action):
        try:
            from .refinement import local_construction as local
        except ImportError:
            from refinement import local_construction as local

        if not isinstance(action, RecombinationChoice):
            return ()
        lineage = state.active_lineages[int(action.active_lineage_i)]
        if self.is_vcf_mode:
            return tuple(
                int(breakpoint)
                for breakpoint, _weight in local._vcf_breakpoints(
                    lineage.material_segments,
                    state,
                    self.base_env,
                )
            )
        return tuple(
            range(
                int(action.span_start) + 1,
                int(action.span_end) + 1,
            )
        )

    def _rollout_time_data(self, state, rates):
        total_rate = float(
            rates["lambda_coal"] + rates["lambda_recomb"]
        )
        next_fixed = self.next_fixed_ancestor_time(state)
        max_delta = (
            None
            if next_fixed is None
            else max(0.0, float(next_fixed) - float(state.current_time))
        )
        if total_rate <= 0.0:
            mask = [False] * int(self.time_env.bins)
        else:
            mask = [
                math.isfinite(
                    self.time_env.time_action_log_probability(
                        action,
                        total_rate,
                        max_delta=max_delta,
                    )
                )
                for action in range(int(self.time_env.bins))
            ]
        return {
            "total_rate": total_rate,
            "next_fixed_time": next_fixed,
            "max_delta": max_delta,
            "time_action_mask": tuple(mask),
            "can_generate": bool(total_rate > 0.0 and any(mask)),
            "can_attach_fixed": next_fixed is not None,
        }

    def prepare_state_rollout_inputs(
        self,
        states,
        random_spec=None,
        action_filter=None,
    ):
        if not states:
            raise ValueError("states must contain at least one ARGState")
        actions = []
        rollout = []
        for state in states:
            coal, recomb = self.enumerate_actions(
                state,
                action_filter=action_filter,
            )
            options = state.prior_options
            if options is None:
                options = self.enumerate_prior_options(state)
            decision = self._rollout_time_data(state, options.rates)
            if not decision["can_generate"] and not decision["can_attach_fixed"]:
                raise ValueError(
                    "nonterminal local state has no legal generated event or "
                    "scheduled fixed attachment"
                )
            candidate_actions = []
            if float(options.rates["lambda_coal"]) > 0.0:
                candidate_actions.extend(coal)
            if float(options.rates["lambda_recomb"]) > 0.0:
                candidate_actions.extend(recomb)
            actions.append(candidate_actions)
            rollout.append(decision)
        return {
            "states": states,
            "input_actions": actions,
            "random_spec": random_spec,
            "local_mode": True,
            "rollout": rollout,
        }

    def time_action_to_delta(self, state, time_action):
        options = self.enumerate_prior_options(state)
        rollout = self._rollout_time_data(state, options.rates)
        return self.time_env.time_action_to_delta(
            int(time_action),
            rollout["total_rate"],
            max_delta=rollout["max_delta"],
        )

    def compute_cwr_event_log_prior(
        self,
        state,
        combined_actions,
        action=None,
        rates=None,
    ):
        try:
            from .refinement import local_construction as local
        except ImportError:
            from refinement import local_construction as local

        if action is None:
            action = combined_actions
            combined_actions = self.enumerate_actions(state)
        coal_actions, recomb_actions = combined_actions
        if rates is None:
            rates = self.enumerate_prior_options(state).rates
        total_rate = float(
            rates["lambda_coal"] + rates["lambda_recomb"]
        )
        if total_rate <= 0.0:
            raise ValueError("generated local events require a positive rate")
        rollout = self._rollout_time_data(state, rates)
        wait_log_prior = self.time_env.time_action_log_probability(
            action.time_action,
            total_rate,
            max_delta=rollout["max_delta"],
        )
        if isinstance(action, CoalescenceChoice):
            if not action.is_valid_for(state.active_lineages):
                raise ValueError(f"invalid local coalescence action: {action}")
            action_log_prior = (
                math.log(float(rates["lambda_coal"]) / total_rate)
                - math.log(len(coal_actions))
            )
        elif isinstance(action, RecombinationChoice):
            lineage = state.active_lineages[int(action.active_lineage_i)]
            recomb_weights = [
                local._local_recombination_weight(
                    state.active_lineages[int(choice.active_lineage_i)],
                    state,
                    self.base_env,
                )
                for choice in recomb_actions
            ]
            total_recomb_weight = float(sum(recomb_weights))
            matching = [
                index
                for index, choice in enumerate(recomb_actions)
                if int(choice.active_lineage_i)
                == int(action.active_lineage_i)
            ]
            if not matching or total_recomb_weight <= 0.0:
                raise ValueError(f"invalid local recombination action: {action}")
            lineage_weight = float(recomb_weights[matching[0]])
            breakpoints = local._vcf_breakpoints(
                lineage.material_segments,
                state,
                self.base_env,
            )
            breakpoint_weight = next(
                (
                    float(weight)
                    for breakpoint, weight in breakpoints
                    if int(breakpoint) == int(action.breakpoint)
                ),
                0.0,
            )
            total_breakpoint_weight = float(
                sum(weight for _breakpoint, weight in breakpoints)
            )
            if breakpoint_weight <= 0.0 or total_breakpoint_weight <= 0.0:
                raise ValueError(
                    f"invalid local breakpoint {action.breakpoint}"
                )
            action_log_prior = (
                math.log(float(rates["lambda_recomb"]) / total_rate)
                + math.log(lineage_weight / total_recomb_weight)
                + math.log(
                    breakpoint_weight / total_breakpoint_weight
                )
            )
        else:
            raise ValueError(f"unsupported generated local action: {action}")
        return float(wait_log_prior + action_log_prior)

    def fixed_attachment_log_prior(self, state):
        rates = self.enumerate_prior_options(state).rates
        rollout = self._rollout_time_data(state, rates)
        if not rollout["can_attach_fixed"]:
            raise ValueError("state has no scheduled fixed attachment")
        if rollout["total_rate"] <= 0.0:
            return 0.0
        return float(
            -rollout["total_rate"] * float(rollout["max_delta"])
        )

    def apply_fixed_attachment(self, state, log_prior=None):
        try:
            from .refinement import local_construction as local
        except ImportError:
            from refinement import local_construction as local

        prepared = self.context_for_state(state)
        next_fixed_time = self.next_fixed_ancestor_time(state)
        if next_fixed_time is None:
            raise ValueError("state has no scheduled fixed attachment")
        if log_prior is None:
            log_prior = self.fixed_attachment_log_prior(state)
        next_state = state.clone(copy_partials=False)
        next_state.current_time = float(next_fixed_time)
        next_state.accumulated_log_prior += float(log_prior)
        next_state.partial_log_reward += float(log_prior)
        next_state = local.reveal_due_fixed_ancestors(
            next_state,
            prepared.context,
            next_fixed_time,
            env=self.base_env,
        )
        attachment_record = dict(next_state.transition_records[-1])
        attachment_record.update(
            {
                "waited_from_time": float(
                    state.current_time * state.time_scale
                ),
                "waited_from_scaled_time": float(state.current_time),
                "log_prior_increment": float(log_prior),
                "fixed_event_survival": True,
            }
        )
        undo = dict(attachment_record["_undo"])
        undo.update(
            {
                "previous_current_time": float(state.current_time),
                "previous_accumulated_log_prior": float(
                    state.accumulated_log_prior
                ),
                "previous_partial_log_reward": float(
                    state.partial_log_reward
                ),
                "previous_is_done": bool(state.is_done),
            }
        )
        attachment_record["_undo"] = undo
        next_state.transition_records[-1] = attachment_record
        next_state.is_done = local.local_is_terminal(
            next_state,
            prepared.context,
        )
        if next_state.is_done and next_state.likelihood_scope != "none":
            local._finalize_local_terminal_likelihood(
                next_state,
                prepared.context,
                self.base_env,
            )
        return next_state

    def apply_action(self, state, action, log_prior=None):
        try:
            from .refinement import local_construction as local
        except ImportError:
            from refinement import local_construction as local

        if isinstance(action, FixedAttachmentChoice):
            next_state = self.apply_fixed_attachment(
                state,
                log_prior=log_prior,
            )
        else:
            prepared = self.context_for_state(state)
            if log_prior is None:
                log_prior = self.compute_cwr_event_log_prior(
                    state,
                    self.enumerate_actions(state),
                    action,
                )
            next_state = local.apply_local_action(
                state,
                action,
                prepared.context,
                self.base_env,
                float(log_prior),
            )
        restored = self.undo_transition(next_state)
        if restored.structural_identity() != state.structural_identity():
            raise RuntimeError(
                "local transition failed its exact structural round-trip"
            )
        return next_state

    def is_terminal(self, state):
        try:
            from .refinement.local_construction import local_is_terminal
        except ImportError:
            from refinement.local_construction import local_is_terminal

        prepared = self.context_for_state(state)
        return local_is_terminal(state, prepared.context)

    def compute_terminal_log_likelihood(self, state):
        try:
            from .refinement.local_construction import (
                compute_local_terminal_log_likelihood,
            )
        except ImportError:
            from refinement.local_construction import (
                compute_local_terminal_log_likelihood,
            )

        prepared = self.context_for_state(state)
        return compute_local_terminal_log_likelihood(
            state,
            prepared.context,
            self.base_env,
        )

    def compute_terminal_log_reward(self, state, log_likelihood=None):
        if not self.is_terminal(state):
            raise ValueError("local terminal reward requires a terminal state")
        if log_likelihood is None:
            log_likelihood = self.compute_terminal_log_likelihood(state)
        return float(
            self.reward_fn(
                float(log_likelihood),
                float(state.accumulated_log_prior),
            )
        )

    def state_to_proposal(self, state):
        try:
            from .refinement.local_construction import local_state_to_proposal
        except ImportError:
            from refinement.local_construction import local_state_to_proposal

        return local_state_to_proposal(
            state,
            self.context_for_state(state),
        )

    def undo_transition(self, state):
        try:
            from .refinement.local_construction import undo_local_transition
        except ImportError:
            from refinement.local_construction import undo_local_transition

        prepared = self.context_for_state(state)
        return undo_local_transition(state, prepared.context)

    def backward_parent_count(self, state):
        if len(state.transition_records) <= 1:
            return 0
        latest_record = state.transition_records[-1]
        if latest_record.get("event_type") == "fixed_attachment":
            return 1

        current_time = float(state.current_time)
        count = 0
        for lineage in state.active_lineages:
            if (
                lineage.event_type == "coal"
                and len(lineage.children) == 2
                and math.isclose(
                    float(lineage.time),
                    current_time,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                count += 1

        recombination_groups = {}
        for lineage in state.active_lineages:
            if (
                lineage.event_type == "recomb"
                and len(lineage.children) == 1
                and lineage.recombination_side in {"left", "right"}
                and math.isclose(
                    float(lineage.time),
                    current_time,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                key = (
                    int(lineage.children[0]),
                    int(lineage.breakpoint),
                )
                recombination_groups.setdefault(key, set()).add(
                    lineage.recombination_side
                )
        count += sum(
            sides == {"left", "right"}
            for sides in recombination_groups.values()
        )
        if count <= 0:
            raise RuntimeError(
                "local child state has no valid structural predecessor"
            )
        return int(count)

    def backward_log_probability(self, state):
        count = self.backward_parent_count(state)
        if count <= 0:
            raise ValueError("initial local state has no backward parent")
        return -math.log(float(count))

    def get_active_counts(self, state):
        block_count = max(
            len(state.block_boundaries or ()) - 1,
            0,
        )
        counts = np.zeros(block_count, dtype=int)
        for lineage in state.active_lineages:
            for start, end in lineage.material_segments.segments:
                counts[int(start):int(end)] += 1
        return counts
