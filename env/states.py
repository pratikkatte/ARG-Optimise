import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .actions import PriorActionOptions


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
        likelihood_partials=None,
        likelihood_log_increment: float = 0.0,
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
        self.likelihood_partials = likelihood_partials
        self.likelihood_log_increment = float(likelihood_log_increment)
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
            likelihood_partials=(self.likelihood_partials.clone()
                                 if copy_partials and self.likelihood_partials is not None
                                 else self.likelihood_partials),
            likelihood_log_increment=self.likelihood_log_increment,
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
    partial_log_likelihood: Optional[float] = None

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
            partial_log_likelihood=self.partial_log_likelihood,
        )
