import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from .actions import PriorActionOptions

if TYPE_CHECKING:
    from .action_context import ActionContext, CompatibilityCache


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

    def _iter_intersections(self, other, interval_start=None, interval_end=None):
        other = MaterialSegments.from_segments(other)
        if interval_start is not None:
            interval_start = int(interval_start)
        if interval_end is not None:
            interval_end = int(interval_end)
        i = j = 0
        while i < len(self.segments) and j < len(other.segments):
            left_start, left_end = self.segments[i]
            right_start, right_end = other.segments[j]
            start = max(left_start, right_start)
            end = min(left_end, right_end)
            if interval_start is not None:
                start = max(start, interval_start)
            if interval_end is not None:
                end = min(end, interval_end)
            if start < end:
                yield start, end
            if left_end <= right_end:
                i += 1
            else:
                j += 1

    def intersection(self, other):
        return MaterialSegments(self._iter_intersections(other))

    def intersection_count(self, other, interval_start=None, interval_end=None):
        return sum(end - start for start, end in
                   self._iter_intersections(other, interval_start, interval_end))

    def overlaps(self, other):
        return self.intersection_count(other) > 0

    def covers_interval(self, start, end):
        start = int(start)
        end = int(end)
        if start >= end:
            return False
        return any(seg_start <= start and end <= seg_end for seg_start, seg_end in self.segments)


@dataclass(frozen=True)
class DescendantSegments:
    """Disjoint physical intervals annotated with exact local sample bitsets."""
    segments: tuple = ()

    def __post_init__(self):
        result = []
        for left, right, bits in self.segments:
            if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
                   for x in (left, right, bits)) or left < 0 or right <= left or bits <= 0:
                raise ValueError("descendant intervals require integer 0 <= left < right and positive bitsets")
            left, right, bits = int(left), int(right), int(bits)
            if result and left < result[-1][1]:
                raise ValueError("descendant intervals must be sorted and disjoint")
            if result and left == result[-1][1] and bits == result[-1][2]:
                result[-1] = (result[-1][0], right, bits)
            else:
                result.append((left, right, bits))
        object.__setattr__(self, 'segments', tuple(result))

    @property
    def material(self):
        return MaterialSegments(tuple((l, r) for l, r, _ in self.segments))

    def at(self, position):
        return next((b for l, r, b in self.segments if l <= position < r), 0)

    def restrict(self, material):
        # Both inputs are sorted and disjoint: visit each interval once instead
        # of comparing every descendant interval with every material interval.
        result = []
        i = j = 0
        while i < len(self.segments) and j < len(material.segments):
            l, r, bits = self.segments[i]
            a, z = material.segments[j]
            left, right = max(l, a), min(r, z)
            if left < right:
                result.append((left, right, bits))
            if r <= z:
                i += 1
            else:
                j += 1
        return DescendantSegments(tuple(result))

    def merge(self, other):
        bounds = sorted({x for obj in (self, other) for l, r, _ in obj.segments for x in (l, r)})
        merged = []
        i = j = 0
        for l, r in zip(bounds, bounds[1:]):
            while i < len(self.segments) and self.segments[i][1] <= l:
                i += 1
            while j < len(other.segments) and other.segments[j][1] <= l:
                j += 1
            a = self.segments[i][2] if i < len(self.segments) and self.segments[i][0] <= l else 0
            b = other.segments[j][2] if j < len(other.segments) and other.segments[j][0] <= l else 0
            if a & b:
                raise ValueError("active local descendant sets must be disjoint")
            if a | b:
                merged.append((l, r, a | b))
        return DescendantSegments(tuple(merged))


@dataclass
class ARGLineage:
    node_id: int
    material_segments: MaterialSegments
    num_blocks: int
    descendants: Optional[DescendantSegments] = None
    children: list = field(default_factory=list)
    parents: list = field(default_factory=list)
    event_type: Optional[str] = None
    breakpoint: Optional[int] = None
    recombination_side: Optional[str] = None
    time: float = 0.0
    snp_indices: Optional[np.ndarray] = None
    messages: Optional[np.ndarray] = None
    exposure_increment: float = 0.0

    @property
    def material_mask(self):
        return self.material_segments.to_mask(self.num_blocks)

    @property
    def material_count(self):
        return self.material_segments.count

    @property
    def material_span(self):
        if self.material_count < 2:
            return None
        return (self.material_segments.span_start, self.material_segments.span_end, self.material_count)

    def clone(self, copy_partials=True, copy_mask=True):
        result = copy.copy(self)
        result.children, result.parents = list(self.children), list(self.parents)
        # Shared arrays are immutable. Rebuilt caches always replace them.
        if copy_partials:
            for name in ('snp_indices', 'messages'):
                value = getattr(self, name)
                if value is not None:
                    value = value.copy()
                    value.setflags(write=False)
                    setattr(result, name, value)
        return result


@dataclass
class ARGState:
    active_lineages: List[ARGLineage]
    all_nodes: Dict[int, ARGLineage]
    max_node_idx: int
    completed_site_lengths: np.ndarray
    dataset_fingerprint: str
    log_reward: Optional[float] = None
    accumulated_log_prior: float = 0.0
    is_done: bool = False
    rates: Optional[Dict[str, float]] = None
    prior_options: Optional[PriorActionOptions] = None
    total_active_blocks: int = 0
    current_time: float = 0.0
    partial_log_likelihood: float = 0.0
    exposure: float = 0.0  # internal 2Ne-time * bp, closed edges only
    actions: tuple = ()
    _action_context: Optional['ActionContext'] = field(default=None, init=False, repr=False, compare=False)
    _compatibility: Optional['CompatibilityCache'] = field(default=None, init=False, repr=False, compare=False)

    def clone(self, copy_partials=False):
        nodes = {key: node.clone(copy_partials=copy_partials) for key, node in self.all_nodes.items()}
        result = copy.copy(self)
        result.all_nodes = nodes
        result.active_lineages = [nodes[node.node_id] for node in self.active_lineages]
        result.completed_site_lengths = self.completed_site_lengths.copy()
        result.rates = result.prior_options = None
        result._action_context = None
        # The compatibility snapshot is immutable and can be reused by either
        # branch. Updating it replaces the snapshot, never its shared arrays.
        return result
