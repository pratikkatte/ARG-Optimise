"""Scalable temporal replay for full-ARG tskit tree sequences.

The trace keeps the source nodes, edges, and event schedule in column arrays.
Python graph dictionaries are only materialized for requested steps/windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import tskit
from numba import njit


RECOMBINATION_NODE_FLAG = 131072

EVENT_KIND_RECOMBINATION = 1
EVENT_KIND_COALESCENCE = 2
EVENT_KIND_UNARY = 3
EVENT_KIND_REVEAL = 4

EVENT_KIND_NAMES = {
    EVENT_KIND_RECOMBINATION: "recombination",
    EVENT_KIND_COALESCENCE: "coalescence",
    EVENT_KIND_UNARY: "unary",
    EVENT_KIND_REVEAL: "reveal",
}

_EVENT_KIND_PRIORITY = {
    EVENT_KIND_RECOMBINATION: 0,
    EVENT_KIND_COALESCENCE: 1,
    EVENT_KIND_UNARY: 2,
    EVENT_KIND_REVEAL: 3,
}


Segment = tuple[float, float]
ActiveSegmentMap = dict[int, tuple[Segment, ...]]


@dataclass(frozen=True)
class ActiveLineage:
    """A currently active ancestral lineage and its genomic material."""

    node_id: int
    segments: tuple[Segment, ...]

    @property
    def material_span(self) -> tuple[float, float] | None:
        if not self.segments:
            return None
        return self.segments[0][0], self.segments[-1][1]


@dataclass(frozen=True)
class ARGEvent:
    """A replay event reconstructed from a full-ARG tree sequence."""

    step: int
    kind: str
    time: float
    node_ids: tuple[int, ...]
    edge_ids: tuple[int, ...]


@dataclass(frozen=True)
class CompactActiveFrontier:
    """Compact active lineages with CSR-style genomic segment storage."""

    node_ids: np.ndarray
    segment_offsets: np.ndarray
    segment_left: np.ndarray
    segment_right: np.ndarray

    def __post_init__(self) -> None:
        node_ids = np.asarray(self.node_ids, dtype=np.int32)
        segment_offsets = np.asarray(self.segment_offsets, dtype=np.int64)
        segment_left = np.asarray(self.segment_left, dtype=np.float64)
        segment_right = np.asarray(self.segment_right, dtype=np.float64)
        if segment_offsets.shape != (node_ids.size + 1,):
            raise ValueError("segment_offsets must have one entry per lineage plus one")
        if segment_left.shape != segment_right.shape:
            raise ValueError("segment_left and segment_right must have matching shapes")
        if int(segment_offsets[-1]) != segment_left.size:
            raise ValueError("segment_offsets[-1] must equal the segment array length")
        for values in (node_ids, segment_offsets, segment_left, segment_right):
            values.setflags(write=False)
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "segment_offsets", segment_offsets)
        object.__setattr__(self, "segment_left", segment_left)
        object.__setattr__(self, "segment_right", segment_right)

    def __len__(self) -> int:
        return int(self.node_ids.size)

    @property
    def segment_count(self) -> int:
        return int(self.segment_left.size)

    def segments_for_index(self, lineage_index: int) -> tuple[Segment, ...]:
        lineage_index = int(lineage_index)
        if lineage_index < 0 or lineage_index >= len(self):
            raise IndexError("lineage_index out of range")
        start = int(self.segment_offsets[lineage_index])
        end = int(self.segment_offsets[lineage_index + 1])
        return tuple(
            (float(left), float(right))
            for left, right in zip(
                self.segment_left[start:end],
                self.segment_right[start:end],
            )
        )

    def segments_for_node(self, node_id: int) -> tuple[Segment, ...]:
        matches = np.flatnonzero(self.node_ids == int(node_id))
        if matches.size == 0:
            raise KeyError(int(node_id))
        return self.segments_for_index(int(matches[0]))

    def to_active_lineages(self) -> tuple[ActiveLineage, ...]:
        return tuple(
            ActiveLineage(
                node_id=int(node_id),
                segments=self.segments_for_index(lineage_index),
            )
            for lineage_index, node_id in enumerate(self.node_ids)
        )


@dataclass(frozen=True)
class TraceState:
    """ARG replay state after `step` events have been applied."""

    step: int
    current_time: float
    visible_node_ids: np.ndarray
    visible_edge_ids: np.ndarray
    _active_segments: Mapping[int, tuple[Segment, ...]] | None = field(repr=False)
    _compact_active: CompactActiveFrontier | None = field(
        default=None,
        repr=False,
    )

    @property
    def active_lineages(self) -> tuple[ActiveLineage, ...]:
        if self._compact_active is not None:
            return tuple(
                sorted(
                    self._compact_active.to_active_lineages(),
                    key=lambda lineage: lineage.node_id,
                )
            )
        if self._active_segments is None:
            raise RuntimeError(
                "active segments were not materialized for this state; call "
                "state_at_step(..., include_active=True)"
            )
        return tuple(
            ActiveLineage(node_id=node_id, segments=segments)
            for node_id, segments in sorted(self._active_segments.items())
            if segments
        )

    @property
    def compact_active_frontier(self) -> CompactActiveFrontier:
        if self._compact_active is not None:
            return self._compact_active
        if self._active_segments is None:
            raise RuntimeError(
                "active segments were not materialized for this state; call "
                "state_at_step(..., include_active=True)"
            )
        node_ids = np.asarray(sorted(self._active_segments), dtype=np.int32)
        offsets = np.empty(node_ids.size + 1, dtype=np.int64)
        offsets[0] = 0
        left: list[float] = []
        right: list[float] = []
        for lineage_index, node_id in enumerate(node_ids):
            for segment_left, segment_right in self._active_segments[int(node_id)]:
                left.append(float(segment_left))
                right.append(float(segment_right))
            offsets[lineage_index + 1] = len(left)
        return CompactActiveFrontier(
            node_ids=node_ids,
            segment_offsets=offsets,
            segment_left=np.asarray(left, dtype=np.float64),
            segment_right=np.asarray(right, dtype=np.float64),
        )


@dataclass(frozen=True)
class _EventBuild:
    kind: int
    time: float
    nodes: tuple[int, ...]
    edges: tuple[int, ...]


class ARGTrace:
    """Column-array-backed ARG construction trace."""

    def __init__(
        self,
        *,
        sequence_length: float,
        node_time: np.ndarray,
        node_flags: np.ndarray,
        edge_left: np.ndarray,
        edge_right: np.ndarray,
        edge_parent: np.ndarray,
        edge_child: np.ndarray,
        sample_nodes: np.ndarray,
        event_kind: np.ndarray,
        event_time: np.ndarray,
        event_node_start: np.ndarray,
        event_nodes: np.ndarray,
        event_edge_start: np.ndarray,
        event_edges: np.ndarray,
        node_reveal_step: np.ndarray,
        edge_reveal_step: np.ndarray,
        recombination_flag: int = RECOMBINATION_NODE_FLAG,
        checkpoint_interval: int = 1024,
        strict: bool = True,
    ):
        self.sequence_length = float(sequence_length)
        self.node_time = np.asarray(node_time, dtype=np.float64)
        self.node_flags = np.asarray(node_flags, dtype=np.int64)
        self.edge_left = np.asarray(edge_left, dtype=np.float64)
        self.edge_right = np.asarray(edge_right, dtype=np.float64)
        self.edge_parent = np.asarray(edge_parent, dtype=np.int32)
        self.edge_child = np.asarray(edge_child, dtype=np.int32)
        self.sample_nodes = np.asarray(sample_nodes, dtype=np.int32)

        self.event_kind = np.asarray(event_kind, dtype=np.int8)
        self.event_time = np.asarray(event_time, dtype=np.float64)
        self.event_node_start = np.asarray(event_node_start, dtype=np.int64)
        self.event_nodes = np.asarray(event_nodes, dtype=np.int32)
        self.event_edge_start = np.asarray(event_edge_start, dtype=np.int64)
        self.event_edges = np.asarray(event_edges, dtype=np.int64)
        self.node_reveal_step = np.asarray(node_reveal_step, dtype=np.int64)
        self.edge_reveal_step = np.asarray(edge_reveal_step, dtype=np.int64)

        self.recombination_flag = int(recombination_flag)
        self.checkpoint_interval = max(1, int(checkpoint_interval))
        self.strict = bool(strict)
        self._checkpoints: dict[int, ActiveSegmentMap] = {
            0: self._initial_active_segments()
        }

    @property
    def num_steps(self) -> int:
        return int(self.event_kind.size)

    @property
    def event_count(self) -> int:
        return int(self.event_kind.size)

    @property
    def recombination_event_count(self) -> int:
        return int(np.sum(self.event_kind == EVENT_KIND_RECOMBINATION))

    @property
    def coalescence_event_count(self) -> int:
        return int(np.sum(self.event_kind == EVENT_KIND_COALESCENCE))

    def event_at_index(self, event_index: int) -> ARGEvent:
        event_index = int(event_index)
        if event_index < 0 or event_index >= self.event_count:
            raise IndexError("event_index out of range")
        node_start, node_end = self._event_node_bounds(event_index)
        edge_start, edge_end = self._event_edge_bounds(event_index)
        kind = EVENT_KIND_NAMES[int(self.event_kind[event_index])]
        return ARGEvent(
            step=event_index + 1,
            kind=kind,
            time=float(self.event_time[event_index]),
            node_ids=tuple(int(v) for v in self.event_nodes[node_start:node_end]),
            edge_ids=tuple(int(v) for v in self.event_edges[edge_start:edge_end]),
        )

    def state_at_step(self, step: int, checkpoint: bool = True) -> TraceState:
        step = self._validate_step(step)
        start_step = 0
        active = self._initial_active_segments()

        if checkpoint and self._checkpoints:
            available = [candidate for candidate in self._checkpoints if candidate <= step]
            if available:
                start_step = max(available)
                active = _clone_active_segments(self._checkpoints[start_step])

        for event_index in range(start_step, step):
            self._apply_event(active, event_index)
            completed_step = event_index + 1
            if (
                checkpoint
                and completed_step % self.checkpoint_interval == 0
                and completed_step not in self._checkpoints
            ):
                self._checkpoints[completed_step] = _clone_active_segments(active)

        return self._make_state(step, active)

    def previous_state(self, state: TraceState) -> TraceState:
        if state.step <= 0:
            raise ValueError("initial state has no previous state")
        active = _clone_active_segments(dict(state._active_segments))
        self._apply_inverse_event(active, state.step - 1)
        return self._make_state(state.step - 1, active)

    def graph_at_step(
        self,
        step: int,
        genomic_range: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        state = self.state_at_step(step)
        edge_ids = state.visible_edge_ids
        edge_left = self.edge_left[edge_ids]
        edge_right = self.edge_right[edge_ids]

        start = end = None
        if genomic_range is not None:
            start, end = _validate_genomic_range(genomic_range, self.sequence_length)
            keep = (edge_right > start) & (edge_left < end)
            edge_ids = edge_ids[keep]
            edge_left = np.maximum(edge_left[keep], start)
            edge_right = np.minimum(edge_right[keep], end)

        if genomic_range is None:
            node_ids = state.visible_node_ids
        else:
            node_set = set()
            for edge_id in edge_ids:
                node_set.add(int(self.edge_parent[edge_id]))
                node_set.add(int(self.edge_child[edge_id]))
            for node_id, segments in state._active_segments.items():
                if _segments_overlap(segments, start, end):
                    node_set.add(int(node_id))
            node_ids = np.asarray(sorted(node_set), dtype=np.int32)

        nodes = [self._node_dict(int(node_id)) for node_id in node_ids]
        edges = [
            self._edge_dict(int(edge_id), float(left), float(right))
            for edge_id, left, right in zip(edge_ids, edge_left, edge_right)
            if left < right
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "step": int(step),
                "current_time": float(state.current_time),
                "genomic_range": None if genomic_range is None else [start, end],
                "visible_node_count": int(len(nodes)),
                "visible_edge_count": int(len(edges)),
                "event_count": self.event_count,
            },
        }

    def to_tree_sequence_at_step(
        self,
        step: int,
        genomic_range: tuple[float, float] | None = None,
    ) -> tskit.TreeSequence:
        state = self.state_at_step(step)
        tables = tskit.TableCollection(sequence_length=self.sequence_length)
        for node_id in range(self.node_time.size):
            tables.nodes.add_row(
                flags=int(self.node_flags[node_id]),
                time=float(self.node_time[node_id]),
            )

        start = end = None
        if genomic_range is not None:
            start, end = _validate_genomic_range(genomic_range, self.sequence_length)

        for edge_id in state.visible_edge_ids:
            left = float(self.edge_left[edge_id])
            right = float(self.edge_right[edge_id])
            if genomic_range is not None:
                left = max(left, start)
                right = min(right, end)
            if left < right:
                tables.edges.add_row(
                    left=left,
                    right=right,
                    parent=int(self.edge_parent[edge_id]),
                    child=int(self.edge_child[edge_id]),
                )
        tables.sort()
        return tables.tree_sequence()

    def _initial_active_segments(self) -> ActiveSegmentMap:
        return {
            int(node_id): ((0.0, self.sequence_length),)
            for node_id in self.sample_nodes
        }

    def _validate_step(self, step: int) -> int:
        step = int(step)
        if step < 0 or step > self.num_steps:
            raise ValueError(f"step must be in [0, {self.num_steps}], got {step}")
        return step

    def _event_node_bounds(self, event_index: int) -> tuple[int, int]:
        return (
            int(self.event_node_start[event_index]),
            int(self.event_node_start[event_index + 1]),
        )

    def _event_edge_bounds(self, event_index: int) -> tuple[int, int]:
        return (
            int(self.event_edge_start[event_index]),
            int(self.event_edge_start[event_index + 1]),
        )

    def _event_edge_ids(self, event_index: int) -> np.ndarray:
        start, end = self._event_edge_bounds(event_index)
        return self.event_edges[start:end]

    def _make_state(self, step: int, active: ActiveSegmentMap) -> TraceState:
        visible_node_ids = np.flatnonzero(
            (self.node_reveal_step >= 0) & (self.node_reveal_step <= step)
        ).astype(np.int32)
        visible_edge_ids = np.flatnonzero(
            (self.edge_reveal_step >= 0) & (self.edge_reveal_step <= step)
        ).astype(np.int64)
        current_time = 0.0 if step == 0 else float(self.event_time[step - 1])
        visible_node_ids.setflags(write=False)
        visible_edge_ids.setflags(write=False)
        return TraceState(
            step=int(step),
            current_time=current_time,
            visible_node_ids=visible_node_ids,
            visible_edge_ids=visible_edge_ids,
            _active_segments=_clone_active_segments(active),
        )

    def _apply_event(self, active: ActiveSegmentMap, event_index: int) -> None:
        edge_ids = self._event_edge_ids(event_index)
        parent_segments = self._parent_segment_union(edge_ids)

        for edge_id in edge_ids:
            child = int(self.edge_child[edge_id])
            left = float(self.edge_left[edge_id])
            right = float(self.edge_right[edge_id])
            active[child] = _subtract_segment(
                active.get(child, ()),
                left,
                right,
                strict=self.strict,
            )
            if not active[child]:
                del active[child]

        for parent, segments in parent_segments.items():
            active[parent] = _merge_segments(active.get(parent, ()) + segments)

    def _apply_inverse_event(self, active: ActiveSegmentMap, event_index: int) -> None:
        edge_ids = self._event_edge_ids(event_index)
        parent_segments = self._parent_segment_union(edge_ids)

        for parent, segments in parent_segments.items():
            updated = active.get(parent, ())
            for left, right in segments:
                updated = _subtract_segment(
                    updated,
                    left,
                    right,
                    strict=self.strict,
                )
            if updated:
                active[parent] = updated
            elif parent in active:
                del active[parent]

        for edge_id in edge_ids:
            child = int(self.edge_child[edge_id])
            left = float(self.edge_left[edge_id])
            right = float(self.edge_right[edge_id])
            active[child] = _merge_segments(active.get(child, ()) + ((left, right),))

    def _parent_segment_union(self, edge_ids: np.ndarray) -> dict[int, tuple[Segment, ...]]:
        grouped: dict[int, tuple[Segment, ...]] = {}
        for edge_id in edge_ids:
            parent = int(self.edge_parent[edge_id])
            segment = ((float(self.edge_left[edge_id]), float(self.edge_right[edge_id])),)
            grouped[parent] = _merge_segments(grouped.get(parent, ()) + segment)
        return grouped

    def _node_dict(self, node_id: int) -> dict[str, Any]:
        flags = int(self.node_flags[node_id])
        return {
            "id": int(node_id),
            "time": float(self.node_time[node_id]),
            "flags": flags,
            "is_sample": bool(flags & tskit.NODE_IS_SAMPLE),
            "is_recombination": bool(flags & self.recombination_flag),
        }

    def _edge_dict(self, edge_id: int, left: float, right: float) -> dict[str, Any]:
        return {
            "id": int(edge_id),
            "source": int(self.edge_parent[edge_id]),
            "target": int(self.edge_child[edge_id]),
            "left": float(left),
            "right": float(right),
        }


class FastARGTrace:
    """Compact full-trace backend for large synthetic full-ARG tree sequences."""

    def __init__(
        self,
        *,
        sequence_length: float,
        node_time: np.ndarray,
        node_flags: np.ndarray,
        edge_left: np.ndarray,
        edge_right: np.ndarray,
        edge_parent: np.ndarray,
        edge_child: np.ndarray,
        sample_nodes: np.ndarray,
        event_kind: np.ndarray,
        event_time: np.ndarray,
        event_node1: np.ndarray,
        event_node2: np.ndarray,
        event_edge_start: np.ndarray,
        revealed_edge_ids: np.ndarray,
        revealed_node_ids: np.ndarray,
        visible_node_end: np.ndarray,
        node_reveal_step: np.ndarray,
        recombination_flag: int = RECOMBINATION_NODE_FLAG,
        strict: bool = True,
    ):
        self.sequence_length = float(sequence_length)
        self.node_time = np.asarray(node_time, dtype=np.float64)
        self.node_flags = np.asarray(node_flags, dtype=np.uint32)
        self.edge_left = np.asarray(edge_left, dtype=np.float64)
        self.edge_right = np.asarray(edge_right, dtype=np.float64)
        self.edge_parent = np.asarray(edge_parent, dtype=np.int32)
        self.edge_child = np.asarray(edge_child, dtype=np.int32)
        self.sample_nodes = np.asarray(sample_nodes, dtype=np.int32)

        self.event_kind = np.asarray(event_kind, dtype=np.uint8)
        self.event_time = np.asarray(event_time, dtype=np.float64)
        self.event_node1 = np.asarray(event_node1, dtype=np.int32)
        self.event_node2 = np.asarray(event_node2, dtype=np.int32)
        self.event_edge_start = np.asarray(event_edge_start, dtype=np.int32)
        self.revealed_edge_ids = np.asarray(revealed_edge_ids, dtype=np.int32)
        self.revealed_node_ids = np.asarray(revealed_node_ids, dtype=np.int32)
        self.visible_node_end = np.asarray(visible_node_end, dtype=np.int32)
        self.node_reveal_step = np.asarray(node_reveal_step, dtype=np.int32)

        self.recombination_flag = int(recombination_flag)
        self.strict = bool(strict)
        self._recombination_event_count = int(
            np.sum(self.event_kind == EVENT_KIND_RECOMBINATION)
        )
        self._coalescence_event_count = int(
            np.sum(self.event_kind == EVENT_KIND_COALESCENCE)
        )

    @property
    def num_steps(self) -> int:
        return int(self.event_kind.size)

    @property
    def event_count(self) -> int:
        return int(self.event_kind.size)

    @property
    def recombination_event_count(self) -> int:
        return self._recombination_event_count

    @property
    def coalescence_event_count(self) -> int:
        return self._coalescence_event_count

    def event_at_index(self, event_index: int) -> ARGEvent:
        event_index = int(event_index)
        if event_index < 0 or event_index >= self.event_count:
            raise IndexError("event_index out of range")
        edge_start, edge_end = self._event_edge_bounds(event_index)
        node_ids = [int(self.event_node1[event_index])]
        node2 = int(self.event_node2[event_index])
        if node2 >= 0:
            node_ids.append(node2)
        kind = EVENT_KIND_NAMES[int(self.event_kind[event_index])]
        return ARGEvent(
            step=event_index + 1,
            kind=kind,
            time=float(self.event_time[event_index]),
            node_ids=tuple(node_ids),
            edge_ids=tuple(
                int(v) for v in self.revealed_edge_ids[edge_start:edge_end]
            ),
        )

    def state_at_step(self, step: int, *, include_active: bool = False) -> TraceState:
        step = self._validate_step(step)
        if include_active:
            return self.initial_state().advance_to(step).as_trace_state()
        node_end = int(self.visible_node_end[step])
        edge_end = int(self.event_edge_start[step])
        visible_node_ids = self.revealed_node_ids[:node_end]
        visible_edge_ids = self.revealed_edge_ids[:edge_end]
        visible_node_ids.setflags(write=False)
        visible_edge_ids.setflags(write=False)
        current_time = 0.0 if step == 0 else float(self.event_time[step - 1])
        return TraceState(
            step=int(step),
            current_time=current_time,
            visible_node_ids=visible_node_ids,
            visible_edge_ids=visible_edge_ids,
            _active_segments=None,
        )

    def initial_state(
        self,
        *,
        chunk_size: int = 65_536,
        initial_segment_capacity: int | None = None,
    ) -> FastARGState:
        """Return a mutable full-chromosome state at construction step zero."""
        return FastARGState(
            self,
            chunk_size=chunk_size,
            initial_segment_capacity=initial_segment_capacity,
        )

    def graph_at_step(
        self,
        step: int,
        genomic_range: tuple[float, float] | None = None,
        *,
        max_edges: int | None = 2_000_000,
    ) -> dict[str, Any]:
        state = self.state_at_step(step)
        edge_ids = state.visible_edge_ids

        start = end = None
        if genomic_range is None:
            if max_edges is not None and edge_ids.size > int(max_edges):
                raise ValueError(
                    "unwindowed graph materialization would create "
                    f"{edge_ids.size} edge dictionaries; pass genomic_range or "
                    "set max_edges=None to override"
                )
            edge_left = self.edge_left[edge_ids]
            edge_right = self.edge_right[edge_ids]
            node_ids = state.visible_node_ids
        else:
            start, end = _validate_genomic_range(genomic_range, self.sequence_length)
            edge_left_all = self.edge_left[edge_ids]
            edge_right_all = self.edge_right[edge_ids]
            keep = (edge_right_all > start) & (edge_left_all < end)
            edge_ids = edge_ids[keep]
            if max_edges is not None and edge_ids.size > int(max_edges):
                raise ValueError(
                    "windowed graph materialization would create "
                    f"{edge_ids.size} edge dictionaries; narrow genomic_range or "
                    "set max_edges=None to override"
                )
            edge_left = np.maximum(edge_left_all[keep], start)
            edge_right = np.minimum(edge_right_all[keep], end)
            if edge_ids.size:
                node_ids = np.unique(
                    np.concatenate(
                        (self.edge_parent[edge_ids], self.edge_child[edge_ids])
                    )
                ).astype(np.int32, copy=False)
            else:
                node_ids = np.empty(0, dtype=np.int32)

        nodes = [self._node_dict(int(node_id)) for node_id in node_ids]
        edges = [
            self._edge_dict(int(edge_id), float(left), float(right))
            for edge_id, left, right in zip(edge_ids, edge_left, edge_right)
            if left < right
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "step": int(step),
                "current_time": float(state.current_time),
                "genomic_range": None if genomic_range is None else [start, end],
                "visible_node_count": int(len(nodes)),
                "visible_edge_count": int(len(edges)),
                "event_count": self.event_count,
            },
        }

    def to_tree_sequence_at_step(
        self,
        step: int,
        genomic_range: tuple[float, float] | None = None,
    ) -> tskit.TreeSequence:
        step = self._validate_step(step)
        edge_ids = self.revealed_edge_ids[: int(self.event_edge_start[step])]
        tables = tskit.TableCollection(sequence_length=self.sequence_length)
        tables.nodes.set_columns(
            flags=self.node_flags.astype(np.uint32, copy=False),
            time=self.node_time,
        )

        start = end = None
        if genomic_range is None:
            left = self.edge_left[edge_ids]
            right = self.edge_right[edge_ids]
            parent = self.edge_parent[edge_ids]
            child = self.edge_child[edge_ids]
        else:
            start, end = _validate_genomic_range(genomic_range, self.sequence_length)
            edge_left = self.edge_left[edge_ids]
            edge_right = self.edge_right[edge_ids]
            keep = (edge_right > start) & (edge_left < end)
            edge_ids = edge_ids[keep]
            left = np.maximum(edge_left[keep], start)
            right = np.minimum(edge_right[keep], end)
            parent = self.edge_parent[edge_ids]
            child = self.edge_child[edge_ids]

        non_empty = left < right
        tables.edges.set_columns(
            left=left[non_empty],
            right=right[non_empty],
            parent=parent[non_empty],
            child=child[non_empty],
        )
        tables.sort()
        return tables.tree_sequence()

    def _validate_step(self, step: int) -> int:
        step = int(step)
        if step < 0 or step > self.num_steps:
            raise ValueError(f"step must be in [0, {self.num_steps}], got {step}")
        return step

    def _event_edge_bounds(self, event_index: int) -> tuple[int, int]:
        return (
            int(self.event_edge_start[event_index]),
            int(self.event_edge_start[event_index + 1]),
        )

    def _event_edge_ids(self, event_index: int) -> np.ndarray:
        start, end = self._event_edge_bounds(event_index)
        return self.revealed_edge_ids[start:end]

    def _initial_active_segments(self) -> ActiveSegmentMap:
        return {
            int(node_id): ((0.0, self.sequence_length),)
            for node_id in self.sample_nodes
        }

    def _active_segments_at_step(self, step: int) -> ActiveSegmentMap:
        active = self._initial_active_segments()
        for event_index in range(step):
            self._apply_event(active, event_index)
        return _clone_active_segments(active)

    def _apply_event(self, active: ActiveSegmentMap, event_index: int) -> None:
        edge_ids = self._event_edge_ids(event_index)
        parent_segments = self._parent_segment_union(edge_ids)

        for edge_id in edge_ids:
            child = int(self.edge_child[edge_id])
            left = float(self.edge_left[edge_id])
            right = float(self.edge_right[edge_id])
            active[child] = _subtract_segment(
                active.get(child, ()),
                left,
                right,
                strict=self.strict,
            )
            if not active[child]:
                del active[child]

        for parent, segments in parent_segments.items():
            active[parent] = _merge_segments(active.get(parent, ()) + segments)

    def _parent_segment_union(self, edge_ids: np.ndarray) -> dict[int, tuple[Segment, ...]]:
        grouped: dict[int, tuple[Segment, ...]] = {}
        for edge_id in edge_ids:
            parent = int(self.edge_parent[edge_id])
            segment = ((float(self.edge_left[edge_id]), float(self.edge_right[edge_id])),)
            grouped[parent] = _merge_segments(grouped.get(parent, ()) + segment)
        return grouped

    def _node_dict(self, node_id: int) -> dict[str, Any]:
        flags = int(self.node_flags[node_id])
        return {
            "id": int(node_id),
            "time": float(self.node_time[node_id]),
            "flags": flags,
            "is_sample": bool(flags & tskit.NODE_IS_SAMPLE),
            "is_recombination": bool(flags & self.recombination_flag),
        }

    def _edge_dict(self, edge_id: int, left: float, right: float) -> dict[str, Any]:
        return {
            "id": int(edge_id),
            "source": int(self.edge_parent[edge_id]),
            "target": int(self.edge_child[edge_id]),
            "left": float(left),
            "right": float(right),
        }


class FastARGState:
    """Mutable stepwise full-ARG state backed by compact interval arrays."""

    def __init__(
        self,
        trace: FastARGTrace,
        *,
        chunk_size: int = 65_536,
        initial_segment_capacity: int | None = None,
    ):
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.trace = trace
        self.chunk_size = chunk_size
        self.step = 0

        num_nodes = int(trace.node_time.size)
        self._lineage_segment_head = np.full(num_nodes, -1, dtype=np.int32)

        sample_count = int(trace.sample_nodes.size)
        if initial_segment_capacity is None:
            initial_segment_capacity = max(16, sample_count * 2)
        capacity = max(int(initial_segment_capacity), sample_count, 1)
        _ensure_int32_capacity("active segment capacity", capacity)
        self._segment_left = np.empty(capacity, dtype=np.float64)
        self._segment_right = np.empty(capacity, dtype=np.float64)
        self._segment_next = np.full(capacity, -1, dtype=np.int32)

        previous_samples: set[int] = set()
        for segment_index, node_value in enumerate(trace.sample_nodes):
            node_id = int(node_value)
            if node_id in previous_samples:
                raise ValueError(f"duplicate sample node {node_id}")
            previous_samples.add(node_id)
            self._lineage_segment_head[node_id] = segment_index
            self._segment_left[segment_index] = 0.0
            self._segment_right[segment_index] = trace.sequence_length

        self._pool_high_water = sample_count
        self._free_head = -1
        self._free_count = 0
        self._active_count = sample_count
        self._segment_count = sample_count

    @property
    def current_time(self) -> float:
        if self.step == 0:
            return 0.0
        return float(self.trace.event_time[self.step - 1])

    @property
    def is_terminal(self) -> bool:
        return self.step == self.trace.num_steps

    @property
    def active_count(self) -> int:
        return int(self._active_count)

    @property
    def segment_count(self) -> int:
        return int(self._segment_count)

    @property
    def visible_node_ids(self) -> np.ndarray:
        end = int(self.trace.visible_node_end[self.step])
        values = self.trace.revealed_node_ids[:end]
        values.setflags(write=False)
        return values

    @property
    def visible_edge_ids(self) -> np.ndarray:
        end = int(self.trace.event_edge_start[self.step])
        values = self.trace.revealed_edge_ids[:end]
        values.setflags(write=False)
        return values

    @property
    def active_lineages(self) -> tuple[ActiveLineage, ...]:
        return self.compact_active_frontier().to_active_lineages()

    def advance(self, steps: int = 1) -> FastARGState:
        steps = int(steps)
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        return self.advance_to(self.step + steps)

    def advance_to(self, step: int) -> FastARGState:
        step = self.trace._validate_step(step)
        if step < self.step:
            raise ValueError("advance_to cannot move backward; use backtrack_to")
        while self.step < step:
            chunk_end = min(step, self.step + self.chunk_size)
            self._ensure_transition_capacity(chunk_end, forward=True)
            result = _advance_frontier_range(
                self.step,
                chunk_end,
                self.trace.event_node1,
                self.trace.event_node2,
                self.trace.event_edge_start,
                self.trace.revealed_edge_ids,
                self.trace.edge_left,
                self.trace.edge_right,
                self.trace.edge_parent,
                self.trace.edge_child,
                self._lineage_segment_head,
                self._segment_left,
                self._segment_right,
                self._segment_next,
                self._pool_high_water,
                self._free_head,
                self._free_count,
                self._active_count,
                self._segment_count,
                self.trace.strict,
            )
            self._accept_kernel_result(result, direction="forward")
        return self

    def backtrack(self, steps: int = 1) -> FastARGState:
        steps = int(steps)
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        return self.backtrack_to(self.step - steps)

    def backtrack_to(self, step: int) -> FastARGState:
        step = self.trace._validate_step(step)
        if step > self.step:
            raise ValueError("backtrack_to cannot move forward; use advance_to")
        while self.step > step:
            chunk_start = max(step, self.step - self.chunk_size)
            self._ensure_transition_capacity(chunk_start, forward=False)
            result = _backtrack_frontier_range(
                self.step,
                chunk_start,
                self.trace.event_node1,
                self.trace.event_node2,
                self.trace.event_edge_start,
                self.trace.revealed_edge_ids,
                self.trace.edge_left,
                self.trace.edge_right,
                self.trace.edge_parent,
                self.trace.edge_child,
                self._lineage_segment_head,
                self._segment_left,
                self._segment_right,
                self._segment_next,
                self._pool_high_water,
                self._free_head,
                self._free_count,
                self._active_count,
                self._segment_count,
                self.trace.strict,
            )
            self._accept_kernel_result(result, direction="backward")
        return self

    def move_to(self, step: int) -> FastARGState:
        step = self.trace._validate_step(step)
        if step >= self.step:
            return self.advance_to(step)
        return self.backtrack_to(step)

    def compact_active_frontier(self) -> CompactActiveFrontier:
        node_end = int(self.trace.visible_node_end[self.step])
        (
            node_ids,
            offsets,
            segment_left,
            segment_right,
            observed_active,
            observed_segments,
        ) = _materialize_compact_frontier(
            self.trace.revealed_node_ids,
            node_end,
            self._lineage_segment_head,
            self._segment_left,
            self._segment_right,
            self._segment_next,
            self._active_count,
            self._segment_count,
        )
        if observed_active != self._active_count or observed_segments != self._segment_count:
            raise RuntimeError(
                "active frontier counts are inconsistent: "
                f"expected {self._active_count} lineages/{self._segment_count} segments, "
                f"observed {observed_active}/{observed_segments}"
            )
        return CompactActiveFrontier(
            node_ids=node_ids,
            segment_offsets=offsets,
            segment_left=segment_left,
            segment_right=segment_right,
        )

    def as_trace_state(self) -> TraceState:
        return TraceState(
            step=int(self.step),
            current_time=self.current_time,
            visible_node_ids=self.visible_node_ids,
            visible_edge_ids=self.visible_edge_ids,
            _active_segments=None,
            _compact_active=self.compact_active_frontier(),
        )

    def clone(self) -> FastARGState:
        cloned = object.__new__(FastARGState)
        cloned.trace = self.trace
        cloned.chunk_size = self.chunk_size
        cloned.step = self.step
        cloned._lineage_segment_head = self._lineage_segment_head.copy()
        cloned._segment_left = self._segment_left.copy()
        cloned._segment_right = self._segment_right.copy()
        cloned._segment_next = self._segment_next.copy()
        cloned._pool_high_water = self._pool_high_water
        cloned._free_head = self._free_head
        cloned._free_count = self._free_count
        cloned._active_count = self._active_count
        cloned._segment_count = self._segment_count
        return cloned

    def _ensure_transition_capacity(self, target_step: int, *, forward: bool) -> None:
        lower = min(self.step, int(target_step))
        upper = max(self.step, int(target_step))
        edge_count = int(
            self.trace.event_edge_start[upper]
            - self.trace.event_edge_start[lower]
        )
        allocation_bound = edge_count * (2 if forward else 1)
        new_slots = max(0, allocation_bound - int(self._free_count))
        self._ensure_segment_capacity(self._pool_high_water + new_slots)

    def _ensure_segment_capacity(self, required: int) -> None:
        required = int(required)
        current = int(self._segment_left.size)
        if required <= current:
            return
        capacity = max(required, current + max(current // 2, 16))
        _ensure_int32_capacity("active segment capacity", capacity)
        segment_left = np.empty(capacity, dtype=np.float64)
        segment_right = np.empty(capacity, dtype=np.float64)
        segment_next = np.full(capacity, -1, dtype=np.int32)
        segment_left[: self._pool_high_water] = self._segment_left[: self._pool_high_water]
        segment_right[: self._pool_high_water] = self._segment_right[: self._pool_high_water]
        segment_next[: self._pool_high_water] = self._segment_next[: self._pool_high_water]
        self._segment_left = segment_left
        self._segment_right = segment_right
        self._segment_next = segment_next

    def _accept_kernel_result(self, result: tuple[int, ...], *, direction: str) -> None:
        (
            completed_step,
            self._pool_high_water,
            self._free_head,
            self._free_count,
            self._active_count,
            self._segment_count,
            error_code,
            error_event,
            error_edge,
        ) = (int(value) for value in result)
        self.step = completed_step
        if error_code == 0:
            return
        if error_code == 1:
            edge_id = int(error_edge)
            raise ValueError(
                "cannot subtract inactive material segment while moving "
                f"{direction} at event {error_event}, edge {edge_id}: "
                f"child={int(self.trace.edge_child[edge_id])}, "
                f"interval=[{float(self.trace.edge_left[edge_id])}, "
                f"{float(self.trace.edge_right[edge_id])})"
            )
        if error_code == 2:
            raise ValueError(
                f"event {error_event} reveals a parent that is already active"
            )
        if error_code == 3:
            raise RuntimeError("active segment pool capacity was exhausted")
        if error_code == 4:
            raise ValueError(
                f"cannot backtrack event {error_event}: revealed parent is not active"
            )
        raise RuntimeError(f"unknown active frontier kernel error {error_code}")


def build_fast_trace_from_full_arg(
    ts_or_tables: str | Path | tskit.TreeSequence | tskit.TableCollection,
    *,
    strict: bool = True,
    source_storage: str = "compact",
    recombination_flag: int = RECOMBINATION_NODE_FLAG,
) -> FastARGTrace:
    """Build a compact full-ARG replay trace for large synthetic ARG inputs."""
    if source_storage != "compact":
        raise ValueError("only source_storage='compact' is currently supported")

    tables = _load_table_collection(ts_or_tables)
    sequence_length = float(tables.sequence_length)
    node_time = np.asarray(tables.nodes.time, dtype=np.float64).copy()
    node_flags = np.asarray(tables.nodes.flags, dtype=np.uint32).copy()
    edge_left = np.asarray(tables.edges.left, dtype=np.float64).copy()
    edge_right = np.asarray(tables.edges.right, dtype=np.float64).copy()
    edge_parent = np.asarray(tables.edges.parent, dtype=np.int32).copy()
    edge_child = np.asarray(tables.edges.child, dtype=np.int32).copy()
    del tables

    num_nodes = int(node_time.size)
    num_edges = int(edge_parent.size)
    _ensure_int32_capacity("num_nodes", num_nodes)
    _ensure_int32_capacity("num_edges", num_edges)

    if strict:
        bad_edge = _first_bad_edge_time(node_time, edge_parent, edge_child)
        if bad_edge >= 0:
            edge_id = int(bad_edge)
            raise ValueError(
                "all edges must satisfy parent.time > child.time; "
                f"edge {edge_id} violates this"
            )

    sample_nodes = np.flatnonzero((node_flags & tskit.NODE_IS_SAMPLE) != 0).astype(
        np.int32
    )
    recombination_flag = int(recombination_flag)
    recombination_nodes = np.flatnonzero(
        (node_flags & recombination_flag) != 0
    ).astype(np.int32)
    if strict and recombination_nodes.size == 0:
        raise ValueError(
            "no explicit recombination nodes found; pass a full-ARG `.trees` file "
            "or build a synthetic full ARG first"
        )
    if recombination_nodes.size % 2:
        if strict:
            raise ValueError(
                "fast trace requires an even number of synthetic recombination nodes"
            )
        recombination_nodes = recombination_nodes[:-1]

    parent_edge_start = _build_parent_edge_start(edge_parent, num_nodes)
    parent_edge_ids = _fill_parent_edge_ids(edge_parent, parent_edge_start)
    child_count, first_child = _build_child_summary_by_node(
        parent_edge_start,
        parent_edge_ids,
        edge_child,
    )

    if recombination_nodes.size:
        bad_recombination = _first_invalid_recombination_node(
            recombination_nodes,
            child_count,
        )
        if bad_recombination >= 0:
            node_id = int(bad_recombination)
            raise ValueError(
                "fast trace requires each synthetic recombination node to have "
                f"outgoing edges to exactly one child; node {node_id} violates this"
            )
        bad_pair = _first_bad_recombination_pair(
            recombination_nodes,
            node_time,
            first_child,
        )
        if bad_pair >= 0:
            left = int(recombination_nodes[2 * int(bad_pair)])
            right = int(recombination_nodes[2 * int(bad_pair) + 1])
            raise ValueError(
                "fast trace requires consecutive synthetic recombination nodes "
                f"to be paired by matching time and child; nodes {left}, {right} "
                "violate this"
            )

    sample_mask = np.zeros(num_nodes, dtype=bool)
    sample_mask[sample_nodes] = True
    recombination_mask = np.zeros(num_nodes, dtype=bool)
    recombination_mask[recombination_nodes] = True
    other_nodes = np.flatnonzero(~(sample_mask | recombination_mask)).astype(np.int32)
    del sample_mask, recombination_mask

    recombination_event_count = int(recombination_nodes.size // 2)
    other_event_count = int(other_nodes.size)
    event_count = recombination_event_count + other_event_count
    _ensure_int32_capacity("event_count", event_count)

    raw_kind = np.empty(event_count, dtype=np.uint8)
    raw_time = np.empty(event_count, dtype=np.float64)
    raw_node1 = np.empty(event_count, dtype=np.int32)
    raw_node2 = np.full(event_count, -1, dtype=np.int32)
    raw_edge_count = np.empty(event_count, dtype=np.int32)

    if recombination_event_count:
        left_nodes = recombination_nodes[0::2]
        right_nodes = recombination_nodes[1::2]
        raw_kind[:recombination_event_count] = EVENT_KIND_RECOMBINATION
        raw_time[:recombination_event_count] = node_time[left_nodes]
        raw_node1[:recombination_event_count] = left_nodes
        raw_node2[:recombination_event_count] = right_nodes
        raw_edge_count[:recombination_event_count] = (
            parent_edge_start[left_nodes + 1]
            - parent_edge_start[left_nodes]
            + parent_edge_start[right_nodes + 1]
            - parent_edge_start[right_nodes]
        )

    other_start = recombination_event_count
    if other_event_count:
        other_counts = child_count[other_nodes]
        other_kind = np.full(other_event_count, EVENT_KIND_REVEAL, dtype=np.uint8)
        other_kind[other_counts == 1] = EVENT_KIND_UNARY
        other_kind[other_counts >= 2] = EVENT_KIND_COALESCENCE
        raw_kind[other_start:] = other_kind
        raw_time[other_start:] = node_time[other_nodes]
        raw_node1[other_start:] = other_nodes
        raw_edge_count[other_start:] = (
            parent_edge_start[other_nodes + 1] - parent_edge_start[other_nodes]
        )
        del other_counts, other_kind
    del child_count, first_child, other_nodes, recombination_nodes

    priority = _event_priority_array(raw_kind)
    order = np.lexsort((raw_node2, raw_node1, priority, raw_time))
    del priority

    event_kind = raw_kind[order]
    event_time = raw_time[order]
    event_node1 = raw_node1[order]
    event_node2 = raw_node2[order]
    event_edge_count = raw_edge_count[order]
    del raw_kind, raw_time, raw_node1, raw_node2, raw_edge_count, order

    event_edge_start = _prefix_sum_int32(event_edge_count)
    if strict and int(event_edge_start[-1]) != num_edges:
        raise ValueError(
            "fast trace did not assign every source edge to exactly one event; "
            f"assigned {int(event_edge_start[-1])}, expected {num_edges}"
        )
    revealed_edge_ids = np.empty(int(event_edge_start[-1]), dtype=np.int32)
    _fill_fast_event_edges(
        event_node1,
        event_node2,
        event_edge_start,
        parent_edge_start,
        parent_edge_ids,
        revealed_edge_ids,
    )
    del parent_edge_start, parent_edge_ids, event_edge_count

    (
        revealed_node_ids,
        visible_node_end,
        node_reveal_step,
        duplicate_node,
    ) = _build_fast_node_reveal_arrays(sample_nodes, event_node1, event_node2, num_nodes)
    if strict and duplicate_node >= 0:
        raise ValueError(f"node {int(duplicate_node)} is revealed by multiple events")

    return FastARGTrace(
        sequence_length=sequence_length,
        node_time=node_time,
        node_flags=node_flags,
        edge_left=edge_left,
        edge_right=edge_right,
        edge_parent=edge_parent,
        edge_child=edge_child,
        sample_nodes=sample_nodes,
        event_kind=event_kind,
        event_time=event_time,
        event_node1=event_node1,
        event_node2=event_node2,
        event_edge_start=event_edge_start,
        revealed_edge_ids=revealed_edge_ids,
        revealed_node_ids=revealed_node_ids,
        visible_node_end=visible_node_end,
        node_reveal_step=node_reveal_step,
        recombination_flag=recombination_flag,
        strict=strict,
    )


def build_trace_from_full_arg(
    ts_or_path: str | Path | tskit.TreeSequence,
    *,
    recombination_flag: int = RECOMBINATION_NODE_FLAG,
    strict: bool = True,
) -> ARGTrace:
    """Build a replay trace from a full-ARG-like tskit tree sequence."""
    ts = _load_tree_sequence(ts_or_path)
    tables = ts.tables
    node_time = np.asarray(tables.nodes.time, dtype=np.float64).copy()
    node_flags = np.asarray(tables.nodes.flags, dtype=np.int64).copy()
    edge_left = np.asarray(tables.edges.left, dtype=np.float64).copy()
    edge_right = np.asarray(tables.edges.right, dtype=np.float64).copy()
    edge_parent = np.asarray(tables.edges.parent, dtype=np.int32).copy()
    edge_child = np.asarray(tables.edges.child, dtype=np.int32).copy()
    sample_nodes = np.asarray(ts.samples(), dtype=np.int32)
    recombination_flag = int(recombination_flag)

    if strict:
        _validate_edge_times(node_time, edge_parent, edge_child)

    parent_to_edges = _index_edges_by_parent(edge_parent)
    recombination_nodes = np.flatnonzero((node_flags & recombination_flag) != 0)
    if strict and recombination_nodes.size == 0:
        raise ValueError(
            "no explicit recombination nodes found; pass a full-ARG `.trees` file "
            "or build a synthetic full ARG first"
        )

    events = _extract_recombination_events(
        recombination_nodes=recombination_nodes,
        node_time=node_time,
        edge_child=edge_child,
        parent_to_edges=parent_to_edges,
        strict=strict,
    )
    recombination_node_set = {int(node_id) for node_id in recombination_nodes}
    sample_node_set = {int(node_id) for node_id in sample_nodes}
    grouped_recombination_nodes = {
        node_id for event in events for node_id in event.nodes
    }

    for node_id in range(node_time.size):
        if node_id in sample_node_set or node_id in grouped_recombination_nodes:
            continue
        edge_ids = tuple(parent_to_edges.get(node_id, ()))
        unique_children = {int(edge_child[edge_id]) for edge_id in edge_ids}
        if node_id in recombination_node_set:
            kind = EVENT_KIND_UNARY if edge_ids else EVENT_KIND_REVEAL
        elif len(unique_children) >= 2:
            kind = EVENT_KIND_COALESCENCE
        elif len(unique_children) == 1:
            kind = EVENT_KIND_UNARY
        else:
            kind = EVENT_KIND_REVEAL
        events.append(
            _EventBuild(
                kind=kind,
                time=float(node_time[node_id]),
                nodes=(int(node_id),),
                edges=tuple(int(edge_id) for edge_id in edge_ids),
            )
        )

    events.sort(
        key=lambda event: (
            float(event.time),
            _EVENT_KIND_PRIORITY[int(event.kind)],
            min(event.nodes) if event.nodes else -1,
            event.nodes,
        )
    )
    return _build_trace_from_events(
        ts=ts,
        node_time=node_time,
        node_flags=node_flags,
        edge_left=edge_left,
        edge_right=edge_right,
        edge_parent=edge_parent,
        edge_child=edge_child,
        sample_nodes=sample_nodes,
        events=events,
        recombination_flag=recombination_flag,
        strict=strict,
    )


def _build_trace_from_events(
    *,
    ts: tskit.TreeSequence,
    node_time: np.ndarray,
    node_flags: np.ndarray,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    edge_parent: np.ndarray,
    edge_child: np.ndarray,
    sample_nodes: np.ndarray,
    events: list[_EventBuild],
    recombination_flag: int,
    strict: bool,
) -> ARGTrace:
    event_kind = np.asarray([event.kind for event in events], dtype=np.int8)
    event_time = np.asarray([event.time for event in events], dtype=np.float64)

    node_offsets = [0]
    edge_offsets = [0]
    flat_nodes: list[int] = []
    flat_edges: list[int] = []
    node_reveal_step = np.full(node_time.size, -1, dtype=np.int64)
    node_reveal_step[sample_nodes] = 0

    for event_index, event in enumerate(events):
        step = event_index + 1
        for node_id in event.nodes:
            if strict and node_reveal_step[node_id] >= 0:
                raise ValueError(f"node {node_id} is revealed by multiple events")
            node_reveal_step[node_id] = step
            flat_nodes.append(int(node_id))
        node_offsets.append(len(flat_nodes))

        for edge_id in event.edges:
            flat_edges.append(int(edge_id))
        edge_offsets.append(len(flat_edges))

    edge_reveal_step = np.full(edge_parent.size, -1, dtype=np.int64)
    for edge_id in range(edge_parent.size):
        parent_step = int(node_reveal_step[int(edge_parent[edge_id])])
        child_step = int(node_reveal_step[int(edge_child[edge_id])])
        if parent_step >= 0 and child_step >= 0:
            edge_reveal_step[edge_id] = max(parent_step, child_step)
        elif strict:
            raise ValueError(f"edge {edge_id} has an unrevealed endpoint")

    return ARGTrace(
        sequence_length=float(ts.sequence_length),
        node_time=node_time,
        node_flags=node_flags,
        edge_left=edge_left,
        edge_right=edge_right,
        edge_parent=edge_parent,
        edge_child=edge_child,
        sample_nodes=sample_nodes,
        event_kind=event_kind,
        event_time=event_time,
        event_node_start=np.asarray(node_offsets, dtype=np.int64),
        event_nodes=np.asarray(flat_nodes, dtype=np.int32),
        event_edge_start=np.asarray(edge_offsets, dtype=np.int64),
        event_edges=np.asarray(flat_edges, dtype=np.int64),
        node_reveal_step=node_reveal_step,
        edge_reveal_step=edge_reveal_step,
        recombination_flag=recombination_flag,
        strict=strict,
    )


def _extract_recombination_events(
    *,
    recombination_nodes: np.ndarray,
    node_time: np.ndarray,
    edge_child: np.ndarray,
    parent_to_edges: Mapping[int, tuple[int, ...]],
    strict: bool,
) -> list[_EventBuild]:
    groups: dict[tuple[float, int], list[int]] = {}
    node_edges: dict[int, tuple[int, ...]] = {}

    for node_id in recombination_nodes:
        node_id = int(node_id)
        edge_ids = tuple(int(edge_id) for edge_id in parent_to_edges.get(node_id, ()))
        children = {int(edge_child[edge_id]) for edge_id in edge_ids}
        if len(children) != 1:
            if strict:
                raise ValueError(
                    f"recombination node {node_id} must have outgoing edges to "
                    f"exactly one child, found {len(children)}"
                )
            continue
        child = next(iter(children))
        key = (float(node_time[node_id]), int(child))
        groups.setdefault(key, []).append(node_id)
        node_edges[node_id] = edge_ids

    events: list[_EventBuild] = []
    for (time, _child), nodes in groups.items():
        nodes = sorted(nodes)
        if len(nodes) != 2:
            if strict:
                raise ValueError(
                    "could not pair recombination nodes by same time and child: "
                    f"time={time}, nodes={nodes}"
                )
            nodes = nodes[: len(nodes) - (len(nodes) % 2)]
        for i in range(0, len(nodes), 2):
            pair = tuple(nodes[i : i + 2])
            edge_ids = tuple(
                edge_id for node_id in pair for edge_id in node_edges[node_id]
            )
            events.append(
                _EventBuild(
                    kind=EVENT_KIND_RECOMBINATION,
                    time=float(time),
                    nodes=pair,
                    edges=tuple(sorted(edge_ids)),
                )
            )
    return events


def _load_tree_sequence(
    ts_or_path: str | Path | tskit.TreeSequence,
) -> tskit.TreeSequence:
    if isinstance(ts_or_path, tskit.TreeSequence):
        return ts_or_path
    return tskit.load(str(ts_or_path))


def _load_table_collection(
    ts_or_tables: str | Path | tskit.TreeSequence | tskit.TableCollection,
) -> tskit.TableCollection:
    if isinstance(ts_or_tables, tskit.TableCollection):
        return ts_or_tables
    if isinstance(ts_or_tables, tskit.TreeSequence):
        return ts_or_tables.dump_tables()
    return tskit.TableCollection.load(str(ts_or_tables))


def _ensure_int32_capacity(label: str, value: int) -> None:
    if int(value) > np.iinfo(np.int32).max:
        raise ValueError(f"{label}={value} exceeds int32 compact trace capacity")


def _event_priority_array(event_kind: np.ndarray) -> np.ndarray:
    priority = np.empty(event_kind.size, dtype=np.uint8)
    priority[event_kind == EVENT_KIND_RECOMBINATION] = _EVENT_KIND_PRIORITY[
        EVENT_KIND_RECOMBINATION
    ]
    priority[event_kind == EVENT_KIND_COALESCENCE] = _EVENT_KIND_PRIORITY[
        EVENT_KIND_COALESCENCE
    ]
    priority[event_kind == EVENT_KIND_UNARY] = _EVENT_KIND_PRIORITY[EVENT_KIND_UNARY]
    priority[event_kind == EVENT_KIND_REVEAL] = _EVENT_KIND_PRIORITY[EVENT_KIND_REVEAL]
    return priority


@njit(cache=True)
def _pool_allocate(
    segment_next: np.ndarray,
    pool_high_water: int,
    free_head: int,
    free_count: int,
) -> tuple[int, int, int, int]:
    if free_head >= 0:
        segment_index = free_head
        free_head = int(segment_next[segment_index])
        free_count -= 1
        return segment_index, pool_high_water, free_head, free_count
    if pool_high_water >= segment_next.size:
        return -1, pool_high_water, free_head, free_count
    segment_index = pool_high_water
    pool_high_water += 1
    return segment_index, pool_high_water, free_head, free_count


@njit(cache=True)
def _pool_release(
    segment_index: int,
    segment_next: np.ndarray,
    free_head: int,
    free_count: int,
) -> tuple[int, int]:
    segment_next[segment_index] = free_head
    return segment_index, free_count + 1


@njit(cache=True)
def _add_active_interval(
    node_id: int,
    left: float,
    right: float,
    lineage_segment_head: np.ndarray,
    segment_left: np.ndarray,
    segment_right: np.ndarray,
    segment_next: np.ndarray,
    pool_high_water: int,
    free_head: int,
    free_count: int,
    active_count: int,
    segment_count: int,
) -> tuple[int, int, int, int, int, int]:
    if left >= right:
        return (
            pool_high_water,
            free_head,
            free_count,
            active_count,
            segment_count,
            0,
        )

    was_inactive = lineage_segment_head[node_id] < 0
    previous = -1
    current = int(lineage_segment_head[node_id])
    while current >= 0 and segment_right[current] < left:
        previous = current
        current = int(segment_next[current])

    merged_left = left
    merged_right = right
    while current >= 0 and segment_left[current] <= merged_right:
        if segment_left[current] < merged_left:
            merged_left = segment_left[current]
        if segment_right[current] > merged_right:
            merged_right = segment_right[current]
        following = int(segment_next[current])
        if previous < 0:
            lineage_segment_head[node_id] = following
        else:
            segment_next[previous] = following
        free_head, free_count = _pool_release(
            current,
            segment_next,
            free_head,
            free_count,
        )
        segment_count -= 1
        current = following

    (
        segment_index,
        pool_high_water,
        free_head,
        free_count,
    ) = _pool_allocate(
        segment_next,
        pool_high_water,
        free_head,
        free_count,
    )
    if segment_index < 0:
        return (
            pool_high_water,
            free_head,
            free_count,
            active_count,
            segment_count,
            3,
        )
    segment_left[segment_index] = merged_left
    segment_right[segment_index] = merged_right
    segment_next[segment_index] = current
    if previous < 0:
        lineage_segment_head[node_id] = segment_index
    else:
        segment_next[previous] = segment_index
    segment_count += 1
    if was_inactive:
        active_count += 1
    return (
        pool_high_water,
        free_head,
        free_count,
        active_count,
        segment_count,
        0,
    )


@njit(cache=True)
def _active_interval_coverage(
    node_id: int,
    left: float,
    right: float,
    lineage_segment_head: np.ndarray,
    segment_left: np.ndarray,
    segment_right: np.ndarray,
    segment_next: np.ndarray,
) -> float:
    covered = 0.0
    current = int(lineage_segment_head[node_id])
    while current >= 0:
        current_left = segment_left[current]
        current_right = segment_right[current]
        if current_left >= right:
            break
        overlap_left = max(current_left, left)
        overlap_right = min(current_right, right)
        if overlap_left < overlap_right:
            covered += overlap_right - overlap_left
        current = int(segment_next[current])
    return covered


@njit(cache=True)
def _subtract_active_interval(
    node_id: int,
    left: float,
    right: float,
    lineage_segment_head: np.ndarray,
    segment_left: np.ndarray,
    segment_right: np.ndarray,
    segment_next: np.ndarray,
    pool_high_water: int,
    free_head: int,
    free_count: int,
    active_count: int,
    segment_count: int,
    strict: bool,
) -> tuple[int, int, int, int, int, int]:
    if left >= right:
        return (
            pool_high_water,
            free_head,
            free_count,
            active_count,
            segment_count,
            0,
        )
    if strict:
        covered = _active_interval_coverage(
            node_id,
            left,
            right,
            lineage_segment_head,
            segment_left,
            segment_right,
            segment_next,
        )
        target = right - left
        if abs(covered - target) > 1e-8 + 1e-5 * abs(target):
            return (
                pool_high_water,
                free_head,
                free_count,
                active_count,
                segment_count,
                1,
            )

    previous = -1
    current = int(lineage_segment_head[node_id])
    was_active = current >= 0
    while current >= 0:
        current_left = segment_left[current]
        current_right = segment_right[current]
        if current_left >= right:
            break
        following = int(segment_next[current])
        overlap_left = max(current_left, left)
        overlap_right = min(current_right, right)
        if overlap_left >= overlap_right:
            previous = current
            current = following
            continue

        if overlap_left <= current_left and overlap_right >= current_right:
            if previous < 0:
                lineage_segment_head[node_id] = following
            else:
                segment_next[previous] = following
            free_head, free_count = _pool_release(
                current,
                segment_next,
                free_head,
                free_count,
            )
            segment_count -= 1
            current = following
            continue

        if current_left < overlap_left and overlap_right < current_right:
            segment_right[current] = overlap_left
            (
                right_segment,
                pool_high_water,
                free_head,
                free_count,
            ) = _pool_allocate(
                segment_next,
                pool_high_water,
                free_head,
                free_count,
            )
            if right_segment < 0:
                return (
                    pool_high_water,
                    free_head,
                    free_count,
                    active_count,
                    segment_count,
                    3,
                )
            segment_left[right_segment] = overlap_right
            segment_right[right_segment] = current_right
            segment_next[right_segment] = following
            segment_next[current] = right_segment
            segment_count += 1
            break

        if overlap_left <= current_left:
            segment_left[current] = overlap_right
        else:
            segment_right[current] = overlap_left
        previous = current
        current = following

    if was_active and lineage_segment_head[node_id] < 0:
        active_count -= 1
    return (
        pool_high_water,
        free_head,
        free_count,
        active_count,
        segment_count,
        0,
    )


@njit(cache=True)
def _clear_active_lineage(
    node_id: int,
    lineage_segment_head: np.ndarray,
    segment_next: np.ndarray,
    free_head: int,
    free_count: int,
    active_count: int,
    segment_count: int,
) -> tuple[int, int, int, int]:
    current = int(lineage_segment_head[node_id])
    if current < 0:
        return free_head, free_count, active_count, segment_count
    lineage_segment_head[node_id] = -1
    active_count -= 1
    while current >= 0:
        following = int(segment_next[current])
        free_head, free_count = _pool_release(
            current,
            segment_next,
            free_head,
            free_count,
        )
        segment_count -= 1
        current = following
    return free_head, free_count, active_count, segment_count


@njit(cache=True)
def _advance_frontier_range(
    start_step: int,
    end_step: int,
    event_node1: np.ndarray,
    event_node2: np.ndarray,
    event_edge_start: np.ndarray,
    revealed_edge_ids: np.ndarray,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    edge_parent: np.ndarray,
    edge_child: np.ndarray,
    lineage_segment_head: np.ndarray,
    segment_left: np.ndarray,
    segment_right: np.ndarray,
    segment_next: np.ndarray,
    pool_high_water: int,
    free_head: int,
    free_count: int,
    active_count: int,
    segment_count: int,
    strict: bool,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    completed_step = start_step
    for event_index in range(start_step, end_step):
        node1 = int(event_node1[event_index])
        node2 = int(event_node2[event_index])
        if strict and (
            lineage_segment_head[node1] >= 0
            or (node2 >= 0 and lineage_segment_head[node2] >= 0)
        ):
            return (
                completed_step,
                pool_high_water,
                free_head,
                free_count,
                active_count,
                segment_count,
                2,
                event_index,
                -1,
            )

        edge_start = int(event_edge_start[event_index])
        edge_end = int(event_edge_start[event_index + 1])
        for edge_pos in range(edge_start, edge_end):
            edge_id = int(revealed_edge_ids[edge_pos])
            (
                pool_high_water,
                free_head,
                free_count,
                active_count,
                segment_count,
                error_code,
            ) = _subtract_active_interval(
                int(edge_child[edge_id]),
                float(edge_left[edge_id]),
                float(edge_right[edge_id]),
                lineage_segment_head,
                segment_left,
                segment_right,
                segment_next,
                pool_high_water,
                free_head,
                free_count,
                active_count,
                segment_count,
                strict,
            )
            if error_code != 0:
                for rollback_pos in range(edge_start, edge_pos):
                    rollback_edge = int(revealed_edge_ids[rollback_pos])
                    (
                        pool_high_water,
                        free_head,
                        free_count,
                        active_count,
                        segment_count,
                        rollback_error,
                    ) = _add_active_interval(
                        int(edge_child[rollback_edge]),
                        float(edge_left[rollback_edge]),
                        float(edge_right[rollback_edge]),
                        lineage_segment_head,
                        segment_left,
                        segment_right,
                        segment_next,
                        pool_high_water,
                        free_head,
                        free_count,
                        active_count,
                        segment_count,
                    )
                    if rollback_error != 0:
                        error_code = rollback_error
                        break
                return (
                    completed_step,
                    pool_high_water,
                    free_head,
                    free_count,
                    active_count,
                    segment_count,
                    error_code,
                    event_index,
                    edge_id,
                )

        for edge_pos in range(edge_start, edge_end):
            edge_id = int(revealed_edge_ids[edge_pos])
            (
                pool_high_water,
                free_head,
                free_count,
                active_count,
                segment_count,
                error_code,
            ) = _add_active_interval(
                int(edge_parent[edge_id]),
                float(edge_left[edge_id]),
                float(edge_right[edge_id]),
                lineage_segment_head,
                segment_left,
                segment_right,
                segment_next,
                pool_high_water,
                free_head,
                free_count,
                active_count,
                segment_count,
            )
            if error_code != 0:
                return (
                    completed_step,
                    pool_high_water,
                    free_head,
                    free_count,
                    active_count,
                    segment_count,
                    error_code,
                    event_index,
                    edge_id,
                )
        completed_step = event_index + 1
    return (
        completed_step,
        pool_high_water,
        free_head,
        free_count,
        active_count,
        segment_count,
        0,
        -1,
        -1,
    )


@njit(cache=True)
def _backtrack_frontier_range(
    start_step: int,
    end_step: int,
    event_node1: np.ndarray,
    event_node2: np.ndarray,
    event_edge_start: np.ndarray,
    revealed_edge_ids: np.ndarray,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    edge_parent: np.ndarray,
    edge_child: np.ndarray,
    lineage_segment_head: np.ndarray,
    segment_left: np.ndarray,
    segment_right: np.ndarray,
    segment_next: np.ndarray,
    pool_high_water: int,
    free_head: int,
    free_count: int,
    active_count: int,
    segment_count: int,
    strict: bool,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    completed_step = start_step
    for event_index in range(start_step - 1, end_step - 1, -1):
        node1 = int(event_node1[event_index])
        node2 = int(event_node2[event_index])
        edge_start = int(event_edge_start[event_index])
        edge_end = int(event_edge_start[event_index + 1])
        if strict and edge_start < edge_end:
            node1_has_edges = False
            node2_has_edges = False
            for edge_pos in range(edge_start, edge_end):
                edge_id = int(revealed_edge_ids[edge_pos])
                parent = int(edge_parent[edge_id])
                node1_has_edges = node1_has_edges or parent == node1
                node2_has_edges = node2_has_edges or parent == node2
            if (
                (node1_has_edges and lineage_segment_head[node1] < 0)
                or (
                    node2 >= 0
                    and node2_has_edges
                    and lineage_segment_head[node2] < 0
                )
            ):
                return (
                    completed_step,
                    pool_high_water,
                    free_head,
                    free_count,
                    active_count,
                    segment_count,
                    4,
                    event_index,
                    -1,
                )

        free_head, free_count, active_count, segment_count = _clear_active_lineage(
            node1,
            lineage_segment_head,
            segment_next,
            free_head,
            free_count,
            active_count,
            segment_count,
        )
        if node2 >= 0:
            free_head, free_count, active_count, segment_count = _clear_active_lineage(
                node2,
                lineage_segment_head,
                segment_next,
                free_head,
                free_count,
                active_count,
                segment_count,
            )

        for edge_pos in range(edge_start, edge_end):
            edge_id = int(revealed_edge_ids[edge_pos])
            (
                pool_high_water,
                free_head,
                free_count,
                active_count,
                segment_count,
                error_code,
            ) = _add_active_interval(
                int(edge_child[edge_id]),
                float(edge_left[edge_id]),
                float(edge_right[edge_id]),
                lineage_segment_head,
                segment_left,
                segment_right,
                segment_next,
                pool_high_water,
                free_head,
                free_count,
                active_count,
                segment_count,
            )
            if error_code != 0:
                return (
                    completed_step,
                    pool_high_water,
                    free_head,
                    free_count,
                    active_count,
                    segment_count,
                    error_code,
                    event_index,
                    edge_id,
                )
        completed_step = event_index
    return (
        completed_step,
        pool_high_water,
        free_head,
        free_count,
        active_count,
        segment_count,
        0,
        -1,
        -1,
    )


@njit(cache=True)
def _materialize_compact_frontier(
    revealed_node_ids: np.ndarray,
    node_end: int,
    lineage_segment_head: np.ndarray,
    segment_left: np.ndarray,
    segment_right: np.ndarray,
    segment_next: np.ndarray,
    active_count: int,
    segment_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    node_ids = np.empty(active_count, dtype=np.int32)
    offsets = np.empty(active_count + 1, dtype=np.int64)
    output_left = np.empty(segment_count, dtype=np.float64)
    output_right = np.empty(segment_count, dtype=np.float64)
    offsets[0] = 0
    lineage_out = 0
    segment_out = 0
    for reveal_index in range(node_end):
        node_id = int(revealed_node_ids[reveal_index])
        current = int(lineage_segment_head[node_id])
        if current < 0:
            continue
        if lineage_out < active_count:
            node_ids[lineage_out] = node_id
        while current >= 0:
            if segment_out < segment_count:
                output_left[segment_out] = segment_left[current]
                output_right[segment_out] = segment_right[current]
            segment_out += 1
            current = int(segment_next[current])
        lineage_out += 1
        if lineage_out <= active_count:
            offsets[lineage_out] = segment_out
    return (
        node_ids,
        offsets,
        output_left,
        output_right,
        lineage_out,
        segment_out,
    )


@njit(cache=True)
def _first_bad_edge_time(
    node_time: np.ndarray,
    edge_parent: np.ndarray,
    edge_child: np.ndarray,
) -> int:
    for edge_id in range(edge_parent.size):
        if node_time[edge_parent[edge_id]] <= node_time[edge_child[edge_id]]:
            return edge_id
    return -1


@njit(cache=True)
def _build_parent_edge_start(edge_parent: np.ndarray, num_nodes: int) -> np.ndarray:
    parent_edge_start = np.zeros(num_nodes + 1, dtype=np.int32)
    for edge_id in range(edge_parent.size):
        parent_edge_start[edge_parent[edge_id] + 1] += 1

    total = 0
    for node_id in range(num_nodes):
        count = parent_edge_start[node_id + 1]
        parent_edge_start[node_id] = total
        total += count
    parent_edge_start[num_nodes] = total
    return parent_edge_start


@njit(cache=True)
def _fill_parent_edge_ids(
    edge_parent: np.ndarray,
    parent_edge_start: np.ndarray,
) -> np.ndarray:
    parent_edge_ids = np.empty(edge_parent.size, dtype=np.int32)
    cursor = parent_edge_start[:-1].copy()
    for edge_id in range(edge_parent.size):
        parent = edge_parent[edge_id]
        out = cursor[parent]
        parent_edge_ids[out] = edge_id
        cursor[parent] = out + 1
    return parent_edge_ids


@njit(cache=True)
def _build_child_summary_by_node(
    parent_edge_start: np.ndarray,
    parent_edge_ids: np.ndarray,
    edge_child: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    num_nodes = parent_edge_start.size - 1
    child_count = np.zeros(num_nodes, dtype=np.uint8)
    first_child = np.full(num_nodes, -1, dtype=np.int32)

    for node_id in range(num_nodes):
        start = parent_edge_start[node_id]
        end = parent_edge_start[node_id + 1]
        if start == end:
            continue
        child = edge_child[parent_edge_ids[start]]
        first_child[node_id] = child
        child_count[node_id] = 1
        for pos in range(start + 1, end):
            if edge_child[parent_edge_ids[pos]] != child:
                child_count[node_id] = 2
                break
    return child_count, first_child


@njit(cache=True)
def _first_invalid_recombination_node(
    recombination_nodes: np.ndarray,
    child_count: np.ndarray,
) -> int:
    for i in range(recombination_nodes.size):
        node_id = recombination_nodes[i]
        if child_count[node_id] != 1:
            return node_id
    return -1


@njit(cache=True)
def _first_bad_recombination_pair(
    recombination_nodes: np.ndarray,
    node_time: np.ndarray,
    first_child: np.ndarray,
) -> int:
    for i in range(0, recombination_nodes.size, 2):
        left = recombination_nodes[i]
        right = recombination_nodes[i + 1]
        if node_time[left] != node_time[right] or first_child[left] != first_child[right]:
            return i // 2
    return -1


@njit(cache=True)
def _prefix_sum_int32(counts: np.ndarray) -> np.ndarray:
    offsets = np.empty(counts.size + 1, dtype=np.int32)
    total = 0
    offsets[0] = 0
    for i in range(counts.size):
        total += counts[i]
        offsets[i + 1] = total
    return offsets


@njit(cache=True)
def _fill_fast_event_edges(
    event_node1: np.ndarray,
    event_node2: np.ndarray,
    event_edge_start: np.ndarray,
    parent_edge_start: np.ndarray,
    parent_edge_ids: np.ndarray,
    revealed_edge_ids: np.ndarray,
) -> None:
    for event_index in range(event_node1.size):
        node1 = event_node1[event_index]
        node2 = event_node2[event_index]
        out = event_edge_start[event_index]
        start1 = parent_edge_start[node1]
        end1 = parent_edge_start[node1 + 1]

        if node2 < 0:
            for pos in range(start1, end1):
                revealed_edge_ids[out] = parent_edge_ids[pos]
                out += 1
            continue

        start2 = parent_edge_start[node2]
        end2 = parent_edge_start[node2 + 1]
        pos1 = start1
        pos2 = start2
        while pos1 < end1 and pos2 < end2:
            edge1 = parent_edge_ids[pos1]
            edge2 = parent_edge_ids[pos2]
            if edge1 <= edge2:
                revealed_edge_ids[out] = edge1
                pos1 += 1
            else:
                revealed_edge_ids[out] = edge2
                pos2 += 1
            out += 1
        while pos1 < end1:
            revealed_edge_ids[out] = parent_edge_ids[pos1]
            pos1 += 1
            out += 1
        while pos2 < end2:
            revealed_edge_ids[out] = parent_edge_ids[pos2]
            pos2 += 1
            out += 1


@njit(cache=True)
def _build_fast_node_reveal_arrays(
    sample_nodes: np.ndarray,
    event_node1: np.ndarray,
    event_node2: np.ndarray,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    event_node_total = event_node1.size
    for event_index in range(event_node2.size):
        if event_node2[event_index] >= 0:
            event_node_total += 1

    revealed_node_ids = np.empty(sample_nodes.size + event_node_total, dtype=np.int32)
    visible_node_end = np.empty(event_node1.size + 1, dtype=np.int32)
    node_reveal_step = np.full(num_nodes, -1, dtype=np.int32)
    duplicate_node = -1

    cursor = 0
    for i in range(sample_nodes.size):
        node_id = sample_nodes[i]
        if node_reveal_step[node_id] >= 0 and duplicate_node < 0:
            duplicate_node = node_id
        node_reveal_step[node_id] = 0
        revealed_node_ids[cursor] = node_id
        cursor += 1
    visible_node_end[0] = cursor

    for event_index in range(event_node1.size):
        step = event_index + 1
        node1 = event_node1[event_index]
        if node_reveal_step[node1] >= 0 and duplicate_node < 0:
            duplicate_node = node1
        node_reveal_step[node1] = step
        revealed_node_ids[cursor] = node1
        cursor += 1

        node2 = event_node2[event_index]
        if node2 >= 0:
            if node_reveal_step[node2] >= 0 and duplicate_node < 0:
                duplicate_node = node2
            node_reveal_step[node2] = step
            revealed_node_ids[cursor] = node2
            cursor += 1
        visible_node_end[step] = cursor

    return revealed_node_ids, visible_node_end, node_reveal_step, duplicate_node


def _index_edges_by_parent(edge_parent: np.ndarray) -> dict[int, tuple[int, ...]]:
    grouped: dict[int, list[int]] = {}
    for edge_id, parent in enumerate(edge_parent):
        grouped.setdefault(int(parent), []).append(int(edge_id))
    return {parent: tuple(edge_ids) for parent, edge_ids in grouped.items()}


def _validate_edge_times(
    node_time: np.ndarray,
    edge_parent: np.ndarray,
    edge_child: np.ndarray,
) -> None:
    bad = np.flatnonzero(node_time[edge_parent] <= node_time[edge_child])
    if bad.size:
        edge_id = int(bad[0])
        raise ValueError(
            "all edges must satisfy parent.time > child.time; "
            f"edge {edge_id} violates this"
        )


def _validate_genomic_range(
    genomic_range: tuple[float, float],
    sequence_length: float,
) -> tuple[float, float]:
    start, end = float(genomic_range[0]), float(genomic_range[1])
    if not (0 <= start < end <= float(sequence_length)):
        raise ValueError(
            f"genomic_range must satisfy 0 <= start < end <= {sequence_length}"
        )
    return start, end


def _clone_active_segments(active: Mapping[int, tuple[Segment, ...]]) -> ActiveSegmentMap:
    return {int(node_id): tuple(segments) for node_id, segments in active.items()}


def _merge_segments(segments: tuple[Segment, ...]) -> tuple[Segment, ...]:
    cleaned = [
        (float(left), float(right))
        for left, right in segments
        if float(left) < float(right)
    ]
    if not cleaned:
        return ()
    cleaned.sort()
    merged = [cleaned[0]]
    for left, right in cleaned[1:]:
        prev_left, prev_right = merged[-1]
        if left <= prev_right:
            merged[-1] = (prev_left, max(prev_right, right))
        else:
            merged.append((left, right))
    return tuple(merged)


def _subtract_segment(
    segments: tuple[Segment, ...],
    left: float,
    right: float,
    *,
    strict: bool,
) -> tuple[Segment, ...]:
    left = float(left)
    right = float(right)
    if left >= right:
        return tuple(segments)

    covered = 0.0
    updated: list[Segment] = []
    for seg_left, seg_right in segments:
        overlap_left = max(seg_left, left)
        overlap_right = min(seg_right, right)
        if overlap_left < overlap_right:
            covered += overlap_right - overlap_left
            if seg_left < overlap_left:
                updated.append((seg_left, overlap_left))
            if overlap_right < seg_right:
                updated.append((overlap_right, seg_right))
        else:
            updated.append((seg_left, seg_right))

    if strict and not np.isclose(covered, right - left):
        raise ValueError(
            "cannot subtract inactive material segment "
            f"[{left}, {right}); covered {covered}"
        )
    return _merge_segments(tuple(updated))


def _segments_overlap(
    segments: tuple[Segment, ...],
    start: float | None,
    end: float | None,
) -> bool:
    if start is None or end is None:
        return bool(segments)
    return any(seg_left < end and seg_right > start for seg_left, seg_right in segments)
