"""Two-stage discovery and exact verification of closed ancestral cones.

The normal tree-sequence stage is deliberately conservative: topology-derived
intervals are useful diagnostics, while the exhaustive normal-breakpoint
catalog supplies the correctness guarantee.  Exact closure is always decided
on the deterministic balanced synthetic/full ARG.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import tskit

from .synthetic_full_arg import SyntheticFullARGResult, build_synthetic_full_arg
from .trace import (
    EVENT_KIND_COALESCENCE,
    EVENT_KIND_RECOMBINATION,
    FastARGState,
    FastARGTrace,
    build_fast_trace_from_full_arg,
)


DEFAULT_ADJACENCY_TIERS = (1, 2, 4, 8, 16, 32)
Interval = tuple[float, float]


@dataclass(frozen=True)
class NormalTSCandidate:
    """One candidate interval generated from normal-TS breakpoints."""

    left: float
    right: float
    span: float
    smallest_adjacency_tier: int | None
    topology_generated: bool
    exhaustive_only: bool
    youngest_observed_normal_time: float | None
    oldest_observed_normal_time: float | None
    topology_observation_count: int

    @property
    def interval(self) -> Interval:
        return self.left, self.right


@dataclass(frozen=True)
class NormalTSCandidateCatalog:
    """Deduplicated normal-TS candidates and cumulative tier indexes."""

    sequence_length: float
    breakpoints: tuple[float, ...]
    adjacency_tiers: tuple[int, ...]
    candidates: Mapping[Interval, NormalTSCandidate]
    topology_intervals_by_tier: Mapping[int, frozenset[Interval]]
    exhaustive_intervals: frozenset[Interval]
    generation_seconds: float

    def intervals_for_tier(self, tier: int) -> frozenset[Interval]:
        """Return topology candidates available at the requested tier."""

        tier = int(tier)
        if tier not in self.topology_intervals_by_tier:
            raise KeyError(f"unknown adjacency tier {tier}")
        return self.topology_intervals_by_tier[tier]

    @property
    def tier_counts(self) -> dict[int, int]:
        return {
            tier: len(intervals)
            for tier, intervals in self.topology_intervals_by_tier.items()
        }

    @property
    def exhaustive_count(self) -> int:
        return len(self.exhaustive_intervals)


@dataclass(frozen=True)
class ExactConeScanResult:
    """Output of one all-cut exact synthetic-ARG scan."""

    cones: tuple[dict[str, Any], ...]
    per_cut_component_catalog: tuple[dict[str, Any], ...] | None
    per_cut_summary: tuple[dict[str, Any], ...]
    raw_exact_candidate_count: int
    retained_raw_exact_candidate_count: int
    scan_seconds: float


@dataclass(frozen=True)
class TwoStageExactConeEvaluation:
    """Complete normal-candidate and synthetic-oracle comparison."""

    normal_tree_sequence: tskit.TreeSequence
    candidate_catalog: NormalTSCandidateCatalog
    synthetic_conversion: SyntheticFullARGResult
    synthetic_arg: tskit.TreeSequence
    trace: FastARGTrace
    exact_scan: ExactConeScanResult
    exact_verified_cones: tuple[dict[str, Any], ...]
    recall_by_tier: Mapping[int | str, float]
    candidate_count_by_tier: Mapping[int | str, int]
    missed_intervals_by_tier: Mapping[int | str, tuple[Interval, ...]]
    timings: Mapping[str, float]


class _AtomDSU:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(int(size), dtype=np.int32)
        self.size = np.ones(int(size), dtype=np.int32)

    def find(self, item: int) -> int:
        item = int(item)
        root = item
        while int(self.parent[root]) != root:
            root = int(self.parent[root])
        while int(self.parent[item]) != item:
            following = int(self.parent[item])
            self.parent[item] = root
            item = following
        return root

    def union(self, left: int, right: int) -> int:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        if int(self.size[left_root]) < int(self.size[right_root]):
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]
        return left_root


def _load_tree_sequence(
    ts_or_path: str | Path | tskit.TreeSequence,
) -> tskit.TreeSequence:
    if isinstance(ts_or_path, tskit.TreeSequence):
        return ts_or_path
    return tskit.load(str(Path(ts_or_path).expanduser()))


def _validate_adjacency_tiers(values: Iterable[int]) -> tuple[int, ...]:
    tiers = tuple(sorted(set(int(value) for value in values)))
    if not tiers or tiers[0] != 1 or any(value <= 0 for value in tiers):
        raise ValueError("adjacency tiers must be positive and include tier 1")
    return tiers


def _contiguous_atom_components(dsu: _AtomDSU, atom_count: int) -> list[tuple[int, int]]:
    roots = [dsu.find(index) for index in range(int(atom_count))]
    components: list[tuple[int, int]] = []
    start = 0
    for index in range(1, int(atom_count) + 1):
        if index == atom_count or roots[index] != roots[start]:
            components.append((start, index))
            start = index
    seen = set()
    for start, _ in components:
        root = roots[start]
        if root in seen:
            raise AssertionError("normal-edge connectivity produced a noncontiguous component")
        seen.add(root)
    return components


def generate_normal_ts_candidates(
    ts_or_path: str | Path | tskit.TreeSequence,
    *,
    adjacency_tiers: Sequence[int] = DEFAULT_ADJACENCY_TIERS,
    exhaustive_fallback: bool = True,
) -> NormalTSCandidateCatalog:
    """Generate conservative interval candidates directly from a normal TS.

    Edges are activated from oldest parent time to youngest, with tied times
    processed as a batch.  Both sides of every batch are observed.  At each
    snapshot, the function emits individual contiguous components and unions
    of adjacent components up to the largest requested tier.
    """

    started = time.perf_counter()
    ts = _load_tree_sequence(ts_or_path)
    tiers = _validate_adjacency_tiers(adjacency_tiers)
    breakpoints = np.asarray(list(ts.breakpoints()), dtype=np.float64)
    if breakpoints.size < 2 or np.any(np.diff(breakpoints) <= 0):
        raise ValueError("tree-sequence breakpoints must be strictly increasing")
    if breakpoints[0] != 0.0 or breakpoints[-1] != float(ts.sequence_length):
        raise ValueError("tree-sequence breakpoints must span the chromosome")

    atom_count = int(breakpoints.size - 1)
    endpoint_index = {float(value): index for index, value in enumerate(breakpoints)}
    dsu = _AtomDSU(atom_count)
    mutable: dict[Interval, dict[str, Any]] = {}

    def record(interval: Interval, tier: int, observed_time: float) -> None:
        left, right = interval
        if not left < right:
            return
        if left == 0.0 and right == float(ts.sequence_length):
            return
        item = mutable.get(interval)
        if item is None:
            item = {
                "smallest_adjacency_tier": int(tier),
                "youngest": float(observed_time),
                "oldest": float(observed_time),
                "count": 1,
            }
            mutable[interval] = item
        else:
            item["smallest_adjacency_tier"] = min(
                int(item["smallest_adjacency_tier"]), int(tier)
            )
            item["youngest"] = min(float(item["youngest"]), float(observed_time))
            item["oldest"] = max(float(item["oldest"]), float(observed_time))
            item["count"] += 1

    def observe(observed_time: float) -> None:
        components = _contiguous_atom_components(dsu, atom_count)
        max_width = min(tiers[-1], len(components))
        for width in range(1, max_width + 1):
            tier = next(value for value in tiers if value >= width)
            for start in range(0, len(components) - width + 1):
                left_atom = components[start][0]
                right_atom = components[start + width - 1][1]
                record(
                    (float(breakpoints[left_atom]), float(breakpoints[right_atom])),
                    tier,
                    observed_time,
                )

    edges = ts.tables.edges
    if ts.num_edges:
        parent_time = np.asarray(ts.nodes_time, dtype=np.float64)[
            np.asarray(edges.parent, dtype=np.int64)
        ]
        order = np.argsort(-parent_time, kind="stable")
        sorted_times = parent_time[order]
        position = 0
        while position < order.size:
            batch_time = float(sorted_times[position])
            batch_end = position + 1
            while batch_end < order.size and sorted_times[batch_end] == batch_time:
                batch_end += 1
            observe(batch_time)
            for edge_id in order[position:batch_end]:
                left = float(edges.left[int(edge_id)])
                right = float(edges.right[int(edge_id)])
                try:
                    left_atom = endpoint_index[left]
                    right_atom = endpoint_index[right]
                except KeyError as error:
                    raise AssertionError("edge endpoint is not a TS breakpoint") from error
                for atom in range(left_atom + 1, right_atom):
                    dsu.union(left_atom, atom)
            observe(batch_time)
            position = batch_end
    else:
        observe(0.0)

    topology_by_tier = {
        tier: frozenset(
            interval
            for interval, item in mutable.items()
            if int(item["smallest_adjacency_tier"]) <= tier
        )
        for tier in tiers
    }

    if exhaustive_fallback:
        exhaustive = {
            (float(breakpoints[left]), float(breakpoints[right]))
            for left in range(breakpoints.size - 1)
            for right in range(left + 1, breakpoints.size)
            if not (left == 0 and right == breakpoints.size - 1)
        }
    else:
        exhaustive = set(topology_by_tier[tiers[-1]])

    candidates: dict[Interval, NormalTSCandidate] = {}
    for interval in sorted(exhaustive | set(mutable)):
        item = mutable.get(interval)
        candidates[interval] = NormalTSCandidate(
            left=interval[0],
            right=interval[1],
            span=interval[1] - interval[0],
            smallest_adjacency_tier=(
                None if item is None else int(item["smallest_adjacency_tier"])
            ),
            topology_generated=item is not None,
            exhaustive_only=item is None,
            youngest_observed_normal_time=(
                None if item is None else float(item["youngest"])
            ),
            oldest_observed_normal_time=(
                None if item is None else float(item["oldest"])
            ),
            topology_observation_count=(0 if item is None else int(item["count"])),
        )

    return NormalTSCandidateCatalog(
        sequence_length=float(ts.sequence_length),
        breakpoints=tuple(float(value) for value in breakpoints),
        adjacency_tiers=tiers,
        candidates=candidates,
        topology_intervals_by_tier=topology_by_tier,
        exhaustive_intervals=frozenset(exhaustive),
        generation_seconds=time.perf_counter() - started,
    )


def uf_find(parent: np.ndarray, item: int) -> int:
    item = int(item)
    root = item
    while int(parent[root]) != root:
        root = int(parent[root])
    while int(parent[item]) != item:
        following = int(parent[item])
        parent[item] = root
        item = following
    return root


def uf_union(parent: np.ndarray, sizes: np.ndarray, left_item: int, right_item: int) -> int:
    left_root = uf_find(parent, left_item)
    right_root = uf_find(parent, right_item)
    if left_root == right_root:
        return left_root
    if int(sizes[left_root]) < int(sizes[right_root]):
        left_root, right_root = right_root, left_root
    parent[right_root] = left_root
    sizes[left_root] += sizes[right_root]
    return left_root


def initial_material_components(
    segment_offsets: np.ndarray,
    segment_left: np.ndarray,
    segment_right: np.ndarray,
) -> dict[str, np.ndarray]:
    segment_offsets = np.asarray(segment_offsets, dtype=np.int64)
    segment_left = np.asarray(segment_left, dtype=np.float64)
    segment_right = np.asarray(segment_right, dtype=np.float64)
    if segment_offsets.ndim != 1 or segment_offsets.size == 0:
        raise ValueError("segment_offsets must be a nonempty vector")
    if segment_left.shape != segment_right.shape or segment_left.ndim != 1:
        raise ValueError("segment arrays must be matching vectors")
    if int(segment_offsets[0]) != 0 or int(segment_offsets[-1]) != segment_left.size:
        raise ValueError("segment_offsets do not span the segment arrays")
    if np.any(segment_left >= segment_right):
        raise ValueError("all material intervals must be nonempty")

    lineage_count = segment_offsets.size - 1
    parent = np.arange(lineage_count, dtype=np.int32)
    sizes = np.ones(lineage_count, dtype=np.int32)
    segment_owner = np.empty(segment_left.size, dtype=np.int32)
    for lineage_index in range(lineage_count):
        start = int(segment_offsets[lineage_index])
        end = int(segment_offsets[lineage_index + 1])
        segment_owner[start:end] = lineage_index

    order = np.argsort(segment_left, kind="stable")
    run_owner = -1
    run_right = -np.inf
    for segment_index in order:
        owner = int(segment_owner[segment_index])
        left = float(segment_left[segment_index])
        right = float(segment_right[segment_index])
        if run_owner < 0 or left >= run_right:
            run_owner = owner
            run_right = right
        else:
            uf_union(parent, sizes, run_owner, owner)
            run_right = max(run_right, right)

    lineage_roots = np.asarray(
        [uf_find(parent, index) for index in range(lineage_count)], dtype=np.int32
    )
    return {
        "parent": parent,
        "sizes": sizes,
        "segment_owner": segment_owner,
        "lineage_roots": lineage_roots,
    }


def close_components_through_suffix(
    frontier_node_ids: np.ndarray,
    cut_step: int,
    event_node1: np.ndarray,
    event_node2: np.ndarray,
    event_edge_start: np.ndarray,
    revealed_edge_ids: np.ndarray,
    edge_child: np.ndarray,
    node_count: int,
    parent: np.ndarray,
    sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frontier_node_ids = np.asarray(frontier_node_ids, dtype=np.int32)
    node_component = np.full(int(node_count), -1, dtype=np.int32)
    for lineage_index, node_id in enumerate(frontier_node_ids):
        node_component[int(node_id)] = lineage_index

    event_roots = np.full(len(event_node1) - int(cut_step), -1, dtype=np.int32)
    for event_index in range(int(cut_step), len(event_node1)):
        edge_start = int(event_edge_start[event_index])
        edge_end = int(event_edge_start[event_index + 1])
        if edge_start == edge_end:
            continue
        event_root = -1
        for edge_position in range(edge_start, edge_end):
            edge_id = int(revealed_edge_ids[edge_position])
            child_id = int(edge_child[edge_id])
            child_component = int(node_component[child_id])
            if child_component < 0:
                raise RuntimeError(
                    f"event {event_index} references unassigned child node {child_id}"
                )
            if event_root < 0:
                event_root = uf_find(parent, child_component)
            else:
                event_root = uf_union(parent, sizes, event_root, child_component)

        for node_id in (int(event_node1[event_index]), int(event_node2[event_index])):
            if node_id < 0:
                continue
            previous = int(node_component[node_id])
            if previous >= 0:
                event_root = uf_union(parent, sizes, event_root, previous)
            node_component[node_id] = event_root
        event_roots[event_index - int(cut_step)] = event_root

    frontier_roots = np.asarray(
        [uf_find(parent, index) for index in range(frontier_node_ids.size)],
        dtype=np.int32,
    )
    for offset, root in enumerate(event_roots):
        if root >= 0:
            event_roots[offset] = uf_find(parent, int(root))
    return parent, sizes, frontier_roots, event_roots, node_component


def canonical_component_intervals(
    component_count: int,
    segment_component: np.ndarray,
    segment_left: np.ndarray,
    segment_right: np.ndarray,
) -> list[tuple[Interval, ...]]:
    buckets: list[list[Interval]] = [[] for _ in range(int(component_count))]
    for component, left, right in zip(segment_component, segment_left, segment_right):
        buckets[int(component)].append((float(left), float(right)))
    output = []
    for intervals in buckets:
        merged: list[Interval] = []
        for left, right in sorted(intervals):
            if merged and left <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], right))
            else:
                merged.append((left, right))
        output.append(tuple(merged))
    return output


def event_edge_ids(local_trace: FastARGTrace, event_index: int) -> np.ndarray:
    start = int(local_trace.event_edge_start[int(event_index)])
    end = int(local_trace.event_edge_start[int(event_index) + 1])
    return np.asarray(local_trace.revealed_edge_ids[start:end], dtype=np.int32)


def grouped_values(
    values: np.ndarray, labels: np.ndarray, component_count: int
) -> list[list[int]]:
    groups: list[list[int]] = [[] for _ in range(int(component_count))]
    for value, label in zip(values, labels):
        label = int(label)
        if label < 0 or label >= component_count:
            raise AssertionError(f"invalid component label {label}")
        groups[label].append(int(value))
    return groups


def exact_components_at_cut(
    state: FastARGState,
    cut_step: int,
    reference_terminal_frontier: Any,
) -> dict[str, Any]:
    """Classify every older-suffix connectivity component at one event cut."""

    local_trace = state.trace
    cut_step = int(cut_step)
    if state.step != cut_step:
        raise ValueError(f"state is at step {state.step}, expected {cut_step}")
    frontier = state.compact_active_frontier()
    if frontier.node_ids.size == 0 or frontier.segment_count == 0:
        raise RuntimeError(f"cut step {cut_step} has an empty frontier")

    initial = initial_material_components(
        frontier.segment_offsets, frontier.segment_left, frontier.segment_right
    )
    parent, sizes, frontier_roots, event_roots, node_component = (
        close_components_through_suffix(
            frontier.node_ids,
            cut_step,
            local_trace.event_node1,
            local_trace.event_node2,
            local_trace.event_edge_start,
            local_trace.revealed_edge_ids,
            local_trace.edge_child,
            local_trace.node_time.size,
            initial["parent"],
            initial["sizes"],
        )
    )

    final_roots = np.unique(frontier_roots)
    root_to_component = {int(root): index for index, root in enumerate(final_roots)}
    frontier_components = np.asarray(
        [root_to_component[uf_find(parent, int(root))] for root in frontier_roots],
        dtype=np.int32,
    )

    suffix_event_indices = np.arange(cut_step, local_trace.num_steps, dtype=np.int64)
    suffix_edgeful = (
        np.diff(local_trace.event_edge_start[cut_step:]) > 0
        if cut_step < local_trace.num_steps
        else np.empty(0, dtype=bool)
    )
    event_components = np.full(event_roots.size, -1, dtype=np.int32)
    for offset, root in enumerate(event_roots):
        if root >= 0:
            event_components[offset] = root_to_component[uf_find(parent, int(root))]
    if not np.array_equal(event_components >= 0, suffix_edgeful):
        raise AssertionError("every edgeful suffix event must have one component")

    terminal_components = []
    for node_id in reference_terminal_frontier.node_ids:
        seed = int(node_component[int(node_id)])
        if seed < 0:
            raise RuntimeError(f"terminal node {int(node_id)} has no component")
        terminal_components.append(root_to_component[uf_find(parent, seed)])
    terminal_components = np.asarray(terminal_components, dtype=np.int32)

    segment_components = frontier_components[initial["segment_owner"]]
    intervals_by_component = canonical_component_intervals(
        len(final_roots),
        segment_components,
        frontier.segment_left,
        frontier.segment_right,
    )
    lower_nodes = grouped_values(frontier.node_ids, frontier_components, len(final_roots))
    terminal_nodes = grouped_values(
        reference_terminal_frontier.node_ids,
        terminal_components,
        len(final_roots),
    )
    events_by_component = grouped_values(
        suffix_event_indices[suffix_edgeful],
        event_components[suffix_edgeful],
        len(final_roots),
    )

    separation_event = None if cut_step == 0 else local_trace.event_at_index(cut_step - 1)
    candidates = []
    for component_index, intervals in enumerate(intervals_by_component):
        component_events = np.asarray(events_by_component[component_index], dtype=np.int64)
        component_edge_chunks = [
            event_edge_ids(local_trace, event_index) for event_index in component_events
        ]
        component_edge_ids = (
            np.unique(np.concatenate(component_edge_chunks)).astype(np.int32)
            if component_edge_chunks
            else np.empty(0, dtype=np.int32)
        )
        outside_event_indices = suffix_event_indices[
            suffix_edgeful & (event_components != component_index)
        ]
        outside_edge_chunks = [
            event_edge_ids(local_trace, event_index) for event_index in outside_event_indices
        ]
        outside_edge_ids = (
            np.unique(np.concatenate(outside_edge_chunks)).astype(np.int32)
            if outside_edge_chunks
            else np.empty(0, dtype=np.int32)
        )

        left = float(intervals[0][0])
        right = float(intervals[-1][1])
        contiguous = len(intervals) == 1
        proper_subregion = not (
            left <= 0.0 and right >= float(local_trace.sequence_length)
        )
        assigned_edges_inside = bool(
            np.all(local_trace.edge_left[component_edge_ids] >= left)
            and np.all(local_trace.edge_right[component_edge_ids] <= right)
        )
        outside_overlap_edge_ids = outside_edge_ids[
            (local_trace.edge_left[outside_edge_ids] < right)
            & (local_trace.edge_right[outside_edge_ids] > left)
        ]

        reasons = []
        if not contiguous:
            reasons.append("noncontiguous")
        if not proper_subregion:
            reasons.append("whole_sequence")
        if component_events.size == 0:
            reasons.append("zero_event")
        if not assigned_edges_inside:
            reasons.append("assigned_edge_outside_region")
        if outside_overlap_edge_ids.size:
            reasons.append("outside_suffix_edge_overlap")

        component_lineages = np.flatnonzero(frontier_components == component_index)
        frontier_segments = []
        for lineage_index in component_lineages:
            start = int(frontier.segment_offsets[lineage_index])
            end = int(frontier.segment_offsets[lineage_index + 1])
            for segment_index in range(start, end):
                frontier_segments.append(
                    (
                        int(frontier.node_ids[lineage_index]),
                        float(frontier.segment_left[segment_index]),
                        float(frontier.segment_right[segment_index]),
                    )
                )

        event_node_values = []
        for event_index in component_events:
            for node_id in (
                int(local_trace.event_node1[event_index]),
                int(local_trace.event_node2[event_index]),
            ):
                if node_id >= 0:
                    event_node_values.append(node_id)
        node_values = list(lower_nodes[component_index])
        node_values.extend(terminal_nodes[component_index])
        node_values.extend(event_node_values)
        if component_edge_ids.size:
            node_values.extend(local_trace.edge_parent[component_edge_ids].tolist())
            node_values.extend(local_trace.edge_child[component_edge_ids].tolist())
        component_node_ids = tuple(sorted(set(int(value) for value in node_values)))

        event_kinds = local_trace.event_kind[component_events]
        material_length = float(sum(right_value - left_value for left_value, right_value in intervals))
        candidates.append(
            {
                "boundary_step": cut_step,
                "boundary_time": float(state.current_time),
                "component_index": component_index,
                "region_key": (left, right) if contiguous else tuple(intervals),
                "intervals": tuple(intervals),
                "left": left,
                "right": right,
                "span": right - left,
                "material_length": material_length,
                "contiguous": contiguous,
                "proper_subregion": proper_subregion,
                "lower_frontier_anchor_node_ids": tuple(lower_nodes[component_index]),
                "frontier_segments": tuple(frontier_segments),
                "suffix_event_indices": tuple(int(value) for value in component_events),
                "edge_ids": tuple(int(value) for value in component_edge_ids),
                "node_ids": component_node_ids,
                "terminal_lineage_ids": tuple(terminal_nodes[component_index]),
                "event_count": int(component_events.size),
                "recombination_event_count": int(
                    np.sum(event_kinds == EVENT_KIND_RECOMBINATION)
                ),
                "coalescence_event_count": int(
                    np.sum(event_kinds == EVENT_KIND_COALESCENCE)
                ),
                "node_count": len(component_node_ids),
                "edge_count": int(component_edge_ids.size),
                "assigned_edges_inside": assigned_edges_inside,
                "outside_overlap_edge_ids": tuple(
                    int(value) for value in outside_overlap_edge_ids
                ),
                "closure_verified": assigned_edges_inside
                and not outside_overlap_edge_ids.size,
                "exact": not reasons,
                "rejection_reasons": tuple(reasons),
                "separation_event_index": None if separation_event is None else cut_step - 1,
                "separation_event_time": (
                    None if separation_event is None else separation_event.time
                ),
                "separation_event_kind": (
                    None if separation_event is None else separation_event.kind
                ),
                "separation_event_node_ids": (
                    () if separation_event is None else separation_event.node_ids
                ),
            }
        )

    candidates.sort(key=lambda item: (item["left"], item["right"]))
    for candidate_index, candidate in enumerate(candidates):
        candidate["candidate_id"] = f"step-{cut_step}-component-{candidate_index:04d}"

    if sum(candidate["event_count"] for candidate in candidates) != int(
        np.sum(suffix_edgeful)
    ):
        raise AssertionError("suffix events do not partition across components")
    exact_intervals = sorted(
        (candidate["left"], candidate["right"])
        for candidate in candidates
        if candidate["exact"]
    )
    for previous, current in zip(exact_intervals, exact_intervals[1:]):
        if current[0] < previous[1]:
            raise AssertionError("exact components overlap at one cut")

    return {
        "boundary_step": cut_step,
        "boundary_time": float(state.current_time),
        "suffix_event_count": int(local_trace.num_steps - cut_step),
        "frontier_lineage_count": int(frontier.node_ids.size),
        "frontier_segment_count": int(frontier.segment_count),
        "component_count": len(candidates),
        "exact_component_count": sum(candidate["exact"] for candidate in candidates),
        "candidates": candidates,
    }


def scan_exact_closed_cones(
    trace: FastARGTrace,
    *,
    candidate_intervals: Iterable[Interval] | None = None,
    retain_per_cut_catalog: bool = False,
) -> ExactConeScanResult:
    """Run the exact all-cut oracle once, streaming diagnostics by default."""

    started = time.perf_counter()
    allowed = None
    if candidate_intervals is not None:
        allowed = frozenset((float(left), float(right)) for left, right in candidate_intervals)
    terminal_state = trace.initial_state().advance_to(trace.num_steps)
    terminal_frontier = terminal_state.compact_active_frontier()
    scan_state = trace.initial_state()
    catalog = [] if retain_per_cut_catalog else None
    per_cut_summary = []
    occurrences: dict[Interval, list[dict[str, Any]]] = {}
    raw_exact_count = 0
    retained_raw_count = 0

    for cut_step in range(trace.num_steps + 1):
        if scan_state.step != cut_step:
            raise AssertionError("scan cursor is not at the requested cut")
        cut_result = exact_components_at_cut(scan_state, cut_step, terminal_frontier)
        if catalog is not None:
            catalog.append(cut_result)
        per_cut_summary.append(
            {
                "boundary_step": cut_result["boundary_step"],
                "boundary_time": cut_result["boundary_time"],
                "suffix_events": cut_result["suffix_event_count"],
                "frontier_lineages": cut_result["frontier_lineage_count"],
                "frontier_segments": cut_result["frontier_segment_count"],
                "components": cut_result["component_count"],
                "exact_nonempty_components": cut_result["exact_component_count"],
            }
        )
        for candidate in cut_result["candidates"]:
            if not candidate["exact"]:
                continue
            raw_exact_count += 1
            interval = (float(candidate["left"]), float(candidate["right"]))
            if allowed is not None and interval not in allowed:
                continue
            retained_raw_count += 1
            occurrences.setdefault(interval, []).append(candidate)
        if cut_step < trace.num_steps:
            scan_state.advance()

    cones = []
    for interval, interval_occurrences in occurrences.items():
        earliest = copy.deepcopy(interval_occurrences[0])
        latest = interval_occurrences[-1]
        earliest["first_closed_step"] = int(earliest["boundary_step"])
        earliest["first_closed_time"] = float(earliest["boundary_time"])
        earliest["last_closed_step"] = int(latest["boundary_step"])
        earliest["last_closed_time"] = float(latest["boundary_time"])
        earliest["valid_cut_count"] = len(interval_occurrences)
        earliest["valid_cut_steps"] = tuple(
            int(item["boundary_step"]) for item in interval_occurrences
        )
        cones.append(earliest)
    cones.sort(key=lambda item: (item["first_closed_step"], item["left"], item["right"]))
    for cone_index, cone in enumerate(cones):
        cone["cone_id"] = f"cone-{cone_index:04d}"

    return ExactConeScanResult(
        cones=tuple(cones),
        per_cut_component_catalog=None if catalog is None else tuple(catalog),
        per_cut_summary=tuple(per_cut_summary),
        raw_exact_candidate_count=raw_exact_count,
        retained_raw_exact_candidate_count=retained_raw_count,
        scan_seconds=time.perf_counter() - started,
    )


def assert_synthetic_endpoints_are_normal_breakpoints(
    normal_ts: tskit.TreeSequence,
    synthetic_ts: tskit.TreeSequence,
) -> None:
    """Verify the endpoint-preservation fact used by the exhaustive fallback."""

    normal_breakpoints = np.asarray(list(normal_ts.breakpoints()), dtype=np.float64)
    synthetic_edges = synthetic_ts.tables.edges
    endpoint_values = np.concatenate(
        (
            np.asarray(synthetic_edges.left, dtype=np.float64),
            np.asarray(synthetic_edges.right, dtype=np.float64),
        )
    )
    if not np.all(np.isin(endpoint_values, normal_breakpoints)):
        missing = np.unique(endpoint_values[~np.isin(endpoint_values, normal_breakpoints)])
        raise AssertionError(
            f"synthetic conversion introduced non-normal breakpoints: {missing.tolist()}"
        )


def evaluate_two_stage_exact_cones(
    ts_or_path: str | Path | tskit.TreeSequence,
    *,
    adjacency_tiers: Sequence[int] = DEFAULT_ADJACENCY_TIERS,
    exhaustive_fallback: bool = True,
    retain_per_cut_catalog: bool = False,
) -> TwoStageExactConeEvaluation:
    """Evaluate normal-TS candidates against the exact synthetic oracle."""

    normal_ts = _load_tree_sequence(ts_or_path)
    candidate_catalog = generate_normal_ts_candidates(
        normal_ts,
        adjacency_tiers=adjacency_tiers,
        exhaustive_fallback=exhaustive_fallback,
    )

    conversion_started = time.perf_counter()
    conversion = build_synthetic_full_arg(
        normal_ts,
        split_rule="balanced",
        ensure_unique_event_times=True,
    )
    synthetic_arg = conversion.tree_sequence
    conversion_seconds = time.perf_counter() - conversion_started
    assert_synthetic_endpoints_are_normal_breakpoints(normal_ts, synthetic_arg)

    trace_started = time.perf_counter()
    trace = build_fast_trace_from_full_arg(
        synthetic_arg,
        require_unique_event_times=True,
    )
    trace_seconds = time.perf_counter() - trace_started
    if trace.num_steps > 1 and not np.all(np.diff(trace.event_time) > 0.0):
        raise AssertionError("synthetic event times must be strictly increasing")

    exact_scan = scan_exact_closed_cones(
        trace,
        retain_per_cut_catalog=retain_per_cut_catalog,
    )
    exact_intervals = frozenset(
        (float(cone["left"]), float(cone["right"])) for cone in exact_scan.cones
    )
    exhaustive_missing = exact_intervals - candidate_catalog.exhaustive_intervals
    if exhaustive_missing:
        raise AssertionError(
            "normal-breakpoint exhaustive candidates missed exact cones: "
            f"{sorted(exhaustive_missing)}"
        )

    verified = []
    for cone in exact_scan.cones:
        attached = copy.deepcopy(cone)
        interval = (float(cone["left"]), float(cone["right"]))
        normal_candidate = candidate_catalog.candidates[interval]
        attached["normal_candidate_smallest_adjacency_tier"] = (
            normal_candidate.smallest_adjacency_tier
        )
        attached["normal_candidate_topology_generated"] = (
            normal_candidate.topology_generated
        )
        attached["normal_candidate_exhaustive_only"] = normal_candidate.exhaustive_only
        attached["normal_candidate_youngest_observed_time"] = (
            normal_candidate.youngest_observed_normal_time
        )
        attached["normal_candidate_oldest_observed_time"] = (
            normal_candidate.oldest_observed_normal_time
        )
        verified.append(attached)

    denominator = len(exact_intervals)
    recall: dict[int | str, float] = {}
    counts: dict[int | str, int] = {}
    missed: dict[int | str, tuple[Interval, ...]] = {}
    for tier in candidate_catalog.adjacency_tiers:
        intervals = candidate_catalog.intervals_for_tier(tier)
        tier_missing = tuple(sorted(exact_intervals - intervals))
        recall[tier] = 1.0 if denominator == 0 else 1.0 - len(tier_missing) / denominator
        counts[tier] = len(intervals)
        missed[tier] = tier_missing
    exhaustive_key = "exhaustive"
    exhaustive_missing_tuple = tuple(sorted(exhaustive_missing))
    recall[exhaustive_key] = (
        1.0 if denominator == 0 else 1.0 - len(exhaustive_missing_tuple) / denominator
    )
    counts[exhaustive_key] = candidate_catalog.exhaustive_count
    missed[exhaustive_key] = exhaustive_missing_tuple

    return TwoStageExactConeEvaluation(
        normal_tree_sequence=normal_ts,
        candidate_catalog=candidate_catalog,
        synthetic_conversion=conversion,
        synthetic_arg=synthetic_arg,
        trace=trace,
        exact_scan=exact_scan,
        exact_verified_cones=tuple(verified),
        recall_by_tier=recall,
        candidate_count_by_tier=counts,
        missed_intervals_by_tier=missed,
        timings={
            "candidate_generation_seconds": candidate_catalog.generation_seconds,
            "synthetic_conversion_seconds": conversion_seconds,
            "trace_construction_seconds": trace_seconds,
            "exact_scan_seconds": exact_scan.scan_seconds,
            "total_seconds": (
                candidate_catalog.generation_seconds
                + conversion_seconds
                + trace_seconds
                + exact_scan.scan_seconds
            ),
        },
    )


__all__ = [
    "DEFAULT_ADJACENCY_TIERS",
    "ExactConeScanResult",
    "NormalTSCandidate",
    "NormalTSCandidateCatalog",
    "TwoStageExactConeEvaluation",
    "assert_synthetic_endpoints_are_normal_breakpoints",
    "canonical_component_intervals",
    "close_components_through_suffix",
    "evaluate_two_stage_exact_cones",
    "event_edge_ids",
    "exact_components_at_cut",
    "generate_normal_ts_candidates",
    "grouped_values",
    "initial_material_components",
    "scan_exact_closed_cones",
    "uf_find",
    "uf_union",
]
