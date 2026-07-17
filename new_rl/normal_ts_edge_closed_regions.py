"""Direct edge-closure discovery on an ordinary tskit tree sequence.

For a normal-tree-sequence cut at time ``t``, an edge is *older* when its
parent is older than ``t``.  This module groups those older edges by their
ancestral node connectivity.  A group is a normal edge-closed region only when
its genomic support is one proper interval and no other older-edge group
overlaps that interval.

This is intentionally a normal-TS first stage.  It does not construct a
synthetic/full ARG and does not assert event-level closure across hidden
recombination structure.
"""

from __future__ import annotations

import copy
import bisect
from dataclasses import dataclass, field
from pathlib import Path
import time
import tracemalloc
from typing import Any, Mapping

import numpy as np
import tskit


Interval = tuple[float, float]


@dataclass(frozen=True)
class NormalEdgeClosedRegionScan:
    """Direct normal-tree edge-closure scan output."""

    regions: tuple[dict[str, Any], ...]
    per_cut_summary: tuple[dict[str, Any], ...]
    per_cut_component_catalog: tuple[dict[str, Any], ...] | None
    raw_closed_component_count: int
    scan_seconds: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalEdgeClosedRegionBenchmark:
    """Warm benchmark data for the reference and incremental scanners."""

    reference_seconds: float
    incremental_seconds: float
    speedup: float
    reference_peak_bytes: int
    incremental_peak_bytes: int
    reference_diagnostics: Mapping[str, Any]
    incremental_diagnostics: Mapping[str, Any]


class _DisjointSet:
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


def _canonical_intervals(left: np.ndarray, right: np.ndarray) -> tuple[Interval, ...]:
    merged: list[Interval] = []
    for interval_left, interval_right in sorted(
        (float(value_left), float(value_right))
        for value_left, value_right in zip(left, right)
    ):
        if not interval_left < interval_right:
            raise ValueError("normal tree-sequence edges must be nonempty")
        if merged and interval_left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval_right))
        else:
            merged.append((interval_left, interval_right))
    return tuple(merged)


def _roots_with_external_overlap(
    active_edge_ids: np.ndarray,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    edge_roots: np.ndarray,
) -> set[int]:
    """Return component roots sharing positive genomic overlap with another root."""

    if active_edge_ids.size < 2:
        return set()
    order = active_edge_ids[
        np.argsort(edge_left[active_edge_ids], kind="stable")
    ]
    dirty: set[int] = set()
    cluster_roots: set[int] = set()
    cluster_right = -np.inf

    def finish_cluster() -> None:
        if len(cluster_roots) > 1:
            dirty.update(cluster_roots)

    for edge_id in order:
        edge_id = int(edge_id)
        left = float(edge_left[edge_id])
        right = float(edge_right[edge_id])
        # Half-open intervals touching at one endpoint do not overlap.
        if cluster_roots and left >= cluster_right:
            finish_cluster()
            cluster_roots = set()
            cluster_right = -np.inf
        cluster_roots.add(int(edge_roots[edge_id]))
        cluster_right = max(cluster_right, right)
    finish_cluster()
    return dirty


def normal_edge_components_at_cut(
    ts_or_path: str | Path | tskit.TreeSequence,
    cut_time: float,
    *,
    cut_index: int | None = None,
) -> dict[str, Any]:
    """Classify all normal-TS older-edge components at one time cut.

    ``cut_time`` is a normal-node time.  The older graph contains precisely
    edges whose parent time is strictly greater than the cut.  Thus an edge
    with parent time equal to the cut is the separation edge immediately below
    the older graph.
    """

    ts = _load_tree_sequence(ts_or_path)
    cut_time = float(cut_time)
    tables = ts.tables
    edge_left = np.asarray(tables.edges.left, dtype=np.float64)
    edge_right = np.asarray(tables.edges.right, dtype=np.float64)
    edge_parent = np.asarray(tables.edges.parent, dtype=np.int32)
    edge_child = np.asarray(tables.edges.child, dtype=np.int32)
    node_time = np.asarray(tables.nodes.time, dtype=np.float64)
    parent_time = node_time[edge_parent]
    child_time = node_time[edge_child]

    active_edge_ids = np.flatnonzero(parent_time > cut_time).astype(np.int32)
    separation_edge_ids = np.flatnonzero(parent_time == cut_time).astype(np.int32)
    if active_edge_ids.size == 0:
        return {
            "cut_index": cut_index,
            "cut_time": cut_time,
            "older_edge_count": 0,
            "frontier_edge_count": 0,
            "component_count": 0,
            "closed_component_count": 0,
            "separation_edge_ids": tuple(int(value) for value in separation_edge_ids),
            "components": [],
        }

    dsu = _DisjointSet(ts.num_nodes)
    for edge_id in active_edge_ids:
        dsu.union(int(edge_parent[edge_id]), int(edge_child[edge_id]))

    edge_roots = np.full(ts.num_edges, -1, dtype=np.int32)
    groups: dict[int, list[int]] = {}
    for edge_id in active_edge_ids:
        root = dsu.find(int(edge_parent[edge_id]))
        edge_roots[edge_id] = root
        groups.setdefault(root, []).append(int(edge_id))
    externally_overlapped_roots = _roots_with_external_overlap(
        active_edge_ids,
        edge_left,
        edge_right,
        edge_roots,
    )

    components: list[dict[str, Any]] = []
    for root, edge_id_values in groups.items():
        edge_ids = np.asarray(edge_id_values, dtype=np.int32)
        intervals = _canonical_intervals(edge_left[edge_ids], edge_right[edge_ids])
        left = float(intervals[0][0])
        right = float(intervals[-1][1])
        contiguous = len(intervals) == 1
        proper_subregion = not (
            left <= 0.0 and right >= float(ts.sequence_length)
        )
        crosses_cut = (child_time[edge_ids] <= cut_time) & (
            parent_time[edge_ids] > cut_time
        )
        frontier_edge_ids = edge_ids[crosses_cut]
        frontier_anchor_node_ids = tuple(
            sorted(set(int(value) for value in edge_child[frontier_edge_ids]))
        )
        node_ids = tuple(
            sorted(
                set(
                    int(value)
                    for value in np.concatenate(
                        (edge_parent[edge_ids], edge_child[edge_ids])
                    )
                )
            )
        )
        assigned_edges_inside = bool(
            np.all(edge_left[edge_ids] >= left)
            and np.all(edge_right[edge_ids] <= right)
        )
        reasons = []
        if not contiguous:
            reasons.append("noncontiguous_support")
        if not proper_subregion:
            reasons.append("whole_sequence")
        if frontier_edge_ids.size == 0:
            reasons.append("no_frontier_edge")
        if root in externally_overlapped_roots:
            reasons.append("outside_older_edge_overlap")
        if not assigned_edges_inside:
            reasons.append("edge_outside_support")

        components.append(
            {
                "cut_index": cut_index,
                "cut_time": cut_time,
                "root_node_id": int(root),
                "region_key": (left, right) if contiguous else tuple(intervals),
                "intervals": intervals,
                "left": left,
                "right": right,
                "span": right - left,
                "material_length": float(sum(end - start for start, end in intervals)),
                "contiguous": contiguous,
                "proper_subregion": proper_subregion,
                "older_edge_ids": tuple(int(value) for value in edge_ids),
                "older_edge_count": int(edge_ids.size),
                "node_ids": node_ids,
                "node_count": len(node_ids),
                "frontier_edge_ids": tuple(int(value) for value in frontier_edge_ids),
                "frontier_edge_count": int(frontier_edge_ids.size),
                "frontier_anchor_node_ids": frontier_anchor_node_ids,
                "assigned_edges_inside": assigned_edges_inside,
                "outside_older_edge_overlap": root in externally_overlapped_roots,
                "normal_edge_closed": not reasons,
                "rejection_reasons": tuple(reasons),
                "separation_edge_ids": tuple(
                    int(value) for value in separation_edge_ids
                ),
                "separation_parent_time": cut_time,
            }
        )
    components.sort(key=lambda item: (item["left"], item["right"]))
    for component_index, component in enumerate(components):
        component["component_id"] = (
            f"normal-cut-{cut_index}-component-{component_index:04d}"
        )

    closed_intervals = sorted(
        (component["left"], component["right"])
        for component in components
        if component["normal_edge_closed"]
    )
    for previous, current in zip(closed_intervals, closed_intervals[1:]):
        if current[0] < previous[1]:
            raise AssertionError("normal edge-closed components overlap at one cut")

    return {
        "cut_index": cut_index,
        "cut_time": cut_time,
        "older_edge_count": int(active_edge_ids.size),
        "frontier_edge_count": int(
            np.sum((child_time[active_edge_ids] <= cut_time) & (parent_time[active_edge_ids] > cut_time))
        ),
        "component_count": len(components),
        "closed_component_count": sum(
            component["normal_edge_closed"] for component in components
        ),
        "separation_edge_ids": tuple(int(value) for value in separation_edge_ids),
        "components": components,
    }


def _scan_normal_ts_edge_closed_regions_reference(
    ts_or_path: str | Path | tskit.TreeSequence,
    *,
    retain_per_cut_catalog: bool = False,
) -> NormalEdgeClosedRegionScan:
    """Reference scan which rebuilds older-edge connectivity at every cut."""

    started = time.perf_counter()
    ts = _load_tree_sequence(ts_or_path)
    parent_ids = np.asarray(ts.tables.edges.parent, dtype=np.int32)
    parent_times = np.asarray(ts.nodes_time, dtype=np.float64)[parent_ids]
    cut_times = np.unique(np.concatenate((np.asarray([0.0]), parent_times)))
    cut_times.sort()

    catalog = [] if retain_per_cut_catalog else None
    summary = []
    raw_closed_components: list[dict[str, Any]] = []
    older_edge_visits = 0
    for cut_index, cut_time in enumerate(cut_times):
        cut_result = normal_edge_components_at_cut(
            ts,
            float(cut_time),
            cut_index=cut_index,
        )
        older_edge_visits += int(cut_result["older_edge_count"])
        if catalog is not None:
            catalog.append(cut_result)
        summary.append(
            {
                "cut_index": cut_index,
                "cut_time": float(cut_time),
                "older_edges": cut_result["older_edge_count"],
                "frontier_edges": cut_result["frontier_edge_count"],
                "components": cut_result["component_count"],
                "edge_closed_components": cut_result["closed_component_count"],
            }
        )
        raw_closed_components.extend(
            component
            for component in cut_result["components"]
            if component["normal_edge_closed"]
        )

    regions_by_interval: dict[Interval, list[dict[str, Any]]] = {}
    for component in raw_closed_components:
        interval = (float(component["left"]), float(component["right"]))
        regions_by_interval.setdefault(interval, []).append(component)

    regions = []
    for interval, occurrences in regions_by_interval.items():
        occurrences.sort(key=lambda item: int(item["cut_index"]))
        first = copy.deepcopy(occurrences[0])
        last = occurrences[-1]
        first["first_normal_cut_index"] = int(first["cut_index"])
        first["first_normal_cut_time"] = float(first["cut_time"])
        first["last_normal_cut_index"] = int(last["cut_index"])
        first["last_normal_cut_time"] = float(last["cut_time"])
        first["valid_normal_cut_count"] = len(occurrences)
        first["valid_normal_cut_indices"] = tuple(
            int(item["cut_index"]) for item in occurrences
        )
        first["valid_normal_cut_times"] = tuple(
            float(item["cut_time"]) for item in occurrences
        )
        regions.append(first)
    regions.sort(
        key=lambda item: (
            item["first_normal_cut_index"],
            item["left"],
            item["right"],
        )
    )
    for region_index, region in enumerate(regions):
        region["region_id"] = f"normal-edge-region-{region_index:04d}"

    return NormalEdgeClosedRegionScan(
        regions=tuple(regions),
        per_cut_summary=tuple(summary),
        per_cut_component_catalog=None if catalog is None else tuple(catalog),
        raw_closed_component_count=len(raw_closed_components),
        scan_seconds=time.perf_counter() - started,
        diagnostics={
            "algorithm": "reference",
            "normal_time_cuts": int(cut_times.size),
            "normal_edges": int(ts.num_edges),
            "older_edge_visits": older_edge_visits,
        },
    )


class _IntervalUnion:
    """Mutable canonical half-open interval union for one DSU component."""

    def __init__(self) -> None:
        self.intervals: list[Interval] = []

    def add(self, left: float, right: float) -> None:
        if not left < right:
            raise ValueError("normal tree-sequence edges must be nonempty")
        intervals = self.intervals
        starts = [interval[0] for interval in intervals]
        index = bisect.bisect_left(starts, float(left))
        if index and intervals[index - 1][1] >= left:
            index -= 1
            left = min(left, intervals[index][0])
            right = max(right, intervals[index][1])
        end = index
        while end < len(intervals) and intervals[end][0] <= right:
            left = min(left, intervals[end][0])
            right = max(right, intervals[end][1])
            end += 1
        intervals[index:end] = [(float(left), float(right))]

    def merge_from(self, other: _IntervalUnion) -> None:
        if not other.intervals:
            return
        if not self.intervals:
            self.intervals = list(other.intervals)
            return
        merged: list[Interval] = []
        left: float | None = None
        right: float | None = None
        first_index = 0
        second_index = 0
        while first_index < len(self.intervals) or second_index < len(other.intervals):
            if second_index == len(other.intervals) or (
                first_index < len(self.intervals)
                and self.intervals[first_index][0] <= other.intervals[second_index][0]
            ):
                next_left, next_right = self.intervals[first_index]
                first_index += 1
            else:
                next_left, next_right = other.intervals[second_index]
                second_index += 1
            if left is None or next_left > right:
                if left is not None:
                    merged.append((left, right))
                left, right = next_left, next_right
            else:
                right = max(right, next_right)
        if left is not None:
            merged.append((left, right))
        self.intervals = merged

    @property
    def contiguous(self) -> bool:
        return len(self.intervals) == 1

    @property
    def left(self) -> float:
        return float(self.intervals[0][0])

    @property
    def right(self) -> float:
        return float(self.intervals[-1][1])

    @property
    def material_length(self) -> float:
        return float(sum(right - left for left, right in self.intervals))

    def as_tuple(self) -> tuple[Interval, ...]:
        return tuple(self.intervals)


@dataclass
class _IncrementalComponent:
    edge_ids: list[int]
    node_ids: set[int]
    support: _IntervalUnion
    frontier_edge_count: int
    overlap_neighbors: dict[int, int]
    closed: bool = False

    @property
    def weight(self) -> int:
        return len(self.edge_ids) + len(self.overlap_neighbors)


class _IncrementalDisjointSet:
    """Monotone edge connectivity with mergeable component metadata."""

    def __init__(self, node_count: int) -> None:
        self.parent = np.arange(int(node_count), dtype=np.int32)
        self.node_size = np.ones(int(node_count), dtype=np.int32)
        self.components: list[_IncrementalComponent | None] = [
            None for _ in range(int(node_count))
        ]
        self.active_roots: set[int] = set()

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

    def component(self, root: int) -> _IncrementalComponent:
        root = self.find(root)
        component = self.components[root]
        if component is None:
            raise KeyError(f"node {root} has no active older-edge component")
        return component

    def union(self, left: int, right: int) -> tuple[int, set[int]]:
        """Union node roots and return the surviving root and changed roots."""

        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root, {left_root}
        left_component = self.components[left_root]
        right_component = self.components[right_root]

        if left_component is None and right_component is not None:
            left_root, right_root = right_root, left_root
            left_component, right_component = right_component, left_component
        elif left_component is not None and right_component is not None:
            if left_component.weight < right_component.weight:
                left_root, right_root = right_root, left_root
                left_component, right_component = right_component, left_component
        elif left_component is None and right_component is None:
            if int(self.node_size[left_root]) < int(self.node_size[right_root]):
                left_root, right_root = right_root, left_root

        self.parent[right_root] = left_root
        self.node_size[left_root] += self.node_size[right_root]
        if right_component is None:
            return left_root, {left_root, right_root}
        if left_component is None:
            self.components[left_root] = right_component
            self.components[right_root] = None
            self.active_roots.discard(right_root)
            self.active_roots.add(left_root)
            return left_root, {left_root, right_root}

        changed_roots = {left_root, right_root}
        # Relations between the roots become internal after the union.
        left_component.overlap_neighbors.pop(right_root, None)
        right_component.overlap_neighbors.pop(left_root, None)
        for neighbor_root, count in list(right_component.overlap_neighbors.items()):
            neighbor_root = self.find(neighbor_root)
            if neighbor_root == left_root:
                continue
            neighbor_component = self.components[neighbor_root]
            if neighbor_component is None:
                continue
            neighbor_component.overlap_neighbors.pop(right_root, None)
            neighbor_component.overlap_neighbors[left_root] = (
                neighbor_component.overlap_neighbors.get(left_root, 0) + count
            )
            left_component.overlap_neighbors[neighbor_root] = (
                left_component.overlap_neighbors.get(neighbor_root, 0) + count
            )
            changed_roots.add(neighbor_root)
        right_component.overlap_neighbors.clear()
        if len(left_component.edge_ids) < len(right_component.edge_ids):
            left_component.edge_ids, right_component.edge_ids = (
                right_component.edge_ids,
                left_component.edge_ids,
            )
        left_component.edge_ids.extend(right_component.edge_ids)
        if len(left_component.node_ids) < len(right_component.node_ids):
            left_component.node_ids, right_component.node_ids = (
                right_component.node_ids,
                left_component.node_ids,
            )
        left_component.node_ids.update(right_component.node_ids)
        left_component.support.merge_from(right_component.support)
        left_component.frontier_edge_count += right_component.frontier_edge_count
        self.components[right_root] = None
        self.active_roots.discard(right_root)
        self.active_roots.add(left_root)
        return left_root, changed_roots

    def attach_edge(
        self,
        edge_id: int,
        parent_node: int,
        child_node: int,
        left: float,
        right: float,
    ) -> int:
        root = self.find(parent_node)
        component = self.components[root]
        if component is None:
            component = _IncrementalComponent(
                edge_ids=[],
                node_ids=set(),
                support=_IntervalUnion(),
                frontier_edge_count=0,
                overlap_neighbors={},
            )
            self.components[root] = component
            self.active_roots.add(root)
        component.edge_ids.append(int(edge_id))
        component.node_ids.add(int(parent_node))
        component.node_ids.add(int(child_node))
        component.support.add(float(left), float(right))
        return root

    def add_overlap_relation(self, left_node: int, right_node: int) -> set[int]:
        left_root = self.find(left_node)
        right_root = self.find(right_node)
        if left_root == right_root:
            return {left_root}
        left_component = self.component(left_root)
        right_component = self.component(right_root)
        left_component.overlap_neighbors[right_root] = (
            left_component.overlap_neighbors.get(right_root, 0) + 1
        )
        right_component.overlap_neighbors[left_root] = (
            right_component.overlap_neighbors.get(left_root, 0) + 1
        )
        return {left_root, right_root}

    def change_frontier_edge(self, parent_node: int, delta: int) -> int:
        root = self.find(parent_node)
        component = self.component(root)
        component.frontier_edge_count += int(delta)
        if component.frontier_edge_count < 0:
            raise AssertionError("negative incremental frontier-edge count")
        return root


def _edge_overlap_pairs_by_activation_time(
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    parent_time: np.ndarray,
) -> tuple[dict[float, list[tuple[int, int]]], int, int]:
    """Group positive-overlap pairs by the later edge-insertion batch."""

    order = np.argsort(edge_left, kind="stable")
    active: list[int] = []
    pairs_by_time: dict[float, list[tuple[int, int]]] = {}
    pair_count = 0
    max_active = 0
    for edge_value in order:
        edge_id = int(edge_value)
        left = float(edge_left[edge_id])
        active = [
            other_edge
            for other_edge in active
            if float(edge_right[other_edge]) > left
        ]
        for other_edge in active:
            activation_time = float(
                min(parent_time[edge_id], parent_time[other_edge])
            )
            pairs_by_time.setdefault(activation_time, []).append(
                (edge_id, int(other_edge))
            )
            pair_count += 1
        active.append(edge_id)
        max_active = max(max_active, len(active))
    return pairs_by_time, pair_count, max_active


def _normal_cut_schedule(
    parent_time: np.ndarray,
    child_time: np.ndarray,
) -> tuple[
    np.ndarray,
    dict[float, list[int]],
    dict[int, list[int]],
    dict[int, list[int]],
]:
    cut_times = np.unique(np.concatenate((np.asarray([0.0]), parent_time)))
    cut_times.sort()
    batches: dict[float, list[int]] = {}
    activate_at: dict[int, list[int]] = {}
    deactivate_at: dict[int, list[int]] = {}
    for edge_id, (edge_parent_time, edge_child_time) in enumerate(
        zip(parent_time, child_time)
    ):
        parent_index = int(np.searchsorted(cut_times, edge_parent_time, side="left"))
        child_index = int(np.searchsorted(cut_times, edge_child_time, side="left"))
        if parent_index <= child_index:
            raise AssertionError("normal edge has no representable crossing cut")
        batches.setdefault(float(edge_parent_time), []).append(edge_id)
        activate_at.setdefault(parent_index - 1, []).append(edge_id)
        if child_index > 0:
            deactivate_at.setdefault(child_index - 1, []).append(edge_id)
    return cut_times, batches, activate_at, deactivate_at


def _incremental_component_is_closed(
    component: _IncrementalComponent,
    sequence_length: float,
) -> bool:
    if not component.support.contiguous or component.frontier_edge_count == 0:
        return False
    if component.support.left <= 0.0 and component.support.right >= sequence_length:
        return False
    return not component.overlap_neighbors


def _materialize_incremental_component(
    component: _IncrementalComponent,
    root: int,
    cut_index: int,
    cut_time: float,
    sequence_length: float,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    edge_parent: np.ndarray,
    edge_child: np.ndarray,
    parent_time: np.ndarray,
    child_time: np.ndarray,
    separation_edge_ids: tuple[int, ...],
) -> dict[str, Any]:
    edge_ids = np.asarray(sorted(component.edge_ids), dtype=np.int32)
    intervals = component.support.as_tuple()
    left = component.support.left
    right = component.support.right
    contiguous = component.support.contiguous
    proper_subregion = not (left <= 0.0 and right >= sequence_length)
    crosses_cut = (child_time[edge_ids] <= cut_time) & (parent_time[edge_ids] > cut_time)
    frontier_edge_ids = edge_ids[crosses_cut]
    frontier_anchor_node_ids = tuple(
        sorted(set(int(value) for value in edge_child[frontier_edge_ids]))
    )
    assigned_edges_inside = bool(
        np.all(edge_left[edge_ids] >= left)
        and np.all(edge_right[edge_ids] <= right)
    )
    reasons = []
    if not contiguous:
        reasons.append("noncontiguous_support")
    if not proper_subregion:
        reasons.append("whole_sequence")
    if frontier_edge_ids.size == 0:
        reasons.append("no_frontier_edge")
    if component.overlap_neighbors:
        reasons.append("outside_older_edge_overlap")
    if not assigned_edges_inside:
        reasons.append("edge_outside_support")
    return {
        "cut_index": cut_index,
        "cut_time": cut_time,
        "root_node_id": int(root),
        "region_key": (left, right) if contiguous else tuple(intervals),
        "intervals": intervals,
        "left": left,
        "right": right,
        "span": right - left,
        "material_length": component.support.material_length,
        "contiguous": contiguous,
        "proper_subregion": proper_subregion,
        "older_edge_ids": tuple(int(value) for value in edge_ids),
        "older_edge_count": int(edge_ids.size),
        "node_ids": tuple(sorted(component.node_ids)),
        "node_count": len(component.node_ids),
        "frontier_edge_ids": tuple(int(value) for value in frontier_edge_ids),
        "frontier_edge_count": int(frontier_edge_ids.size),
        "frontier_anchor_node_ids": frontier_anchor_node_ids,
        "assigned_edges_inside": assigned_edges_inside,
        "outside_older_edge_overlap": bool(component.overlap_neighbors),
        "normal_edge_closed": not reasons,
        "rejection_reasons": tuple(reasons),
        "separation_edge_ids": separation_edge_ids,
        "separation_parent_time": cut_time,
    }


def _scan_normal_ts_edge_closed_regions_incremental(
    ts_or_path: str | Path | tskit.TreeSequence,
    *,
    retain_per_cut_catalog: bool = False,
) -> NormalEdgeClosedRegionScan:
    """Single reverse-time pass over normal edges and normal time cuts."""

    started = time.perf_counter()
    ts = _load_tree_sequence(ts_or_path)
    tables = ts.tables
    edge_left = np.asarray(tables.edges.left, dtype=np.float64)
    edge_right = np.asarray(tables.edges.right, dtype=np.float64)
    edge_parent = np.asarray(tables.edges.parent, dtype=np.int32)
    edge_child = np.asarray(tables.edges.child, dtype=np.int32)
    node_time = np.asarray(tables.nodes.time, dtype=np.float64)
    parent_time = node_time[edge_parent]
    child_time = node_time[edge_child]
    (
        cut_times,
        edge_batches,
        activate_at,
        deactivate_at,
    ) = _normal_cut_schedule(parent_time, child_time)
    overlap_pairs, overlap_pair_count, max_interval_overlap = (
        _edge_overlap_pairs_by_activation_time(edge_left, edge_right, parent_time)
    )

    dsu = _IncrementalDisjointSet(ts.num_nodes)
    closed_roots: set[int] = set()
    catalog = [] if retain_per_cut_catalog else None
    summary: list[dict[str, Any]] = []
    raw_closed_components: list[dict[str, Any]] = []
    frontier_edge_count = 0
    older_edge_count = 0
    component_refresh_count = 0

    def refresh(root_value: int) -> None:
        nonlocal component_refresh_count
        component_refresh_count += 1
        root = dsu.find(root_value)
        if root != root_value:
            closed_roots.discard(root_value)
        component = dsu.components[root]
        if component is None:
            closed_roots.discard(root_value)
            closed_roots.discard(root)
            return
        now_closed = _incremental_component_is_closed(
            component, float(ts.sequence_length)
        )
        component.closed = now_closed
        if now_closed:
            closed_roots.add(root)
        else:
            closed_roots.discard(root)

    for cut_index in range(len(cut_times) - 1, -1, -1):
        cut_time = float(cut_times[cut_index])
        changed_roots: set[int] = set()
        for edge_id in deactivate_at.get(cut_index, ()):
            root = dsu.change_frontier_edge(int(edge_parent[edge_id]), -1)
            changed_roots.add(root)
            frontier_edge_count -= 1
        for edge_id in activate_at.get(cut_index, ()):
            root = dsu.change_frontier_edge(int(edge_parent[edge_id]), 1)
            changed_roots.add(root)
            frontier_edge_count += 1
        for root in changed_roots:
            refresh(root)

        separation_edge_ids = tuple(
            int(value) for value in edge_batches.get(cut_time, ())
        )
        if catalog is not None:
            components = [
                _materialize_incremental_component(
                    dsu.component(root),
                    root,
                    cut_index,
                    cut_time,
                    float(ts.sequence_length),
                    edge_left,
                    edge_right,
                    edge_parent,
                    edge_child,
                    parent_time,
                    child_time,
                    separation_edge_ids,
                )
                for root in dsu.active_roots
            ]
            components.sort(key=lambda item: (item["left"], item["right"]))
            for component_index, component in enumerate(components):
                component["component_id"] = (
                    f"normal-cut-{cut_index}-component-{component_index:04d}"
                )
            catalog.append(
                {
                    "cut_index": cut_index,
                    "cut_time": cut_time,
                    "older_edge_count": older_edge_count,
                    "frontier_edge_count": frontier_edge_count,
                    "component_count": len(components),
                    "closed_component_count": len(closed_roots),
                    "separation_edge_ids": separation_edge_ids,
                    "components": components,
                }
            )

        closed_components = [
            _materialize_incremental_component(
                dsu.component(root),
                root,
                cut_index,
                cut_time,
                float(ts.sequence_length),
                edge_left,
                edge_right,
                edge_parent,
                edge_child,
                parent_time,
                child_time,
                separation_edge_ids,
            )
            for root in closed_roots
        ]
        closed_components.sort(key=lambda item: (item["left"], item["right"]))
        for component_index, component in enumerate(closed_components):
            component["component_id"] = (
                f"normal-cut-{cut_index}-closed-{component_index:04d}"
            )
        raw_closed_components.extend(closed_components)
        summary.append(
            {
                "cut_index": cut_index,
                "cut_time": cut_time,
                "older_edges": older_edge_count,
                "frontier_edges": frontier_edge_count,
                "components": len(dsu.active_roots),
                "edge_closed_components": len(closed_roots),
                "edge_overlap_pairs": overlap_pair_count,
                "max_simultaneous_edge_overlap": max_interval_overlap,
            }
        )

        # The snapshot for this cut has been emitted. Add the parent-time batch
        # so the next younger snapshot contains precisely parent.time > cut.
        batch_changed_roots: set[int] = set()
        batch_edge_ids = edge_batches.get(cut_time, ())
        for edge_id in batch_edge_ids:
            root, affected = dsu.union(
                int(edge_parent[edge_id]), int(edge_child[edge_id])
            )
            batch_changed_roots.update(affected)
            root = dsu.attach_edge(
                edge_id,
                int(edge_parent[edge_id]),
                int(edge_child[edge_id]),
                float(edge_left[edge_id]),
                float(edge_right[edge_id]),
            )
            batch_changed_roots.add(root)
        older_edge_count += len(batch_edge_ids)
        for left_edge, right_edge in overlap_pairs.get(cut_time, ()):
            batch_changed_roots.update(
                dsu.add_overlap_relation(
                    int(edge_parent[left_edge]), int(edge_parent[right_edge])
                )
            )
        for root in batch_changed_roots:
            refresh(root)

    # Reverse-time iteration is convenient for incremental updates; public
    # results retain the original increasing-time order.
    summary.sort(key=lambda item: int(item["cut_index"]))
    if catalog is not None:
        catalog.sort(key=lambda item: int(item["cut_index"]))

    regions_by_interval: dict[Interval, list[dict[str, Any]]] = {}
    for component in raw_closed_components:
        interval = (float(component["left"]), float(component["right"]))
        regions_by_interval.setdefault(interval, []).append(component)
    regions = []
    for occurrences in regions_by_interval.values():
        occurrences.sort(key=lambda item: int(item["cut_index"]))
        first = copy.deepcopy(occurrences[0])
        last = occurrences[-1]
        first["first_normal_cut_index"] = int(first["cut_index"])
        first["first_normal_cut_time"] = float(first["cut_time"])
        first["last_normal_cut_index"] = int(last["cut_index"])
        first["last_normal_cut_time"] = float(last["cut_time"])
        first["valid_normal_cut_count"] = len(occurrences)
        first["valid_normal_cut_indices"] = tuple(
            int(item["cut_index"]) for item in occurrences
        )
        first["valid_normal_cut_times"] = tuple(
            float(item["cut_time"]) for item in occurrences
        )
        regions.append(first)
    regions.sort(
        key=lambda item: (
            item["first_normal_cut_index"],
            item["left"],
            item["right"],
        )
    )
    for region_index, region in enumerate(regions):
        region["region_id"] = f"normal-edge-region-{region_index:04d}"

    return NormalEdgeClosedRegionScan(
        regions=tuple(regions),
        per_cut_summary=tuple(summary),
        per_cut_component_catalog=None if catalog is None else tuple(catalog),
        raw_closed_component_count=len(raw_closed_components),
        scan_seconds=time.perf_counter() - started,
        diagnostics={
            "algorithm": "incremental",
            "normal_time_cuts": int(cut_times.size),
            "normal_edges": int(ts.num_edges),
            "older_edge_visits": int(ts.num_edges),
            "edge_overlap_pairs": overlap_pair_count,
            "max_simultaneous_edge_overlap": max_interval_overlap,
            "component_refreshes": component_refresh_count,
        },
    )


def scan_normal_ts_edge_closed_regions(
    ts_or_path: str | Path | tskit.TreeSequence,
    *,
    retain_per_cut_catalog: bool = False,
    algorithm: str = "incremental",
) -> NormalEdgeClosedRegionScan:
    """Scan normal time cuts for direct edge-closed genomic regions.

    ``algorithm=\"incremental\"`` updates connectivity one parent-time batch at
    a time. ``algorithm=\"reference\"`` is the slower rebuild-at-every-cut
    implementation retained for parity tests and diagnostics.
    """

    if algorithm == "incremental":
        return _scan_normal_ts_edge_closed_regions_incremental(
            ts_or_path,
            retain_per_cut_catalog=retain_per_cut_catalog,
        )
    if algorithm == "reference":
        return _scan_normal_ts_edge_closed_regions_reference(
            ts_or_path,
            retain_per_cut_catalog=retain_per_cut_catalog,
        )
    raise ValueError("algorithm must be 'incremental' or 'reference'")


def benchmark_normal_ts_edge_closed_regions(
    ts_or_path: str | Path | tskit.TreeSequence,
    *,
    warm_up: bool = True,
) -> NormalEdgeClosedRegionBenchmark:
    """Benchmark both implementations without retaining component catalogs."""

    if warm_up:
        scan_normal_ts_edge_closed_regions(ts_or_path, algorithm="reference")
        scan_normal_ts_edge_closed_regions(ts_or_path, algorithm="incremental")

    def measure(algorithm: str) -> tuple[NormalEdgeClosedRegionScan, int]:
        tracemalloc.start()
        result = scan_normal_ts_edge_closed_regions(ts_or_path, algorithm=algorithm)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return result, int(peak)

    reference, reference_peak = measure("reference")
    incremental, incremental_peak = measure("incremental")
    if reference.scan_seconds <= 0.0:
        raise AssertionError("reference benchmark duration must be positive")
    return NormalEdgeClosedRegionBenchmark(
        reference_seconds=reference.scan_seconds,
        incremental_seconds=incremental.scan_seconds,
        speedup=reference.scan_seconds / incremental.scan_seconds,
        reference_peak_bytes=reference_peak,
        incremental_peak_bytes=incremental_peak,
        reference_diagnostics=reference.diagnostics,
        incremental_diagnostics=incremental.diagnostics,
    )


__all__ = [
    "NormalEdgeClosedRegionBenchmark",
    "NormalEdgeClosedRegionScan",
    "benchmark_normal_ts_edge_closed_regions",
    "normal_edge_components_at_cut",
    "scan_normal_ts_edge_closed_regions",
]
