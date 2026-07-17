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
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

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


def scan_normal_ts_edge_closed_regions(
    ts_or_path: str | Path | tskit.TreeSequence,
    *,
    retain_per_cut_catalog: bool = False,
) -> NormalEdgeClosedRegionScan:
    """Scan all normal parent-time cuts for directly edge-closed regions."""

    started = time.perf_counter()
    ts = _load_tree_sequence(ts_or_path)
    parent_ids = np.asarray(ts.tables.edges.parent, dtype=np.int32)
    parent_times = np.asarray(ts.nodes_time, dtype=np.float64)[parent_ids]
    cut_times = np.unique(np.concatenate((np.asarray([0.0]), parent_times)))
    cut_times.sort()

    catalog = [] if retain_per_cut_catalog else None
    summary = []
    raw_closed_components: list[dict[str, Any]] = []
    for cut_index, cut_time in enumerate(cut_times):
        cut_result = normal_edge_components_at_cut(
            ts,
            float(cut_time),
            cut_index=cut_index,
        )
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
    )


__all__ = [
    "NormalEdgeClosedRegionScan",
    "normal_edge_components_at_cut",
    "scan_normal_ts_edge_closed_regions",
]
