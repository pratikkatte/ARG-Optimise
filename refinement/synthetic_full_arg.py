"""Core conversion from an inferred tree sequence to a synthetic full ARG.

The converter inserts explicit paired recombination nodes, assigns a distinct
time to every ARG event, and records enough provenance for downstream strict
validation. Graph extraction, visualization, animation, and persistence
wrappers intentionally live outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import tskit


NODE_IS_RE_EVENT = 131072
SYNTHETIC_FULL_ARG_PROVENANCE_NAME = "new_rl_synthetic_full_arg"
_LEGACY_PROVENANCE_NAMES = frozenset({"local_argscape_synthetic_full_arg"})

_TIME_EPSILON_SCALE = 1e-9
_MIN_TIME_EPSILON = 1e-12

_EVENT_KIND_RECOMBINATION = 0
_EVENT_KIND_COALESCENCE = 1
_EVENT_KIND_UNARY = 2
_EVENT_KIND_REVEAL = 3


@dataclass(frozen=True)
class SyntheticFullARGResult:
    """Converted tree sequence plus a compact conversion summary."""

    tree_sequence: tskit.TreeSequence
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _MergedEdgeGroups:
    child: np.ndarray
    parent: np.ndarray
    left: np.ndarray
    right: np.ndarray


@dataclass(frozen=True)
class _CompactCandidates:
    children: np.ndarray
    group_start: np.ndarray
    group_end: np.ndarray
    groups: _MergedEdgeGroups

    @property
    def candidate_count(self) -> int:
        return int(self.children.size)

    @property
    def event_count(self) -> int:
        if self.children.size == 0:
            return 0
        return int(np.sum(self.group_end - self.group_start - 1))


@dataclass(frozen=True)
class _EventTimeAdjustment:
    adjusted_event_count: int
    max_adjustment: float
    times_are_globally_unique: bool


def build_synthetic_full_arg(
    ts_or_path: str | Path | tskit.TreeSequence,
    *,
    time_rule: str = "midpoint",
    split_rule: str = "balanced",
    ensure_unique_event_times: bool = True,
) -> SyntheticFullARGResult:
    """Insert explicit synthetic recombination events into a tree sequence.

    Each recombination event is represented by two consecutive flagged nodes
    with the same time and child. By default, different ARG events receive
    globally unique times so they can be replayed one at a time.
    """
    if time_rule != "midpoint":
        raise ValueError("only time_rule='midpoint' is currently implemented")
    if split_rule not in {"balanced", "left_to_right"}:
        raise ValueError("split_rule must be 'balanced' or 'left_to_right'")

    ts = _load_tree_sequence(ts_or_path)
    tables = ts.dump_tables()
    _mark_mutation_times_unknown(tables)
    compact = _find_compact_candidates(ts)
    original_num_nodes = int(ts.num_nodes)
    synthetic_event_count = compact.event_count

    if synthetic_event_count:
        node_time = np.asarray(tables.nodes.time, dtype=np.float64)
        edge_left = np.asarray(tables.edges.left, dtype=np.float64)
        edge_right = np.asarray(tables.edges.right, dtype=np.float64)
        edge_parent = np.asarray(tables.edges.parent, dtype=np.int32)
        edge_child = np.asarray(tables.edges.child, dtype=np.int32)

        candidate_edge_mask = np.isin(edge_child, compact.children)
        kept_edge_count = int(np.sum(~candidate_edge_mask))
        synthetic_edge_count = _synthetic_edge_count(
            compact.group_end - compact.group_start,
            split_rule,
        )
        output_edge_count = kept_edge_count + synthetic_edge_count

        output_left = np.empty(output_edge_count, dtype=np.float64)
        output_right = np.empty(output_edge_count, dtype=np.float64)
        output_parent = np.empty(output_edge_count, dtype=np.int32)
        output_child = np.empty(output_edge_count, dtype=np.int32)

        keep = ~candidate_edge_mask
        edge_cursor = kept_edge_count
        output_left[:edge_cursor] = edge_left[keep]
        output_right[:edge_cursor] = edge_right[keep]
        output_parent[:edge_cursor] = edge_parent[keep]
        output_child[:edge_cursor] = edge_child[keep]

        synthetic_node_time = np.empty(synthetic_event_count * 2, dtype=np.float64)
        node_cursor = 0

        for start, end, child in zip(
            compact.group_start,
            compact.group_end,
            compact.children,
        ):
            writer = (
                _write_balanced_topology
                if split_rule == "balanced"
                else _write_left_to_right_topology
            )
            edge_cursor, node_cursor = writer(
                groups=compact.groups,
                start=int(start),
                end=int(end),
                downstream_node=int(child),
                downstream_time=float(node_time[int(child)]),
                node_time=node_time,
                original_num_nodes=original_num_nodes,
                synthetic_node_time=synthetic_node_time,
                node_cursor=node_cursor,
                output_left=output_left,
                output_right=output_right,
                output_parent=output_parent,
                output_child=output_child,
                edge_cursor=edge_cursor,
            )

        if edge_cursor != output_edge_count:
            raise RuntimeError(
                "synthetic edge writer filled "
                f"{edge_cursor} rows, expected {output_edge_count}"
            )
        if node_cursor != synthetic_node_time.size:
            raise RuntimeError(
                "synthetic node writer filled "
                f"{node_cursor} rows, expected {synthetic_node_time.size}"
            )

        tables.nodes.append_columns(
            flags=np.full(
                synthetic_node_time.size,
                NODE_IS_RE_EVENT,
                dtype=np.uint32,
            ),
            time=synthetic_node_time,
        )
        tables.edges.set_columns(
            left=output_left,
            right=output_right,
            parent=output_parent,
            child=output_child,
        )

    # Event classification expects parent rows to be contiguous. A second sort
    # is required after installing any adjusted event times.
    tables.sort()
    if ensure_unique_event_times:
        time_adjustment = _ensure_globally_unique_event_times(tables)
    else:
        time_adjustment = _summarize_event_times(tables)

    _add_provenance(
        tables=tables,
        original_num_nodes=original_num_nodes,
        candidate_count=compact.candidate_count,
        synthetic_event_count=synthetic_event_count,
        time_rule=time_rule,
        split_rule=split_rule,
        ensure_unique_event_times=ensure_unique_event_times,
        time_adjustment=time_adjustment,
    )
    tables.sort()
    synthetic_ts = tables.tree_sequence()

    metadata = {
        "source": "synthetic_full_arg",
        "time_rule": time_rule,
        "split_rule": split_rule,
        "ensure_unique_event_times": bool(ensure_unique_event_times),
        "event_times_are_globally_unique": (
            time_adjustment.times_are_globally_unique
        ),
        "event_time_adjustment_rule": (
            "scale_aware_bidirectional_spacing"
            if ensure_unique_event_times
            else "none"
        ),
        "event_time_adjusted_event_count": time_adjustment.adjusted_event_count,
        "max_event_time_adjustment": time_adjustment.max_adjustment,
        "original_num_nodes": original_num_nodes,
        "original_num_edges": int(ts.num_edges),
        "original_num_trees": int(ts.num_trees),
        "synthetic_recombination_event_count": synthetic_event_count,
        "synthetic_recombination_node_count": synthetic_event_count * 2,
        "augmented_num_nodes": int(synthetic_ts.num_nodes),
        "augmented_num_edges": int(synthetic_ts.num_edges),
        "augmented_num_trees": int(synthetic_ts.num_trees),
        "mutation_times_set_unknown": True,
        "candidate_count": compact.candidate_count,
        "imputed_times_are_synthetic": True,
    }
    return SyntheticFullARGResult(
        tree_sequence=synthetic_ts,
        metadata=metadata,
    )


def get_synthetic_full_arg_provenance(
    ts: tskit.TreeSequence,
) -> dict[str, Any] | None:
    """Return recognized synthetic/full-ARG provenance, newest first.

    Legacy Argscape provenance remains recognized so existing converted files
    do not need to be regenerated during the migration to ``new_rl``.
    """
    accepted_names = _LEGACY_PROVENANCE_NAMES | {
        SYNTHETIC_FULL_ARG_PROVENANCE_NAME
    }
    for provenance in reversed(list(ts.provenances())):
        try:
            record = json.loads(provenance.record)
        except (TypeError, ValueError):
            continue
        if record.get("software", {}).get("name") in accepted_names:
            return record
    return None


def _load_tree_sequence(
    ts_or_path: str | Path | tskit.TreeSequence,
) -> tskit.TreeSequence:
    if isinstance(ts_or_path, tskit.TreeSequence):
        return ts_or_path
    return tskit.load(str(Path(ts_or_path)))


def _find_compact_candidates(ts: tskit.TreeSequence) -> _CompactCandidates:
    groups = _merged_edge_groups(ts)
    if groups.child.size == 0:
        return _CompactCandidates(
            children=np.empty(0, dtype=np.int32),
            group_start=np.empty(0, dtype=np.int64),
            group_end=np.empty(0, dtype=np.int64),
            groups=groups,
        )

    child_boundary = np.empty(groups.child.size, dtype=bool)
    child_boundary[0] = True
    child_boundary[1:] = groups.child[1:] != groups.child[:-1]
    child_start = np.flatnonzero(child_boundary).astype(np.int64)
    child_end = np.empty_like(child_start)
    child_end[:-1] = child_start[1:]
    child_end[-1] = groups.child.size

    explicit_children = _explicit_recombination_children(ts)
    candidate_children: list[int] = []
    candidate_start: list[int] = []
    candidate_end: list[int] = []
    for start, end in zip(child_start, child_end):
        child_id = int(groups.child[start])
        if child_id in explicit_children:
            continue
        if end - start <= 1:
            continue
        parent = groups.parent[start:end]
        if not np.any(parent != parent[0]):
            continue
        candidate_children.append(child_id)
        candidate_start.append(int(start))
        candidate_end.append(int(end))

    return _CompactCandidates(
        children=np.asarray(candidate_children, dtype=np.int32),
        group_start=np.asarray(candidate_start, dtype=np.int64),
        group_end=np.asarray(candidate_end, dtype=np.int64),
        groups=groups,
    )


def _explicit_recombination_children(
    ts: tskit.TreeSequence,
) -> set[int]:
    """Return children already represented by valid flagged node pairs."""

    tables = ts.tables
    node_flags = np.asarray(tables.nodes.flags, dtype=np.uint32)
    node_time = np.asarray(tables.nodes.time, dtype=np.float64)
    recombination_nodes = np.flatnonzero(
        (node_flags & NODE_IS_RE_EVENT) != 0
    ).astype(np.int32)
    if recombination_nodes.size == 0:
        return set()
    if recombination_nodes.size % 2:
        raise ValueError(
            "explicit recombination nodes must occur in consecutive pairs"
        )
    children_by_parent: dict[int, set[int]] = {}
    for parent, child in zip(tables.edges.parent, tables.edges.child):
        children_by_parent.setdefault(int(parent), set()).add(int(child))

    explicit: set[int] = set()
    for left_node, right_node in zip(
        recombination_nodes[0::2],
        recombination_nodes[1::2],
    ):
        left_node = int(left_node)
        right_node = int(right_node)
        left_children = children_by_parent.get(left_node, set())
        right_children = children_by_parent.get(right_node, set())
        if (
            len(left_children) != 1
            or left_children != right_children
            or float(node_time[left_node]) != float(node_time[right_node])
        ):
            raise ValueError(
                "flagged recombination nodes must be paired by matching "
                f"time and one common child: nodes {left_node}, {right_node}"
            )
        explicit.update(left_children)
    return explicit


def _merged_edge_groups(ts: tskit.TreeSequence) -> _MergedEdgeGroups:
    edges = ts.tables.edges
    if edges.num_rows == 0:
        return _MergedEdgeGroups(
            child=np.empty(0, dtype=np.int32),
            parent=np.empty(0, dtype=np.int32),
            left=np.empty(0, dtype=np.float64),
            right=np.empty(0, dtype=np.float64),
        )

    left = np.asarray(edges.left, dtype=np.float64)
    right = np.asarray(edges.right, dtype=np.float64)
    parent = np.asarray(edges.parent, dtype=np.int32)
    child = np.asarray(edges.child, dtype=np.int32)

    order = np.lexsort((parent, right, left, child))
    sorted_left = left[order]
    sorted_right = right[order]
    sorted_parent = parent[order]
    sorted_child = child[order]

    starts_group = np.empty(edges.num_rows, dtype=bool)
    starts_group[0] = True
    starts_group[1:] = (
        (sorted_child[1:] != sorted_child[:-1])
        | (sorted_parent[1:] != sorted_parent[:-1])
        | (sorted_left[1:] != sorted_right[:-1])
    )
    starts = np.flatnonzero(starts_group)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = edges.num_rows

    return _MergedEdgeGroups(
        child=sorted_child[starts].astype(np.int32, copy=False),
        parent=sorted_parent[starts].astype(np.int32, copy=False),
        left=sorted_left[starts].astype(np.float64, copy=False),
        right=sorted_right[ends - 1].astype(np.float64, copy=False),
    )


def _synthetic_edge_count(group_counts: np.ndarray, split_rule: str) -> int:
    total = 0
    for count in group_counts:
        if split_rule == "balanced":
            total += _balanced_topology_edge_count(int(count))
        else:
            total += _left_to_right_topology_edge_count(int(count))
    return int(total)


def _balanced_topology_edge_count(group_count: int) -> int:
    total = 0
    stack = [int(group_count)]
    while stack:
        count = stack.pop()
        if count <= 1:
            total += count
        else:
            mid = count // 2
            total += count
            stack.append(mid)
            stack.append(count - mid)
    return total


def _left_to_right_topology_edge_count(group_count: int) -> int:
    count = int(group_count)
    if count <= 1:
        return count
    return int(
        sum(2 + (count - event_index - 1) for event_index in range(count - 1))
        + 1
    )


def _write_balanced_topology(
    *,
    groups: _MergedEdgeGroups,
    start: int,
    end: int,
    downstream_node: int,
    downstream_time: float,
    node_time: np.ndarray,
    original_num_nodes: int,
    synthetic_node_time: np.ndarray,
    node_cursor: int,
    output_left: np.ndarray,
    output_right: np.ndarray,
    output_parent: np.ndarray,
    output_child: np.ndarray,
    edge_cursor: int,
) -> tuple[int, int]:
    parent_time = node_time[groups.parent[start:end]]

    def write_partition(
        lo: int,
        hi: int,
        current_downstream_node: int,
        current_downstream_time: float,
    ) -> None:
        nonlocal edge_cursor, node_cursor

        count = hi - lo
        if count == 1:
            output_left[edge_cursor] = groups.left[start + lo]
            output_right[edge_cursor] = groups.right[start + lo]
            output_parent[edge_cursor] = groups.parent[start + lo]
            output_child[edge_cursor] = current_downstream_node
            edge_cursor += 1
            return

        mid = lo + count // 2
        upper_bound = float(np.min(parent_time[lo:hi]))
        event_time = _impute_midpoint_time(
            current_downstream_time,
            upper_bound,
            _balanced_depth(count),
        )

        left_node = original_num_nodes + node_cursor
        right_node = left_node + 1
        synthetic_node_time[node_cursor : node_cursor + 2] = event_time
        node_cursor += 2

        left_len = mid - lo
        right_len = hi - mid

        output_slice = slice(edge_cursor, edge_cursor + left_len)
        source_slice = slice(start + lo, start + mid)
        output_left[output_slice] = groups.left[source_slice]
        output_right[output_slice] = groups.right[source_slice]
        output_parent[output_slice] = left_node
        output_child[output_slice] = current_downstream_node
        edge_cursor += left_len

        output_slice = slice(edge_cursor, edge_cursor + right_len)
        source_slice = slice(start + mid, start + hi)
        output_left[output_slice] = groups.left[source_slice]
        output_right[output_slice] = groups.right[source_slice]
        output_parent[output_slice] = right_node
        output_child[output_slice] = current_downstream_node
        edge_cursor += right_len

        write_partition(lo, mid, left_node, event_time)
        write_partition(mid, hi, right_node, event_time)

    write_partition(0, end - start, downstream_node, downstream_time)
    return edge_cursor, node_cursor


def _write_left_to_right_topology(
    *,
    groups: _MergedEdgeGroups,
    start: int,
    end: int,
    downstream_node: int,
    downstream_time: float,
    node_time: np.ndarray,
    original_num_nodes: int,
    synthetic_node_time: np.ndarray,
    node_cursor: int,
    output_left: np.ndarray,
    output_right: np.ndarray,
    output_parent: np.ndarray,
    output_child: np.ndarray,
    edge_cursor: int,
) -> tuple[int, int]:
    parent_time = node_time[groups.parent[start:end]]
    current_downstream_node = int(downstream_node)
    current_downstream_time = float(downstream_time)
    count = end - start

    for event_index in range(count - 1):
        upper_bound = float(np.min(parent_time[event_index:]))
        event_time = _impute_midpoint_time(
            current_downstream_time,
            upper_bound,
            count - event_index - 1,
        )

        left_node = original_num_nodes + node_cursor
        right_node = left_node + 1
        synthetic_node_time[node_cursor : node_cursor + 2] = event_time
        node_cursor += 2

        group_index = start + event_index
        output_left[edge_cursor] = groups.left[group_index]
        output_right[edge_cursor] = groups.right[group_index]
        output_parent[edge_cursor] = groups.parent[group_index]
        output_child[edge_cursor] = left_node
        edge_cursor += 1

        output_left[edge_cursor] = groups.left[group_index]
        output_right[edge_cursor] = groups.right[group_index]
        output_parent[edge_cursor] = left_node
        output_child[edge_cursor] = current_downstream_node
        edge_cursor += 1

        remaining = count - event_index - 1
        output_slice = slice(edge_cursor, edge_cursor + remaining)
        source_slice = slice(group_index + 1, end)
        output_left[output_slice] = groups.left[source_slice]
        output_right[output_slice] = groups.right[source_slice]
        output_parent[output_slice] = right_node
        output_child[output_slice] = current_downstream_node
        edge_cursor += remaining

        current_downstream_node = right_node
        current_downstream_time = event_time

    last = end - 1
    output_left[edge_cursor] = groups.left[last]
    output_right[edge_cursor] = groups.right[last]
    output_parent[edge_cursor] = groups.parent[last]
    output_child[edge_cursor] = current_downstream_node
    edge_cursor += 1
    return edge_cursor, node_cursor


def _balanced_depth(group_count: int) -> int:
    return max(1, (int(group_count) - 1).bit_length())


def _impute_midpoint_time(
    lower_bound: float,
    upper_bound: float,
    levels_to_fit: int = 1,
) -> float:
    lower = float(lower_bound)
    upper = float(upper_bound)
    if not upper > lower:
        raise ValueError(
            "cannot place synthetic recombination time with parent.time > "
            f"event.time > child.time: lower={lower}, upper={upper}"
        )
    denominator = max(2, int(levels_to_fit) + 1)
    event_time = lower + (upper - lower) / denominator
    if not lower < event_time < upper:
        event_time = float(np.nextafter(lower, upper))
    if not lower < event_time < upper:
        raise ValueError(
            "cannot place synthetic recombination time in available "
            f"floating-point gap: lower={lower}, upper={upper}"
        )
    return float(event_time)


def _event_schedule_arrays(
    tables: tskit.TableCollection,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return one row per ARG event plus sorted fixed sample times."""
    node_flags = np.asarray(tables.nodes.flags, dtype=np.uint32)
    node_time = np.asarray(tables.nodes.time, dtype=np.float64)
    edge_parent = np.asarray(tables.edges.parent, dtype=np.int32)
    edge_child = np.asarray(tables.edges.child, dtype=np.int32)
    num_nodes = int(node_time.size)

    unique_child_count = np.zeros(num_nodes, dtype=np.int32)
    first_child = np.full(num_nodes, -1, dtype=np.int32)
    if edge_parent.size:
        starts_parent = np.empty(edge_parent.size, dtype=bool)
        starts_parent[0] = True
        starts_parent[1:] = edge_parent[1:] != edge_parent[:-1]
        parent_starts = np.flatnonzero(starts_parent)
        first_child[edge_parent[parent_starts]] = edge_child[parent_starts]

        starts_pair = np.empty(edge_parent.size, dtype=bool)
        starts_pair[0] = True
        starts_pair[1:] = (
            (edge_parent[1:] != edge_parent[:-1])
            | (edge_child[1:] != edge_child[:-1])
        )
        unique_child_count = np.bincount(
            edge_parent[starts_pair],
            minlength=num_nodes,
        ).astype(np.int32, copy=False)

    recombination_nodes = np.flatnonzero(
        (node_flags & NODE_IS_RE_EVENT) != 0
    ).astype(np.int32)
    if recombination_nodes.size % 2:
        raise ValueError(
            "globally unique event times require an even number of "
            "recombination nodes"
        )
    left_nodes = recombination_nodes[0::2]
    right_nodes = recombination_nodes[1::2]
    if left_nodes.size:
        valid_pairs = (
            (unique_child_count[left_nodes] == 1)
            & (unique_child_count[right_nodes] == 1)
            & (first_child[left_nodes] == first_child[right_nodes])
            & (node_time[left_nodes] == node_time[right_nodes])
        )
        if not np.all(valid_pairs):
            pair_index = int(np.flatnonzero(~valid_pairs)[0])
            raise ValueError(
                "globally unique event times require consecutive recombination "
                "nodes to be paired by matching time and child; nodes "
                f"{int(left_nodes[pair_index])}, "
                f"{int(right_nodes[pair_index])} violate this"
            )

    sample_mask = (node_flags & tskit.NODE_IS_SAMPLE) != 0
    recombination_mask = np.zeros(num_nodes, dtype=bool)
    recombination_mask[recombination_nodes] = True
    other_nodes = np.flatnonzero(~(sample_mask | recombination_mask)).astype(np.int32)

    event_count = int(left_nodes.size + other_nodes.size)
    event_time = np.empty(event_count, dtype=np.float64)
    event_node1 = np.empty(event_count, dtype=np.int32)
    event_node2 = np.full(event_count, -1, dtype=np.int32)
    event_priority = np.empty(event_count, dtype=np.uint8)

    recombination_count = int(left_nodes.size)
    if recombination_count:
        event_time[:recombination_count] = node_time[left_nodes]
        event_node1[:recombination_count] = left_nodes
        event_node2[:recombination_count] = right_nodes
        event_priority[:recombination_count] = _EVENT_KIND_RECOMBINATION

    if other_nodes.size:
        start = recombination_count
        counts = unique_child_count[other_nodes]
        priority = np.full(other_nodes.size, _EVENT_KIND_REVEAL, dtype=np.uint8)
        priority[counts == 1] = _EVENT_KIND_UNARY
        priority[counts >= 2] = _EVENT_KIND_COALESCENCE
        event_time[start:] = node_time[other_nodes]
        event_node1[start:] = other_nodes
        event_priority[start:] = priority

    sample_times = np.sort(node_time[sample_mask].astype(np.float64, copy=True))
    return event_time, event_node1, event_node2, event_priority, sample_times


def _event_schedule_order(
    event_time: np.ndarray,
    event_node1: np.ndarray,
    event_node2: np.ndarray,
    event_priority: np.ndarray,
) -> np.ndarray:
    return np.lexsort((event_node2, event_node1, event_priority, event_time))


def _summarize_event_times(
    tables: tskit.TableCollection,
) -> _EventTimeAdjustment:
    event_time, event_node1, event_node2, priority, _sample_times = (
        _event_schedule_arrays(tables)
    )
    order = _event_schedule_order(event_time, event_node1, event_node2, priority)
    ordered_time = event_time[order]
    globally_unique = bool(
        ordered_time.size < 2 or np.all(ordered_time[1:] > ordered_time[:-1])
    )
    return _EventTimeAdjustment(
        adjusted_event_count=0,
        max_adjustment=0.0,
        times_are_globally_unique=globally_unique,
    )


def _try_backward_event_segment(
    original_time: np.ndarray,
    base_step: float,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray | None:
    """Linearize a sample-bounded segment, propagating dense ties backward."""
    size = int(original_time.size)
    values = np.empty(size, dtype=np.float64)
    values[-1] = float(original_time[-1])
    if np.isfinite(upper_bound) and not values[-1] < upper_bound:
        return None

    step = base_step
    if np.isfinite(lower_bound):
        available = float(original_time[0]) - lower_bound
        if not available > 0.0:
            return None
        step = min(step, available / (2 * (size + 1)))

    direction = lower_bound if np.isfinite(lower_bound) else -np.inf
    for index in range(size - 2, -1, -1):
        next_time = float(values[index + 1])
        if original_time[index] == original_time[index + 1]:
            candidate = min(float(original_time[index]), next_time - step)
        elif original_time[index] < next_time:
            candidate = float(original_time[index])
        else:
            candidate = float(np.nextafter(next_time, direction))

        if not np.isfinite(candidate) or not candidate < next_time:
            candidate = float(np.nextafter(next_time, direction))
        if not np.isfinite(candidate) or not candidate < next_time:
            return None
        if np.isfinite(lower_bound) and not candidate > lower_bound:
            return None
        values[index] = candidate
    return values


def _try_forward_event_segment(
    original_time: np.ndarray,
    base_step: float,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray | None:
    """Linearize a sample-bounded segment, propagating dense ties forward."""
    size = int(original_time.size)
    values = np.empty(size, dtype=np.float64)
    values[0] = float(original_time[0])
    if np.isfinite(lower_bound) and not values[0] > lower_bound:
        values[0] = float(np.nextafter(lower_bound, upper_bound))
    if not np.isfinite(values[0]) or (
        np.isfinite(upper_bound) and not values[0] < upper_bound
    ):
        return None

    step = base_step
    if np.isfinite(upper_bound):
        available = upper_bound - values[0]
        if not available > 0.0:
            return None
        step = min(step, available / (size + 1))

    direction = upper_bound if np.isfinite(upper_bound) else np.inf
    for index in range(1, size):
        previous_time = float(values[index - 1])
        if original_time[index] == original_time[index - 1]:
            candidate = max(float(original_time[index]), previous_time + step)
        elif original_time[index] > previous_time:
            candidate = float(original_time[index])
        else:
            candidate = float(np.nextafter(previous_time, direction))

        if not np.isfinite(candidate) or not candidate > previous_time:
            candidate = float(np.nextafter(previous_time, direction))
        if not np.isfinite(candidate) or not candidate > previous_time:
            return None
        if np.isfinite(upper_bound) and not candidate < upper_bound:
            return None
        values[index] = candidate
    return values


def _linearize_event_segment(
    original_time: np.ndarray,
    base_step: float,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    values = _try_backward_event_segment(
        original_time,
        base_step,
        lower_bound,
        upper_bound,
    )
    if values is None:
        values = _try_forward_event_segment(
            original_time,
            base_step,
            lower_bound,
            upper_bound,
        )
    if values is None:
        raise ValueError(
            "cannot assign globally unique ARG event times between fixed sample "
            f"times: event_count={original_time.size}, "
            f"lower_bound={lower_bound}, upper_bound={upper_bound}"
        )
    return values


def _ensure_globally_unique_event_times(
    tables: tskit.TableCollection,
) -> _EventTimeAdjustment:
    node_time = np.asarray(tables.nodes.time, dtype=np.float64).copy()
    event_time, event_node1, event_node2, priority, sample_times = (
        _event_schedule_arrays(tables)
    )
    if event_time.size == 0:
        return _EventTimeAdjustment(
            adjusted_event_count=0,
            max_adjustment=0.0,
            times_are_globally_unique=True,
        )
    if not np.all(np.isfinite(event_time)):
        raise ValueError("ARG event times must be finite")

    order = _event_schedule_order(event_time, event_node1, event_node2, priority)
    ordered_time = event_time[order]
    finite_node_time = node_time[np.isfinite(node_time)]
    max_node_time = (
        float(np.max(np.abs(finite_node_time))) if finite_node_time.size else 0.0
    )
    base_step = max(_MIN_TIME_EPSILON, max_node_time * _TIME_EPSILON_SCALE)

    adjusted_ordered_time = ordered_time.copy()
    unique_sample_times = np.unique(sample_times)
    start = 0
    while start < ordered_time.size:
        sample_index = int(
            np.searchsorted(
                unique_sample_times,
                ordered_time[start],
                side="right",
            )
        )
        lower_bound = (
            float(unique_sample_times[sample_index - 1])
            if sample_index > 0
            else -np.inf
        )
        upper_bound = (
            float(unique_sample_times[sample_index])
            if sample_index < unique_sample_times.size
            else np.inf
        )
        stop = (
            int(np.searchsorted(ordered_time, upper_bound, side="left"))
            if np.isfinite(upper_bound)
            else int(ordered_time.size)
        )
        segment_time = ordered_time[start:stop]
        if segment_time.size >= 2 and np.any(
            segment_time[1:] <= segment_time[:-1]
        ):
            adjusted_ordered_time[start:stop] = _linearize_event_segment(
                segment_time,
                base_step,
                lower_bound,
                upper_bound,
            )
        start = stop

    if adjusted_ordered_time.size >= 2 and not np.all(
        adjusted_ordered_time[1:] > adjusted_ordered_time[:-1]
    ):
        raise RuntimeError("failed to construct globally unique ARG event times")

    adjusted_event_time = event_time.copy()
    adjusted_event_time[order] = adjusted_ordered_time
    adjustment = np.abs(adjusted_event_time - event_time)
    adjusted_event_count = int(np.count_nonzero(adjustment))
    max_adjustment = float(np.max(adjustment)) if adjustment.size else 0.0
    node_time[event_node1] = adjusted_event_time
    paired = event_node2 >= 0
    node_time[event_node2[paired]] = adjusted_event_time[paired]

    edge_parent = np.asarray(tables.edges.parent, dtype=np.int32)
    edge_child = np.asarray(tables.edges.child, dtype=np.int32)
    bad_edges = np.flatnonzero(node_time[edge_parent] <= node_time[edge_child])
    if bad_edges.size:
        edge_id = int(bad_edges[0])
        raise ValueError(
            "event-time adjustment would violate parent.time > child.time; "
            f"edge {edge_id} has parent={int(edge_parent[edge_id])}, "
            f"child={int(edge_child[edge_id])}"
        )

    nodes = tables.nodes
    nodes.set_columns(
        flags=nodes.flags,
        time=node_time,
        population=nodes.population,
        individual=nodes.individual,
        metadata=nodes.metadata,
        metadata_offset=nodes.metadata_offset,
    )
    return _EventTimeAdjustment(
        adjusted_event_count=adjusted_event_count,
        max_adjustment=max_adjustment,
        times_are_globally_unique=True,
    )


def _mark_mutation_times_unknown(tables: tskit.TableCollection) -> None:
    mutations = tables.mutations
    if mutations.num_rows == 0:
        return
    mutations.set_columns(
        site=mutations.site,
        node=mutations.node,
        parent=mutations.parent,
        time=np.full(mutations.num_rows, tskit.UNKNOWN_TIME),
        derived_state=mutations.derived_state,
        derived_state_offset=mutations.derived_state_offset,
        metadata=mutations.metadata,
        metadata_offset=mutations.metadata_offset,
    )


def _add_provenance(
    *,
    tables: tskit.TableCollection,
    original_num_nodes: int,
    candidate_count: int,
    synthetic_event_count: int,
    time_rule: str,
    split_rule: str,
    ensure_unique_event_times: bool,
    time_adjustment: _EventTimeAdjustment,
) -> None:
    record = {
        "schema_version": "1.0.0",
        "software": {"name": SYNTHETIC_FULL_ARG_PROVENANCE_NAME},
        "parameters": {
            "time_rule": time_rule,
            "split_rule": split_rule,
            "node_flag": NODE_IS_RE_EVENT,
            "ensure_unique_event_times": bool(ensure_unique_event_times),
            "event_time_adjustment_rule": (
                "scale_aware_bidirectional_spacing"
                if ensure_unique_event_times
                else "none"
            ),
        },
        "summary": {
            "original_num_nodes": int(original_num_nodes),
            "candidate_count": int(candidate_count),
            "synthetic_recombination_event_count": int(synthetic_event_count),
            "synthetic_recombination_node_count": int(synthetic_event_count) * 2,
            "imputed_times_are_synthetic": True,
            "mutation_times_set_unknown": True,
            "event_times_are_globally_unique": (
                time_adjustment.times_are_globally_unique
            ),
            "event_time_adjusted_event_count": int(
                time_adjustment.adjusted_event_count
            ),
            "max_event_time_adjustment": float(time_adjustment.max_adjustment),
        },
    }
    tables.provenances.add_row(record=json.dumps(record, sort_keys=True))
