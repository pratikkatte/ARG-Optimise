import numpy as np
import pytest
import tskit

from argscape import NODE_IS_RE_EVENT, build_synthetic_full_arg
from new_rl import build_trace_from_full_arg


SOURCE_TREES = "arg/validation/output/tsinfer/l25kb_dated.trees"
L1MB_DATED_TREES = "arg/validation/output/tsinfer/l1mb_dated.trees"
SIM_L1MB_TREES = "arg/validation/trees/sim_l1mb_0.trees"


def _active_signature(state):
    return tuple(
        (lineage.node_id, lineage.segments)
        for lineage in state.active_lineages
    )


def _many_interval_child_tree_sequence(group_count=16):
    tables = tskit.TableCollection(sequence_length=float(group_count))
    child = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
    parents = [
        tables.nodes.add_row(flags=0, time=10.0)
        for _ in range(group_count)
    ]
    for index, parent in enumerate(parents):
        tables.edges.add_row(
            left=float(index),
            right=float(index + 1),
            parent=parent,
            child=child,
        )
    tables.sort()
    return tables.tree_sequence(), tuple(parents)


def _assert_strict_edge_times(ts):
    tables = ts.tables
    assert np.all(
        tables.nodes.time[tables.edges.parent] > tables.nodes.time[tables.edges.child]
    )


def test_strict_loader_requires_explicit_recombination_nodes():
    with pytest.raises(ValueError, match="no explicit recombination nodes"):
        build_trace_from_full_arg(SOURCE_TREES)


def test_trace_replays_synthetic_full_arg_to_final_graph():
    result = build_synthetic_full_arg(SOURCE_TREES)
    ts = result.tree_sequence
    trace = build_trace_from_full_arg(ts)

    assert trace.recombination_event_count == result.metadata[
        "synthetic_recombination_event_count"
    ]
    assert trace.num_steps == trace.event_count

    final_state = trace.state_at_step(trace.num_steps)
    assert set(final_state.visible_node_ids.tolist()) == set(range(ts.num_nodes))
    assert set(final_state.visible_edge_ids.tolist()) == set(range(ts.num_edges))

    assert np.all(trace.node_time[trace.edge_parent] > trace.node_time[trace.edge_child])
    assert sum(
        1 for node in ts.nodes() if node.flags & NODE_IS_RE_EVENT
    ) == trace.recombination_event_count * 2


def test_previous_state_traces_back_one_event():
    trace = build_trace_from_full_arg(build_synthetic_full_arg(SOURCE_TREES).tree_sequence)

    step = min(6, trace.num_steps)
    state = trace.state_at_step(step)
    previous = trace.previous_state(state)
    expected = trace.state_at_step(step - 1)

    assert previous.step == step - 1
    assert _active_signature(previous) == _active_signature(expected)
    assert set(previous.visible_node_ids.tolist()) == set(expected.visible_node_ids.tolist())
    assert set(previous.visible_edge_ids.tolist()) == set(expected.visible_edge_ids.tolist())


def test_windowed_graph_materialization():
    trace = build_trace_from_full_arg(build_synthetic_full_arg(SOURCE_TREES).tree_sequence)
    graph = trace.graph_at_step(trace.num_steps, genomic_range=(5000, 7000))

    assert graph["metadata"]["genomic_range"] == [5000.0, 7000.0]
    assert graph["nodes"]
    assert graph["edges"]
    assert all(5000 <= edge["left"] < edge["right"] <= 7000 for edge in graph["edges"])


def test_trace_uses_column_arrays_for_source_tables():
    trace = build_trace_from_full_arg(build_synthetic_full_arg(SOURCE_TREES).tree_sequence)

    assert isinstance(trace.node_time, np.ndarray)
    assert isinstance(trace.edge_parent, np.ndarray)
    assert isinstance(trace.event_kind, np.ndarray)
    assert trace.edge_parent.shape[0] == trace.edge_child.shape[0]


def test_synthetic_recombination_nodes_are_msprime_style_pairs():
    ts, _parents = _many_interval_child_tree_sequence(group_count=8)
    result = build_synthetic_full_arg(ts)
    synthetic_ts = result.tree_sequence
    tables = synthetic_ts.tables

    recombination_nodes = np.flatnonzero((tables.nodes.flags & NODE_IS_RE_EVENT) != 0)
    assert recombination_nodes.size == result.metadata["synthetic_recombination_node_count"]

    grouped = {}
    for node_id in recombination_nodes:
        outgoing = np.flatnonzero(tables.edges.parent == node_id)
        children = np.unique(tables.edges.child[outgoing])
        assert children.size == 1
        key = (float(tables.nodes.time[node_id]), int(children[0]))
        grouped.setdefault(key, []).append(int(node_id))

    assert len(grouped) == result.metadata["synthetic_recombination_event_count"]
    assert all(len(nodes) == 2 for nodes in grouped.values())
    _assert_strict_edge_times(synthetic_ts)


def test_synthetic_topology_preserves_original_leaf_intervals():
    ts, parents = _many_interval_child_tree_sequence(group_count=12)
    result = build_synthetic_full_arg(ts)
    tables = result.tree_sequence.tables
    parent_set = set(parents)

    leaf_edges = {
        (float(left), float(right), int(parent))
        for left, right, parent in zip(
            tables.edges.left,
            tables.edges.right,
            tables.edges.parent,
        )
        if int(parent) in parent_set
    }
    expected = {
        (float(index), float(index + 1), int(parent))
        for index, parent in enumerate(parents)
    }
    assert leaf_edges == expected


def test_balanced_topology_is_smaller_than_left_to_right_chain():
    ts, _parents = _many_interval_child_tree_sequence(group_count=32)

    balanced = build_synthetic_full_arg(ts).tree_sequence
    left_to_right = build_synthetic_full_arg(ts, split_rule="left_to_right").tree_sequence

    assert balanced.num_edges < left_to_right.num_edges
    assert balanced.num_nodes == left_to_right.num_nodes
    _assert_strict_edge_times(balanced)
    _assert_strict_edge_times(left_to_right)


@pytest.mark.parametrize("path", [L1MB_DATED_TREES, SIM_L1MB_TREES])
def test_larger_local_fixtures_build_with_strict_synthetic_times(path):
    result = build_synthetic_full_arg(path)
    ts = result.tree_sequence

    assert result.metadata["synthetic_recombination_event_count"] > 0
    _assert_strict_edge_times(ts)

    trace = build_trace_from_full_arg(ts)
    assert trace.recombination_event_count == result.metadata[
        "synthetic_recombination_event_count"
    ]
