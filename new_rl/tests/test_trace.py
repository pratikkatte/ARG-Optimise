import numpy as np
import pytest
import tskit

from argscape import NODE_IS_RE_EVENT, build_synthetic_full_arg
from new_rl import build_fast_trace_from_full_arg


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


def _two_independent_recombination_candidates():
    tables = tskit.TableCollection(sequence_length=2.0)
    children = [
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
        for _ in range(2)
    ]
    parents = [tables.nodes.add_row(flags=0, time=10.0) for _ in range(2)]
    for child in children:
        tables.edges.add_row(left=0.0, right=1.0, parent=parents[0], child=child)
        tables.edges.add_row(left=1.0, right=2.0, parent=parents[1], child=child)
    tables.sort()
    return tables.tree_sequence()


def _tied_original_coalescences():
    tables = tskit.TableCollection(sequence_length=1.0)
    samples = [
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
        for _ in range(4)
    ]
    parents = [tables.nodes.add_row(flags=0, time=10.0) for _ in range(2)]
    tables.edges.add_row(0.0, 1.0, parent=parents[0], child=samples[0])
    tables.edges.add_row(0.0, 1.0, parent=parents[0], child=samples[1])
    tables.edges.add_row(0.0, 1.0, parent=parents[1], child=samples[2])
    tables.edges.add_row(0.0, 1.0, parent=parents[1], child=samples[3])
    tables.sort()
    return tables.tree_sequence(), tuple(parents)


def _dense_tied_events_below_adjacent_float(group_size=69):
    tables = tskit.TableCollection(sequence_length=1.0)
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
    tied_time = 0.7142799014136905
    tied_nodes = [
        tables.nodes.add_row(flags=0, time=tied_time)
        for _ in range(group_size)
    ]
    upper_time = float(np.nextafter(tied_time, np.inf))
    upper_node = tables.nodes.add_row(flags=0, time=upper_time)
    tables.sort()
    return (
        tables.tree_sequence(),
        np.asarray(tied_nodes, dtype=np.int32),
        upper_node,
        tied_time,
        upper_time,
    )


def _dense_tied_events_between_adjacent_floats(group_size=87):
    tables = tskit.TableCollection(sequence_length=1.0)
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
    tied_time = 0.9999949137369791
    lower_time = float(np.nextafter(tied_time, -np.inf))
    upper_time = float(np.nextafter(tied_time, np.inf))
    lower_node = tables.nodes.add_row(flags=0, time=lower_time)
    tied_nodes = [
        tables.nodes.add_row(flags=0, time=tied_time)
        for _ in range(group_size)
    ]
    upper_node = tables.nodes.add_row(flags=0, time=upper_time)
    tables.sort()
    return (
        tables.tree_sequence(),
        lower_node,
        np.asarray(tied_nodes, dtype=np.int32),
        upper_node,
        tied_time,
        upper_time,
    )


def _assert_strict_edge_times(ts):
    tables = ts.tables
    assert np.all(
        tables.nodes.time[tables.edges.parent] > tables.nodes.time[tables.edges.child]
    )


def test_dense_tied_group_uses_available_space_below_adjacent_float():
    ts, tied_nodes, upper_node, tied_time, upper_time = (
        _dense_tied_events_below_adjacent_float()
    )

    result = build_synthetic_full_arg(ts)
    adjusted_time = np.asarray(result.tree_sequence.nodes_time)
    tied_adjusted_time = adjusted_time[tied_nodes]

    assert np.all(np.diff(tied_adjusted_time) > 0)
    assert tied_adjusted_time[0] < tied_time
    assert tied_adjusted_time[-1] == tied_time
    assert adjusted_time[upper_node] == upper_time
    assert result.metadata["event_time_adjusted_event_count"] == tied_nodes.size - 1
    assert result.metadata["max_event_time_adjustment"] > 0

    trace = build_fast_trace_from_full_arg(
        result.tree_sequence,
        strict=False,
        require_unique_event_times=True,
    )
    assert np.all(np.diff(trace.event_time) > 0)


def test_dense_tied_group_propagates_past_both_adjacent_floats():
    ts, lower_node, tied_nodes, upper_node, tied_time, upper_time = (
        _dense_tied_events_between_adjacent_floats()
    )

    result = build_synthetic_full_arg(ts)
    adjusted_time = np.asarray(result.tree_sequence.nodes_time)

    assert adjusted_time[lower_node] < adjusted_time[tied_nodes[0]]
    assert np.all(np.diff(adjusted_time[tied_nodes]) > 0)
    assert adjusted_time[tied_nodes[-1]] == tied_time
    assert adjusted_time[upper_node] == upper_time
    trace = build_fast_trace_from_full_arg(
        result.tree_sequence,
        strict=False,
        require_unique_event_times=True,
    )
    assert np.all(np.diff(trace.event_time) > 0)


def test_strict_loader_requires_explicit_recombination_nodes():
    with pytest.raises(ValueError, match="no explicit recombination nodes"):
        build_fast_trace_from_full_arg(SOURCE_TREES)


def test_trace_replays_synthetic_full_arg_to_final_graph():
    result = build_synthetic_full_arg(SOURCE_TREES)
    ts = result.tree_sequence
    trace = build_fast_trace_from_full_arg(ts, require_unique_event_times=True)

    assert trace.recombination_event_count == result.metadata[
        "synthetic_recombination_event_count"
    ]
    assert trace.num_steps == trace.event_count

    final_state = trace.state_at_step(trace.num_steps)
    assert set(final_state.visible_node_ids.tolist()) == set(range(ts.num_nodes))
    assert set(final_state.visible_edge_ids.tolist()) == set(range(ts.num_edges))

    assert np.all(trace.node_time[trace.edge_parent] > trace.node_time[trace.edge_child])
    assert np.all(np.diff(trace.event_time) > 0)
    assert sum(
        1 for node in ts.nodes() if node.flags & NODE_IS_RE_EVENT
    ) == trace.recombination_event_count * 2


def test_previous_state_traces_back_one_event():
    trace = build_fast_trace_from_full_arg(
        build_synthetic_full_arg(SOURCE_TREES).tree_sequence
    )

    step = min(6, trace.num_steps)
    previous = trace.initial_state().advance_to(step).backtrack().as_trace_state()
    expected = trace.state_at_step(step - 1, include_active=True)

    assert previous.step == step - 1
    assert _active_signature(previous) == _active_signature(expected)
    assert set(previous.visible_node_ids.tolist()) == set(expected.visible_node_ids.tolist())
    assert set(previous.visible_edge_ids.tolist()) == set(expected.visible_edge_ids.tolist())


def test_windowed_graph_materialization():
    trace = build_fast_trace_from_full_arg(
        build_synthetic_full_arg(SOURCE_TREES).tree_sequence
    )
    graph = trace.graph_at_step(trace.num_steps, genomic_range=(5000, 7000))

    assert graph["metadata"]["genomic_range"] == [5000.0, 7000.0]
    assert graph["nodes"]
    assert graph["edges"]
    assert all(5000 <= edge["left"] < edge["right"] <= 7000 for edge in graph["edges"])


def test_trace_uses_column_arrays_for_source_tables():
    trace = build_fast_trace_from_full_arg(
        build_synthetic_full_arg(SOURCE_TREES).tree_sequence
    )

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


def test_independent_recombination_events_receive_globally_unique_times():
    result = build_synthetic_full_arg(_two_independent_recombination_candidates())
    trace = build_fast_trace_from_full_arg(
        result.tree_sequence,
        require_unique_event_times=True,
    )

    recombination_times = [
        trace.event_at_index(index).time
        for index in range(trace.event_count)
        if trace.event_at_index(index).kind == "recombination"
    ]
    assert len(recombination_times) == 2
    assert recombination_times[1] > recombination_times[0]
    assert result.metadata["event_times_are_globally_unique"] is True
    assert result.metadata["event_time_adjusted_event_count"] >= 1
    assert result.metadata["max_event_time_adjustment"] > 0.0


def test_parallel_balanced_recombination_events_receive_unique_times():
    ts, _parents = _many_interval_child_tree_sequence(group_count=4)
    result = build_synthetic_full_arg(ts, split_rule="balanced")
    trace = build_fast_trace_from_full_arg(
        result.tree_sequence,
        require_unique_event_times=True,
    )

    assert np.all(np.diff(trace.event_time) > 0)
    assert result.metadata["event_time_adjusted_event_count"] >= 1
    _assert_strict_edge_times(result.tree_sequence)


def test_tied_original_coalescence_events_are_deconflicted():
    ts, parents = _tied_original_coalescences()
    result = build_synthetic_full_arg(ts)
    adjusted_times = result.tree_sequence.tables.nodes.time[list(parents)]

    assert adjusted_times[1] > adjusted_times[0]
    assert result.metadata["event_time_adjusted_event_count"] == 1
    assert result.metadata["event_times_are_globally_unique"] is True
    _assert_strict_edge_times(result.tree_sequence)


def test_unique_event_time_opt_out_preserves_ties():
    result = build_synthetic_full_arg(
        _two_independent_recombination_candidates(),
        ensure_unique_event_times=False,
    )
    trace = build_fast_trace_from_full_arg(result.tree_sequence)

    assert np.any(np.diff(trace.event_time) == 0)
    assert result.metadata["ensure_unique_event_times"] is False
    assert result.metadata["event_times_are_globally_unique"] is False
    with pytest.raises(ValueError, match="strictly increasing event times"):
        build_fast_trace_from_full_arg(
            result.tree_sequence,
            require_unique_event_times=True,
        )


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
    for synthetic_ts in (balanced, left_to_right):
        trace = build_fast_trace_from_full_arg(
            synthetic_ts,
            require_unique_event_times=True,
        )
        assert np.all(np.diff(trace.event_time) > 0)


@pytest.mark.parametrize("path", [L1MB_DATED_TREES, SIM_L1MB_TREES])
def test_larger_local_fixtures_build_with_strict_synthetic_times(path):
    result = build_synthetic_full_arg(path)
    ts = result.tree_sequence

    assert result.metadata["synthetic_recombination_event_count"] > 0
    _assert_strict_edge_times(ts)

    trace = build_fast_trace_from_full_arg(ts, require_unique_event_times=True)
    assert trace.recombination_event_count == result.metadata[
        "synthetic_recombination_event_count"
    ]
    assert np.all(np.diff(trace.event_time) > 0)
