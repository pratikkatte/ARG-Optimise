import numpy as np
import pytest
import tskit

from arg.new_rl import build_fast_trace_from_full_arg, build_synthetic_full_arg


SOURCE_TREES = "arg/validation/output/tsinfer/l25kb_dated.trees"


def _active_signature(state):
    return tuple(
        (lineage.node_id, lineage.segments)
        for lineage in state.active_lineages
    )


def _compact_signature(frontier):
    return tuple(
        (int(node_id), frontier.segments_for_index(lineage_index))
        for lineage_index, node_id in enumerate(frontier.node_ids)
    )


def _merge_reference_segments(segments):
    merged = []
    for left, right in sorted(segments):
        if left >= right:
            continue
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return tuple(merged)


def _subtract_reference_segment(segments, left, right):
    updated = []
    covered = 0.0
    for segment_left, segment_right in segments:
        overlap_left = max(segment_left, left)
        overlap_right = min(segment_right, right)
        if overlap_left < overlap_right:
            covered += overlap_right - overlap_left
            if segment_left < overlap_left:
                updated.append((segment_left, overlap_left))
            if overlap_right < segment_right:
                updated.append((overlap_right, segment_right))
        else:
            updated.append((segment_left, segment_right))
    assert np.isclose(covered, right - left)
    return _merge_reference_segments(updated)


def _reference_active_signature(trace, step):
    active = {
        int(node_id): ((0.0, trace.sequence_length),)
        for node_id in trace.sample_nodes
    }
    for event_index in range(step):
        edge_start = int(trace.event_edge_start[event_index])
        edge_end = int(trace.event_edge_start[event_index + 1])
        edge_ids = trace.revealed_edge_ids[edge_start:edge_end]
        parent_segments = {}
        for edge_id in edge_ids:
            edge_id = int(edge_id)
            parent = int(trace.edge_parent[edge_id])
            child = int(trace.edge_child[edge_id])
            segment = (
                float(trace.edge_left[edge_id]),
                float(trace.edge_right[edge_id]),
            )
            active[child] = _subtract_reference_segment(
                active.get(child, ()),
                *segment,
            )
            if not active[child]:
                del active[child]
            parent_segments.setdefault(parent, []).append(segment)
        for parent, segments in parent_segments.items():
            active[parent] = _merge_reference_segments(
                active.get(parent, ()) + tuple(segments)
            )
    return tuple(sorted(active.items()))


@pytest.fixture(scope="module")
def synthetic_ts():
    return build_synthetic_full_arg(SOURCE_TREES).tree_sequence


@pytest.fixture(scope="module")
def fast_trace(synthetic_ts):
    return build_fast_trace_from_full_arg(
        synthetic_ts,
        require_unique_event_times=True,
    )


def _tied_synthetic_full_arg():
    tables = tskit.TableCollection(sequence_length=2.0)
    children = [
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
        for _ in range(2)
    ]
    parents = [tables.nodes.add_row(flags=0, time=10.0) for _ in range(2)]
    for child in children:
        tables.edges.add_row(0.0, 1.0, parent=parents[0], child=child)
        tables.edges.add_row(1.0, 2.0, parent=parents[1], child=child)
    tables.sort()
    return build_synthetic_full_arg(
        tables.tree_sequence(),
        ensure_unique_event_times=False,
    ).tree_sequence


def test_fast_strict_loader_requires_explicit_recombination_nodes():
    with pytest.raises(ValueError, match="no explicit recombination nodes"):
        build_fast_trace_from_full_arg(SOURCE_TREES)


def test_continuous_time_guard_rejects_tied_events():
    tied_ts = _tied_synthetic_full_arg()

    build_fast_trace_from_full_arg(tied_ts)
    with pytest.raises(ValueError, match="strictly increasing event times"):
        build_fast_trace_from_full_arg(
            tied_ts,
            require_unique_event_times=True,
        )


def test_event_summary_matches_25kb_fixture(fast_trace):
    assert fast_trace.event_count == 25
    assert fast_trace.num_steps == 25
    assert fast_trace.recombination_event_count == 13
    assert fast_trace.coalescence_event_count == 12

    indices = sorted({0, min(5, fast_trace.event_count - 1), fast_trace.event_count - 1})
    for index in indices:
        event = fast_trace.event_at_index(index)
        edge_start = int(fast_trace.event_edge_start[index])
        edge_end = int(fast_trace.event_edge_start[index + 1])
        expected_nodes = [int(fast_trace.event_node1[index])]
        if fast_trace.event_node2[index] >= 0:
            expected_nodes.append(int(fast_trace.event_node2[index]))
        assert event.step == index + 1
        assert event.time == fast_trace.event_time[index]
        assert event.node_ids == tuple(expected_nodes)
        assert event.edge_ids == tuple(
            int(edge_id)
            for edge_id in fast_trace.revealed_edge_ids[edge_start:edge_end]
        )


def test_fast_trace_final_state_reveals_source_tables(fast_trace, synthetic_ts):
    final_state = fast_trace.state_at_step(fast_trace.num_steps)

    assert set(final_state.visible_node_ids.tolist()) == set(range(synthetic_ts.num_nodes))
    assert set(final_state.visible_edge_ids.tolist()) == set(range(synthetic_ts.num_edges))
    with pytest.raises(RuntimeError, match="active segments were not materialized"):
        final_state.active_lineages


def test_optional_active_segments_match_reference_replay(fast_trace):
    step = min(6, fast_trace.num_steps)
    fast_state = fast_trace.state_at_step(step, include_active=True)

    assert _active_signature(fast_state) == _reference_active_signature(fast_trace, step)


def test_fast_state_matches_reference_at_every_step_forward_and_backward(fast_trace):
    state = fast_trace.initial_state(chunk_size=4, initial_segment_capacity=1)

    for step in range(fast_trace.num_steps + 1):
        state.move_to(step)
        frontier = state.compact_active_frontier()
        expected = _reference_active_signature(fast_trace, step)
        assert tuple(sorted(_compact_signature(frontier))) == expected
        expected_ids = {node_id for node_id, _segments in expected}
        expected_order = [
            int(node_id)
            for node_id in state.visible_node_ids
            if int(node_id) in expected_ids
        ]
        assert frontier.node_ids.tolist() == expected_order

    for step in range(fast_trace.num_steps - 1, -1, -1):
        state.move_to(step)
        expected = _reference_active_signature(fast_trace, step)
        observed = _compact_signature(state.compact_active_frontier())
        assert tuple(sorted(observed)) == expected

    assert state.step == 0
    assert state.active_count == fast_trace.sample_nodes.size
    assert state.segment_count == fast_trace.sample_nodes.size


def test_fast_state_chunking_clone_and_snapshot_are_independent(fast_trace):
    target = min(12, fast_trace.num_steps)
    single_step = fast_trace.initial_state(chunk_size=1)
    chunked = fast_trace.initial_state(chunk_size=7)
    single_step.advance(target)
    chunked.advance_to(target)

    expected = _reference_active_signature(fast_trace, target)
    single_signature = _compact_signature(single_step.compact_active_frontier())
    chunked_signature = _compact_signature(chunked.compact_active_frontier())
    assert tuple(sorted(single_signature)) == expected
    assert tuple(sorted(chunked_signature)) == expected

    snapshot = chunked.compact_active_frontier()
    snapshot_values = (
        snapshot.node_ids.copy(),
        snapshot.segment_offsets.copy(),
        snapshot.segment_left.copy(),
        snapshot.segment_right.copy(),
    )
    clone = chunked.clone()
    chunked.backtrack(min(3, target))
    clone.advance_to(min(fast_trace.num_steps, target + 3))

    assert chunked.step != clone.step
    assert np.array_equal(snapshot.node_ids, snapshot_values[0])
    assert np.array_equal(snapshot.segment_offsets, snapshot_values[1])
    assert np.array_equal(snapshot.segment_left, snapshot_values[2])
    assert np.array_equal(snapshot.segment_right, snapshot_values[3])
    with pytest.raises(ValueError):
        snapshot.node_ids[0] = -1


def test_fast_state_structural_properties_and_compatibility_state(fast_trace):
    step = min(6, fast_trace.num_steps)
    state = fast_trace.initial_state().advance_to(step)
    compatibility_state = state.as_trace_state()
    direct_state = fast_trace.state_at_step(step, include_active=True)
    expected = _reference_active_signature(fast_trace, step)

    assert state.step == step
    assert state.current_time == fast_trace.event_time[step - 1]
    assert not state.is_terminal
    assert compatibility_state.compact_active_frontier.segment_count == state.segment_count
    assert _active_signature(compatibility_state) == expected
    assert _active_signature(direct_state) == expected
    assert np.array_equal(state.visible_node_ids, compatibility_state.visible_node_ids)
    assert np.array_equal(state.visible_edge_ids, compatibility_state.visible_edge_ids)

    state.advance_to(fast_trace.num_steps)
    assert state.is_terminal
    with pytest.raises(ValueError, match="step must be"):
        state.advance()
    with pytest.raises(ValueError, match="step must be"):
        state.backtrack(fast_trace.num_steps + 1)


def test_fast_windowed_graph_materialization(fast_trace):
    graph = fast_trace.graph_at_step(
        fast_trace.num_steps,
        genomic_range=(5000, 7000),
        max_edges=None,
    )

    assert graph["metadata"]["genomic_range"] == [5000.0, 7000.0]
    assert graph["nodes"]
    assert graph["edges"]
    assert all(5000 <= edge["left"] < edge["right"] <= 7000 for edge in graph["edges"])


def test_fast_unwindowed_graph_materialization_guard(fast_trace):
    with pytest.raises(ValueError, match="unwindowed graph materialization"):
        fast_trace.graph_at_step(fast_trace.num_steps, max_edges=1)
