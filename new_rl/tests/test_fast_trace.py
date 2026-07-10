import numpy as np
import pytest

from argscape import build_synthetic_full_arg
from new_rl import build_fast_trace_from_full_arg, build_trace_from_full_arg


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


@pytest.fixture(scope="module")
def synthetic_ts():
    return build_synthetic_full_arg(SOURCE_TREES).tree_sequence


@pytest.fixture(scope="module")
def slow_trace(synthetic_ts):
    return build_trace_from_full_arg(synthetic_ts)


@pytest.fixture(scope="module")
def fast_trace(synthetic_ts):
    return build_fast_trace_from_full_arg(synthetic_ts)


def test_fast_strict_loader_requires_explicit_recombination_nodes():
    with pytest.raises(ValueError, match="no explicit recombination nodes"):
        build_fast_trace_from_full_arg(SOURCE_TREES)


def test_fast_trace_matches_slow_event_summary(fast_trace, slow_trace):
    assert fast_trace.event_count == slow_trace.event_count
    assert fast_trace.num_steps == slow_trace.num_steps
    assert fast_trace.recombination_event_count == slow_trace.recombination_event_count
    assert fast_trace.coalescence_event_count == slow_trace.coalescence_event_count

    indices = sorted({0, min(5, fast_trace.event_count - 1), fast_trace.event_count - 1})
    for index in indices:
        fast_event = fast_trace.event_at_index(index)
        slow_event = slow_trace.event_at_index(index)
        assert fast_event.kind == slow_event.kind
        assert fast_event.time == slow_event.time
        assert fast_event.node_ids == slow_event.node_ids
        assert fast_event.edge_ids == slow_event.edge_ids


def test_fast_trace_final_state_reveals_source_tables(fast_trace, synthetic_ts):
    final_state = fast_trace.state_at_step(fast_trace.num_steps)

    assert set(final_state.visible_node_ids.tolist()) == set(range(synthetic_ts.num_nodes))
    assert set(final_state.visible_edge_ids.tolist()) == set(range(synthetic_ts.num_edges))
    with pytest.raises(RuntimeError, match="active segments were not materialized"):
        final_state.active_lineages


def test_fast_trace_optional_active_segments_match_slow_trace(fast_trace, slow_trace):
    step = min(6, fast_trace.num_steps)
    fast_state = fast_trace.state_at_step(step, include_active=True)
    slow_state = slow_trace.state_at_step(step)

    assert _active_signature(fast_state) == _active_signature(slow_state)
    assert set(fast_state.visible_node_ids.tolist()) == set(
        slow_state.visible_node_ids.tolist()
    )
    assert set(fast_state.visible_edge_ids.tolist()) == set(
        slow_state.visible_edge_ids.tolist()
    )


def test_fast_state_matches_slow_trace_at_every_step_forward_and_backward(
    fast_trace,
    slow_trace,
):
    state = fast_trace.initial_state(chunk_size=4, initial_segment_capacity=1)

    for step in range(fast_trace.num_steps + 1):
        state.move_to(step)
        frontier = state.compact_active_frontier()
        expected = _active_signature(slow_trace.state_at_step(step))
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
        expected = _active_signature(slow_trace.state_at_step(step))
        observed = _compact_signature(state.compact_active_frontier())
        assert tuple(sorted(observed)) == expected

    assert state.step == 0
    assert state.active_count == fast_trace.sample_nodes.size
    assert state.segment_count == fast_trace.sample_nodes.size


def test_fast_state_chunking_clone_and_snapshot_are_independent(fast_trace, slow_trace):
    target = min(12, fast_trace.num_steps)
    single_step = fast_trace.initial_state(chunk_size=1)
    chunked = fast_trace.initial_state(chunk_size=7)
    single_step.advance(target)
    chunked.advance_to(target)

    expected = _active_signature(slow_trace.state_at_step(target))
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


def test_fast_state_structural_properties_and_compatibility_state(fast_trace, slow_trace):
    step = min(6, fast_trace.num_steps)
    state = fast_trace.initial_state().advance_to(step)
    compatibility_state = state.as_trace_state()
    direct_state = fast_trace.state_at_step(step, include_active=True)
    expected = _active_signature(slow_trace.state_at_step(step))

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
