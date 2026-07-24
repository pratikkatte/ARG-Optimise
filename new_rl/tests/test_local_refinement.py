from pathlib import Path

import numpy as np
import pytest
import tskit

from arg.new_rl import (
    LocalRefinementRequest,
    build_fast_trace_from_full_arg,
    build_synthetic_full_arg,
    prepare_local_refinement,
    resolve_trace_cut,
    trace_local_dependencies,
)


ARG_ROOT = Path(__file__).resolve().parents[2]
SOURCE_25KB = ARG_ROOT / "validation/output/tsinfer/l25kb_dated.trees"


def _simple_chain_tree_sequence():
    tables = tskit.TableCollection(sequence_length=10.0)
    samples = [
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
        for _ in range(3)
    ]
    younger_parent = tables.nodes.add_row(time=5.0)
    root = tables.nodes.add_row(time=10.0)
    tables.edges.add_row(0.0, 10.0, parent=younger_parent, child=samples[0])
    tables.edges.add_row(0.0, 10.0, parent=younger_parent, child=samples[1])
    tables.edges.add_row(0.0, 10.0, parent=root, child=younger_parent)
    tables.edges.add_row(0.0, 10.0, parent=root, child=samples[2])
    tables.sort()
    return tables.tree_sequence()


def _simple_trace():
    conversion = build_synthetic_full_arg(_simple_chain_tree_sequence())
    return build_fast_trace_from_full_arg(
        conversion.tree_sequence,
        require_unique_event_times=True,
        allow_no_recombination=True,
    )


def _segments_by_edge(items):
    output = {}
    for item in items:
        if not item.edge_ids:
            continue
        for edge_id in item.edge_ids:
            output.setdefault(int(edge_id), []).extend(item.intervals)
    return {
        edge_id: tuple(sorted(segments))
        for edge_id, segments in output.items()
    }


def _merged(segments):
    output = []
    for left, right in sorted(segments):
        if output and left <= output[-1][1]:
            output[-1] = (output[-1][0], max(output[-1][1], right))
        else:
            output.append((left, right))
    return tuple(output)


def test_request_requires_exactly_one_cut_selector():
    with pytest.raises(ValueError, match="exactly one"):
        LocalRefinementRequest((0.0, 1.0))
    with pytest.raises(ValueError, match="exactly one"):
        LocalRefinementRequest(
            (0.0, 1.0),
            cut_time=1.0,
            cut_event_index=0,
        )
    with pytest.raises(ValueError, match="integer"):
        LocalRefinementRequest((0.0, 1.0), cut_event_index=1.5)


def test_time_and_event_cuts_resolve_to_the_younger_side():
    trace = _simple_trace()
    assert trace.event_time.tolist() == [5.0, 10.0]

    exact = resolve_trace_cut(
        trace,
        LocalRefinementRequest((2.0, 8.0), cut_time=5.0),
    )
    assert exact.cut_step == 0
    assert exact.current_time == 0.0
    assert exact.next_event_index == 0
    assert exact.next_event_time == 5.0
    assert exact.time_discrepancy == 0.0

    between = resolve_trace_cut(
        trace,
        LocalRefinementRequest((2.0, 8.0), cut_time=7.0),
    )
    assert between.cut_step == 1
    assert between.current_time == 5.0
    assert between.next_event_index == 1
    assert between.next_event_time == 10.0
    assert between.time_discrepancy == 3.0

    by_event = resolve_trace_cut(
        trace,
        LocalRefinementRequest((2.0, 8.0), cut_event_index=1),
    )
    assert by_event.cut_step == 1
    assert by_event.current_time == 5.0
    assert by_event.next_event_index == 1

    with pytest.raises(ValueError, match="older than"):
        resolve_trace_cut(
            trace,
            LocalRefinementRequest((2.0, 8.0), cut_time=11.0),
        )
    with pytest.raises(ValueError, match="cut_event_index"):
        resolve_trace_cut(
            trace,
            LocalRefinementRequest((2.0, 8.0), cut_event_index=2),
        )
    with pytest.raises(ValueError, match="genomic_range"):
        resolve_trace_cut(
            trace,
            LocalRefinementRequest((8.0, 11.0), cut_time=5.0),
        )


def test_preparation_handles_no_recombination_and_preserves_the_source():
    source = _simple_chain_tree_sequence()
    source_tables = source.dump_tables()
    prepared = prepare_local_refinement(
        source,
        LocalRefinementRequest((2.0, 8.0), cut_time=5.0),
    )
    context = prepared.context

    assert source.dump_tables().equals(source_tables)
    assert prepared.trace.recombination_event_count == 0
    assert context.is_valid
    assert context.selected_event_indices == (0, 1)
    assert {lineage.node_id for lineage in context.cut_active_lineages} == {
        0,
        1,
        2,
    }
    assert all(
        lineage.mutable_segments == ((2.0, 8.0),)
        for lineage in context.cut_active_lineages
    )
    assert all(
        lineage.fixed_segments == ((0.0, 2.0), (8.0, 10.0))
        for lineage in context.cut_active_lineages
    )
    assert {
        lineage.node_id for lineage in context.promoted_dependency_lineages
    } == {3, 4}
    assert all(
        not lineage.active_at_cut
        for lineage in context.promoted_dependency_lineages
    )
    assert all(
        lineage.first_active_step > context.resolved_cut.cut_step
        for lineage in context.promoted_dependency_lineages
    )
    assert not any(
        attachment.role == "terminal_anchor"
        for attachment in context.boundary_attachments
    )


def test_half_open_target_and_fixed_edge_complements_are_explicit():
    prepared = prepare_local_refinement(
        _simple_chain_tree_sequence(),
        LocalRefinementRequest((0.0, 5.0), cut_event_index=0),
    )
    context = prepared.context
    trace = prepared.trace

    assert context.is_valid
    assert all(
        (item.left, item.right) == (0.0, 5.0)
        for item in context.authorized_edge_intervals
    )
    fixed_by_edge = _segments_by_edge(context.boundary_attachments)
    for edge_id in context.authorized_edge_ids:
        assert fixed_by_edge[edge_id] == ((5.0, 10.0),)
        authorized = tuple(
            (item.left, item.right)
            for item in context.authorized_edge_intervals
            if item.edge_id == edge_id
        )
        assert _merged(authorized + fixed_by_edge[edge_id]) == (
            (
                float(trace.edge_left[edge_id]),
                float(trace.edge_right[edge_id]),
            ),
        )


def test_nonclosed_25kb_subregion_uses_mixed_boundaries():
    trace = _simple_25kb_trace()
    request = LocalRefinementRequest(
        (1000.0, 20_000.0),
        cut_event_index=17,
    )
    first = trace_local_dependencies(trace, request)
    second = trace_local_dependencies(trace, request)

    assert first == second
    assert first.is_valid
    assert first.selected_event_indices == (17, 18, 20, 24)
    assert any(event.mode == "mixed_boundary" for event in first.selected_events)
    assert any(
        attachment.role in {"outside_tether", "outside_edge_piece"}
        for attachment in first.boundary_attachments
    )
    assert all(
        1000.0 <= item.left < item.right <= 20_000.0
        for item in first.authorized_edge_intervals
    )
    assert 19 not in first.selected_event_indices
    assert 21 not in first.selected_event_indices
    assert 22 not in first.selected_event_indices
    assert 23 not in first.selected_event_indices


def _simple_25kb_trace():
    return build_fast_trace_from_full_arg(
        build_synthetic_full_arg(SOURCE_25KB).tree_sequence,
        require_unique_event_times=True,
    )


def test_unresolved_selected_event_returns_structured_diagnostic():
    trace = _simple_trace()
    parent = int(trace.event_node1[0])
    trace.node_reveal_step[parent] = trace.num_steps + 10

    context = trace_local_dependencies(
        trace,
        LocalRefinementRequest((2.0, 8.0), cut_event_index=0),
    )

    assert not context.is_valid
    assert context.status == "invalid"
    assert len(context.rejection_diagnostics) == 1
    diagnostic = context.rejection_diagnostics[0]
    assert diagnostic.code == "unresolved_event_coupling"
    assert diagnostic.event_index == 0
    assert diagnostic.node_ids == (parent,)
    assert diagnostic.edge_ids


def test_strict_loader_keeps_no_recombination_opt_in_narrow():
    conversion = build_synthetic_full_arg(_simple_chain_tree_sequence())

    with pytest.raises(ValueError, match="no explicit recombination nodes"):
        build_fast_trace_from_full_arg(
            conversion.tree_sequence,
            require_unique_event_times=True,
        )

    trace = build_fast_trace_from_full_arg(
        conversion.tree_sequence,
        require_unique_event_times=True,
        allow_no_recombination=True,
    )
    assert trace.recombination_event_count == 0
    assert np.all(np.diff(trace.event_time) > 0.0)
