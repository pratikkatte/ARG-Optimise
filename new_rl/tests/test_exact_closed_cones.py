import hashlib
import json

import numpy as np
import pytest
import tskit

from arg.new_rl import build_synthetic_full_arg
from arg.new_rl.exact_closed_cones import (
    _witness_rank,
    assert_synthetic_endpoints_are_normal_breakpoints,
    canonical_component_intervals,
    evaluate_two_stage_exact_cones,
    generate_normal_ts_candidates,
    scan_exact_region_witnesses,
)
from arg.new_rl.normal_ts_edge_closed_regions import (
    scan_normal_ts_edge_closed_regions,
)
from arg.new_rl.trace import build_fast_trace_from_full_arg


SOURCE_25KB = "arg/validation/output/tsinfer/l25kb_dated.trees"
SOURCE_1MB = "arg/validation/output/tsinfer/l1mb_dated.trees"

_SIGNATURE_FIELDS = (
    "left",
    "right",
    "first_closed_step",
    "first_closed_time",
    "last_closed_step",
    "last_closed_time",
    "valid_cut_steps",
    "separation_event_index",
    "separation_event_time",
    "separation_event_kind",
    "separation_event_node_ids",
    "lower_frontier_anchor_node_ids",
    "frontier_segments",
    "suffix_event_indices",
    "event_count",
    "recombination_event_count",
    "coalescence_event_count",
    "node_ids",
    "edge_ids",
    "terminal_lineage_ids",
)

_WITNESS_CERTIFICATE_FIELDS = (
    "left",
    "right",
    "boundary_step",
    "boundary_time",
    "lower_frontier_anchor_node_ids",
    "frontier_segments",
    "suffix_event_indices",
    "event_count",
    "recombination_event_count",
    "coalescence_event_count",
    "node_ids",
    "edge_ids",
    "terminal_lineage_ids",
    "separation_event_index",
    "separation_event_time",
    "separation_event_kind",
    "separation_event_node_ids",
)


def _cone_digest(cones):
    values = [[cone[field] for field in _SIGNATURE_FIELDS] for cone in cones]
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _tied_normal_ts():
    tables = tskit.TableCollection(sequence_length=3.0)
    child0 = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
    child1 = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
    parent0 = tables.nodes.add_row(time=10.0)
    parent1 = tables.nodes.add_row(time=10.0)
    # Both edges are one tied-time batch.  Processing them one at a time would
    # incorrectly expose [0, 2) as a direct component between the two edges.
    tables.edges.add_row(0.0, 2.0, parent=parent0, child=child0)
    tables.edges.add_row(1.0, 3.0, parent=parent1, child=child1)
    tables.sort()
    return tables.tree_sequence()


def _spanning_normal_ts():
    tables = tskit.TableCollection(sequence_length=3.0)
    child = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
    parent = tables.nodes.add_row(time=10.0)
    tables.edges.add_row(0.0, 2.0, parent=parent, child=child)
    tables.sort()
    return tables.tree_sequence()


def test_normal_candidates_use_half_open_breakpoint_intervals_and_exhaustive_coverage():
    catalog = generate_normal_ts_candidates(_tied_normal_ts())

    assert catalog.breakpoints == (0.0, 1.0, 2.0, 3.0)
    assert catalog.exhaustive_count == 5
    assert (0.0, 3.0) not in catalog.exhaustive_intervals
    assert all(left < right for left, right in catalog.exhaustive_intervals)
    assert catalog.exhaustive_intervals == {
        (0.0, 1.0),
        (0.0, 2.0),
        (1.0, 2.0),
        (1.0, 3.0),
        (2.0, 3.0),
    }


def test_tied_parent_times_are_processed_as_one_batch():
    catalog = generate_normal_ts_candidates(_tied_normal_ts())

    # [0, 2) is present as the adjacent two-atom diagnostic candidate before
    # the tied batch. It is not a direct component from an invalid intermediate
    # state inside that batch.
    candidate = catalog.candidates[(0.0, 2.0)]
    assert candidate.smallest_adjacency_tier == 2
    assert candidate.topology_generated


def test_edge_spanning_atoms_joins_the_covered_atomic_interval():
    catalog = generate_normal_ts_candidates(_spanning_normal_ts())

    candidate = catalog.candidates[(0.0, 2.0)]
    assert candidate.smallest_adjacency_tier == 1
    assert (0.0, 2.0) in catalog.intervals_for_tier(1)


def test_canonical_intervals_preserve_half_open_adjacency_and_disjoint_material():
    intervals = canonical_component_intervals(
        2,
        np.asarray([0, 0, 1, 1], dtype=np.int32),
        np.asarray([0.0, 1.0, 0.0, 2.0]),
        np.asarray([1.0, 2.0, 1.0, 3.0]),
    )

    assert intervals == [((0.0, 2.0),), ((0.0, 1.0), (2.0, 3.0))]


def test_synthetic_conversion_preserves_normal_breakpoints_on_25kb_fixture():
    normal_ts = tskit.load(SOURCE_25KB)
    synthetic_ts = build_synthetic_full_arg(normal_ts).tree_sequence
    assert_synthetic_endpoints_are_normal_breakpoints(normal_ts, synthetic_ts)


def test_witness_ranking_prefers_events_then_frontier_then_oldest_cut():
    def candidate(event_count, frontier_count, boundary_step):
        return {
            "event_count": event_count,
            "lower_frontier_anchor_node_ids": tuple(range(frontier_count)),
            "boundary_step": boundary_step,
        }

    assert _witness_rank(candidate(2, 9, 4)) < _witness_rank(candidate(3, 1, 20))
    assert _witness_rank(candidate(2, 3, 4)) < _witness_rank(candidate(2, 4, 20))
    assert _witness_rank(candidate(2, 3, 20)) < _witness_rank(candidate(2, 3, 4))


def test_incremental_exact_region_witnesses_match_the_25kb_oracle():
    normal_regions = scan_normal_ts_edge_closed_regions(SOURCE_25KB).regions
    synthetic_ts = build_synthetic_full_arg(SOURCE_25KB).tree_sequence
    trace = build_fast_trace_from_full_arg(
        synthetic_ts, require_unique_event_times=True
    )

    incremental = scan_exact_region_witnesses(trace, normal_regions)
    reference = scan_exact_region_witnesses(
        trace, normal_regions, algorithm="reference"
    )

    assert len(incremental.regions) == 2
    assert len(incremental.valid_witnesses) == 1
    assert incremental.valid_witnesses[0]["witness_cut_step"] == 17
    assert incremental.valid_witnesses[0]["witness_internal_event_count"] == 8
    assert incremental.valid_witnesses[0]["witness_frontier_node_count"] == 5
    assert incremental.regions[1]["status"] == "no_valid_witness"
    assert incremental.regions[1]["witness"] is None
    assert incremental.regions[1]["rejection_diagnostics"]["support_mismatch"] > 0
    assert incremental.diagnostics["event_insertions"] == trace.num_steps
    assert incremental.diagnostics["edge_insertions"] == trace.revealed_edge_ids.size
    assert incremental.diagnostics["suffix_rebuilds"] == 0
    assert reference.diagnostics["suffix_rebuilds"] == trace.num_steps + 1

    for incremental_region, reference_region in zip(
        incremental.regions, reference.regions
    ):
        assert incremental_region["status"] == reference_region["status"]
        if incremental_region["status"] == "valid":
            for field in _WITNESS_CERTIFICATE_FIELDS:
                assert incremental_region["witness"][field] == (
                    reference_region["witness"][field]
                )


def test_exact_region_witnesses_require_equal_half_open_boundaries():
    synthetic_ts = build_synthetic_full_arg(SOURCE_25KB).tree_sequence
    trace = build_fast_trace_from_full_arg(
        synthetic_ts, require_unique_event_times=True
    )
    result = scan_exact_region_witnesses(
        trace,
        (
            {"region_id": "exact", "left": 386.0, "right": 23963.0},
            {"region_id": "contained", "left": 1000.0, "right": 20000.0},
            {"region_id": "touching", "left": 23963.0, "right": 25000.0},
        ),
    )

    assert [region["status"] for region in result.regions] == [
        "valid",
        "no_valid_witness",
        "no_valid_witness",
    ]


def test_two_stage_25kb_matches_the_established_exact_oracle():
    evaluation = evaluate_two_stage_exact_cones(SOURCE_25KB)

    assert len(evaluation.exact_verified_cones) == 5
    assert evaluation.exact_scan.raw_exact_candidate_count == 24
    assert evaluation.recall_by_tier[2] == 1.0
    assert evaluation.recall_by_tier["exhaustive"] == 1.0
    assert _cone_digest(evaluation.exact_verified_cones) == (
        "dd7b78c5c9fd0dfd39d6869c6a34235e707eddddadad259125ab731f6d56954d"
    )
    for cone in evaluation.exact_verified_cones:
        edge_ids = np.asarray(cone["edge_ids"], dtype=np.int64)
        assert np.all(evaluation.trace.edge_left[edge_ids] >= cone["left"])
        assert np.all(evaluation.trace.edge_right[edge_ids] <= cone["right"])
        assert not cone["outside_overlap_edge_ids"]


@pytest.mark.slow
def test_two_stage_1mb_matches_the_established_exact_oracle():
    evaluation = evaluate_two_stage_exact_cones(SOURCE_1MB)

    assert len(evaluation.exact_verified_cones) == 141
    assert evaluation.exact_scan.raw_exact_candidate_count == 16_093
    assert evaluation.recall_by_tier[32] == 1.0
    assert evaluation.recall_by_tier["exhaustive"] == 1.0
    assert evaluation.candidate_count_by_tier["exhaustive"] == 85_077
    assert _cone_digest(evaluation.exact_verified_cones) == (
        "3868de476784c14c684b7d14783892a479e7d8c600f71be06b31d80d06cbeb25"
    )
    assert_synthetic_endpoints_are_normal_breakpoints(
        evaluation.normal_tree_sequence,
        evaluation.synthetic_arg,
    )

    normal_regions = scan_normal_ts_edge_closed_regions(SOURCE_1MB).regions
    incremental = scan_exact_region_witnesses(evaluation.trace, normal_regions)
    reference = scan_exact_region_witnesses(
        evaluation.trace, normal_regions, algorithm="reference"
    )
    assert len(incremental.valid_witnesses) == 25
    assert len(reference.valid_witnesses) == 25
    for incremental_region, reference_region in zip(
        incremental.regions, reference.regions
    ):
        assert incremental_region["status"] == reference_region["status"]
        if incremental_region["status"] == "valid":
            for field in _WITNESS_CERTIFICATE_FIELDS:
                assert incremental_region["witness"][field] == (
                    reference_region["witness"][field]
                )
